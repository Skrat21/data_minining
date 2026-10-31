#!/usr/bin/env python3
"""
arabic_data_pipeline.py - download (once) + preprocess Arabic ASR data in one script.

Datasets
  cv    Mozilla Common Voice Arabic, via the official Mozilla Data Collective SDK
  mgb2  MGB-2 via Hugging Face streaming (default MohamedRashad/mgb2-arabic), OR a local Kaldi-style dir
        NOTE: MGB-2 has its own license agreement; a community re-upload does not remove it.

Safety features
  * Nothing is downloaded if the archive / processed audio already exists.
  * Asks for confirmation before any download (skip with --yes).
  * Hard storage budget (--budget-gb, default 30) checked before every step.
  * Expensive step (audio conversion) is separate from cheap step (text
    normalization), so changing normalization never re-converts audio.
  * --status shows what exists; --dry-run shows what WOULD happen.

Note: the Common Voice licence forbids re-hosting/re-sharing the data and identifying speakers.
      Keep processed audio local; share only code and manifests of your own derived results.

Setup
  pip install librosa soundfile tqdm datacollective datasets
  export MDC_API_KEY=...        # Common Voice: key from your Mozilla Data Collective account
                                # (you must also accept the dataset terms on the website)
  huggingface-cli login         # MGB-2 on HF: accept the dataset terms first

Usage
  python arabic_data_pipeline.py --status
  python arabic_data_pipeline.py --dry-run
  python arabic_data_pipeline.py --datasets cv                       # Common Voice only
  python arabic_data_pipeline.py --datasets mgb2 --mgb2-hours 50
  python arabic_data_pipeline.py --datasets mgb2 --mgb2-kaldi-dir train=/data/mgb2/train dev=/data/mgb2/dev
  python arabic_data_pipeline.py --delete-raw                        # free ~7 GB after CV is processed
Output: <workdir>/manifests/<dataset>_<split>.jsonl  (+ audio under <workdir>/audio)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import sys
import tarfile
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

SR = 16000
GB = 1e9
GB_PER_HOUR = {"flac": 0.065, "wav": 0.115}   # approx., 16 kHz mono 16-bit
CV_DEFAULT_ID = "cmu5wom3z00ceo107guxtg4xa"   # Common Voice Scripted Speech 27.0 - Arabic (3.30 GB tar.gz)
CV_ARCHIVE_GB = 3.3
CV_AVG_CLIP_SEC = 4.2                         # average clip length listed on the v27.0 page
MGB2_HF_REPO = "MohamedRashad/mgb2-arabic"   # community upload; official QCRI/mgb2 is raw archives (see notes)


class PipelineError(Exception):
    pass


# ----------------------------------------------------------------------------
# Arabic text normalization
# ----------------------------------------------------------------------------
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
CARRIER_MAP = str.maketrans({"ؤ": "و", "ئ": "ي"})
YAA_MAP = str.maketrans({"ى": "ي"})
TA_MAP = str.maketrans({"ة": "ه"})
LOAN_MAP = str.maketrans({"ی": "ي", "ک": "ك", "پ": "ب", "چ": "ج", "ڤ": "ف", "گ": "ك"})
TAGS = re.compile(r"<[^>]*>")  


@dataclass
class NormConfig:
    alef: bool = True
    hamza_carriers: bool = True
    remove_hamza: bool = True
    yaa: bool = True
    ta_marbuta: bool = True
    keep_latin: bool = False


def normalize_arabic(text, cfg: NormConfig) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = TAGS.sub(" ", t)
    t = DIACRITICS.sub("", t).replace("\u0640", "")
    t = t.translate(DIGITS).translate(LOAN_MAP)
    if cfg.alef:
        t = t.translate(ALEF_MAP)
    if cfg.hamza_carriers:
        t = t.translate(CARRIER_MAP)
    if cfg.remove_hamza:
        t = t.replace("ء", "")
    if cfg.yaa:
        t = t.translate(YAA_MAP)
    if cfg.ta_marbuta:
        t = t.translate(TA_MAP)
    t = t.lower()
    allowed = r"\u0621-\u064A0-9" + ("a-z" if cfg.keep_latin else "")
    t = re.sub(rf"[^{allowed}\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------
# Workspace, state, storage guards
# ----------------------------------------------------------------------------
def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def guard(ws: Path, extra_gb: float, budget_gb: float, label: str):
    used = dir_size(ws) / GB
    free = shutil.disk_usage(ws).free / GB
    if used + extra_gb > budget_gb:
        raise PipelineError(
            f"{label}: needs ~{extra_gb:.1f} GB but workspace already uses {used:.1f} GB "
            f"of the {budget_gb:.0f} GB budget. Free space (--delete-raw), lower the hours, or raise --budget-gb.")
    if free < extra_gb + 2:
        raise PipelineError(f"{label}: needs ~{extra_gb:.1f} GB but only {free:.1f} GB free on disk.")


def confirm(args, msg: str):
    if args.yes:
        return
    if input(f"{msg} Continue? [y/N] ").strip().lower() != "y":
        raise PipelineError("cancelled by user (nothing downloaded)")


def load_state(ws: Path) -> dict:
    p = ws / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(ws: Path, st: dict):
    (ws / "state.json").write_text(json.dumps(st, indent=2))


# ----------------------------------------------------------------------------
# Audio conversion
# ----------------------------------------------------------------------------
def _write_audio(y, dst: Path):
    import soundfile as sf
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, SR, subtype="PCM_16")


def _convert_job(job):
    """job = (src, dst, start, end). Returns (ok, duration | error)."""
    src, dst, start, end = job
    try:
        import librosa
        import soundfile as sf
        dst = Path(dst)
        if dst.exists() and dst.stat().st_size > 0:
            return True, sf.info(str(dst)).duration
        off = float(start) if start is not None else 0.0
        dur = (float(end) - off) if end is not None else None
        y, _ = librosa.load(src, sr=SR, mono=True, offset=off, duration=dur)
        _write_audio(y, dst)
        return True, len(y) / SR
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def convert_many(jobs, workers):
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(_convert_job, jobs, chunksize=32), total=len(jobs), desc="audio"))


def write_index(path: Path, records, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_index(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------------
# Common Voice
# ----------------------------------------------------------------------------
def find_archive(raw: Path):
    for p in sorted(raw.glob("*")):
        if p.is_file() and p.suffix in (".gz", ".tgz", ".tar", ".zip"):
            return p
    return None


def safe_extract(arc: Path, dest: Path):
    dest = dest.resolve()

    def ok(name):
        return str((dest / name).resolve()).startswith(str(dest))

    if arc.suffix == ".zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(dest, members=[m for m in z.namelist() if ok(m)])
    else:
        with tarfile.open(arc) as t:
            t.extractall(dest, members=(m for m in t if ok(m.name)))


def cv_info(args):
    """Look up name/size on Mozilla Data Collective (public endpoint). Falls back to defaults offline."""
    try:
        from datacollective import get_dataset_details
        info = get_dataset_details(args.cv_id)
        size = (info.sizeBytes or 0) / GB or CV_ARCHIVE_GB
        print(f"[cv] dataset: {info.name} | {size:.2f} GB | {info.filename}")
        return size
    except Exception as e:
        print(f"[cv] could not look up dataset details ({type(e).__name__}); assuming ~{CV_ARCHIVE_GB} GB")
        return CV_ARCHIVE_GB


def run_cv(ws: Path, args, st: dict, cfg: NormConfig):
    raw = ws / "raw" / "cv"
    ext = raw / "extracted"
    index = ws / "index" / "cv.jsonl"
    cvs = st.setdefault("cv", {})
    ext_fmt = args.audio_format

    if cvs.get("stage_a_done") and index.exists():
        print(f"[cv] audio already processed ({len(read_index(index))} clips) -> no download, no conversion")
    else:
        arc = find_archive(raw)
        has_partial = any(raw.glob("*.checksum"))
        if arc and not has_partial:
            print(f"[cv] archive already downloaded: {arc.name} -> skipping download")
        else:
            size_gb = cv_info(args)
            if args.dry_run:
                print(f"[cv] WOULD download ~{size_gb:.1f} GB archive via MDC SDK"); return
            if not os.environ.get("MDC_API_KEY"):
                raise PipelineError("MDC_API_KEY not set. Create a key in your Mozilla Data Collective account, "
                                    "accept the Common Voice dataset terms, then `export MDC_API_KEY=...`")
            guard(ws, size_gb * 2.1, args.budget_gb, "cv download+extract")
            confirm(args, f"[cv] About to download ~{size_gb:.1f} GB.")
            raw.mkdir(parents=True, exist_ok=True)
            from datacollective import download_dataset
            arc = Path(download_dataset(args.cv_id, download_directory=str(raw)))
            print(f"[cv] downloaded {arc}")

        if not (ext / ".done").exists():
            if args.dry_run:
                print("[cv] WOULD extract archive"); return
            guard(ws, CV_ARCHIVE_GB * 1.1, args.budget_gb, "cv extract")
            print("[cv] extracting (one time)...")
            shutil.rmtree(ext, ignore_errors=True)
            ext.mkdir(parents=True)
            try:
                safe_extract(arc, ext)
            except (EOFError, tarfile.TarError, zipfile.BadZipFile) as e:
                raise PipelineError(f"archive looks corrupt/incomplete ({e}). Delete {arc} and rerun.")
            (ext / ".done").touch()

        tsv0 = next(ext.rglob("train.tsv"), None)
        if tsv0 is None:
            raise PipelineError(f"train.tsv not found under {ext}")
        root = tsv0.parent
        rng = random.Random(args.seed)
        jobs, meta = [], []
        for split in ("train", "dev", "test"):
            tsv = root / f"{split}.tsv"
            if not tsv.exists():
                continue
            with open(tsv, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
            if args.cv_max_clips and len(rows) > args.cv_max_clips:
                rows = rng.sample(rows, args.cv_max_clips)
            for r in rows:
                uid = "cv_" + re.sub(r"\.\w+$", "", r["path"])
                dst = ws / "audio" / "cv" / f"{uid}.{ext_fmt}"
                jobs.append((str(root / "clips" / r["path"]), str(dst), None, None))
                meta.append({"id": uid, "split": split, "audio": str(dst.relative_to(ws)),
                             "text_raw": r.get("sentence", r.get("text", "")),
                             "speaker": r.get("client_id", ""), "gender": r.get("gender", ""),
                             "age": r.get("age", ""), "accent": r.get("accents", r.get("accent", "")),
                             "variant": r.get("variant", ""), "sentence_id": r.get("sentence_id", ""),
                             "sentence_domain": r.get("sentence_domain", ""),
                             "up_votes": r.get("up_votes", ""), "down_votes": r.get("down_votes", "")})
        est = len(jobs) * CV_AVG_CLIP_SEC / 3600 * GB_PER_HOUR[ext_fmt]
        print(f"[cv] {len(jobs)} clips to convert (~{est:.1f} GB as {ext_fmt})")
        if args.dry_run:
            return
        guard(ws, est, args.budget_gb, "cv audio")
        results = convert_many(jobs, args.workers)
        out, bad = [], 0
        for m, (ok, val) in zip(meta, results):
            if ok:
                m["duration"] = round(val, 3)
                out.append(m)
            else:
                bad += 1
        if bad:
            print(f"[cv] {bad} clips failed to decode and were skipped (if all failed, install ffmpeg)")
        if not out:
            raise PipelineError("no audio could be converted")
        write_index(index, out)
        cvs.update(stage_a_done=True, clips=len(out), audio_format=ext_fmt, max_clips=args.cv_max_clips)
        save_state(ws, st)

    if args.delete_raw and raw.exists():
        sz = dir_size(raw) / GB
        shutil.rmtree(raw)
        cvs["raw_deleted"] = True
        save_state(ws, st)
        print(f"[cv] deleted raw download/extracted files (freed {sz:.1f} GB). Processed data is kept.")
    finalize(ws, "cv", args, cfg)


# ----------------------------------------------------------------------------
# MGB-2
# ----------------------------------------------------------------------------
TEXT_KEYS = ("text", "sentence", "transcription", "transcript", "normalized_text", "label")
META_KEYS = ("speaker", "speaker_id", "program", "show", "episode", "dialect", "category")


def read_kv(path: Path):
    res = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split(maxsplit=1)
            if len(p) == 2:
                res[p[0]] = p[1]
    return res


def mgb2_kaldi(ws: Path, args, st: dict, mg: dict):
    """Local Kaldi-style dirs (wav.scp, segments, text). Format: split=path"""
    index = ws / "index" / "mgb2.jsonl"
    rng = random.Random(args.seed)
    jobs, meta, total_gb = [], [], 0.0
    for spec in args.mgb2_kaldi_dir:
        split, _, path = spec.partition("=")
        d = Path(path)
        wav, text = read_kv(d / "wav.scp"), read_kv(d / "text")
        utt2spk = read_kv(d / "utt2spk") if (d / "utt2spk").exists() else {}
        for k, v in wav.items():
            if v.strip().endswith("|"):
                raise PipelineError("wav.scp has pipe commands; convert recordings to plain files first")
        segs = []
        if (d / "segments").exists():
            for line in open(d / "segments", encoding="utf-8"):
                p = line.split()
                if len(p) == 4 and p[0] in text and p[1] in wav:
                    segs.append((p[0], wav[p[1]], float(p[2]), float(p[3])))
        else:
            segs = [(u, wav[u], None, None) for u in text if u in wav]
        rng.shuffle(segs)
        budget_h = args.mgb2_hours if split == "train" else args.mgb2_eval_hours
        acc = 0.0
        for utt, src, s, e in segs:
            if acc >= budget_h * 3600:
                break
            acc += (e - s) if s is not None else 10.0
            uid = f"mgb2_{split}_{utt}"
            dst = ws / "audio" / "mgb2" / f"{uid}.{args.audio_format}"
            jobs.append((src, str(dst), s, e))
            meta.append({"id": uid, "split": split, "audio": str(dst.relative_to(ws)),
                         "text_raw": text[utt], "speaker": utt2spk.get(utt, ""),
                         "gender": "", "age": "", "accent": ""})
    est = len(jobs) and (sum((j[3] - j[2]) if j[2] is not None else 10 for j in jobs) / 3600
                         * GB_PER_HOUR[args.audio_format])
    print(f"[mgb2] {len(jobs)} segments selected (~{est:.1f} GB)")
    if args.dry_run:
        return
    guard(ws, est, args.budget_gb, "mgb2 audio")
    results = convert_many(jobs, args.workers)
    out = []
    for m, (ok, val) in zip(meta, results):
        if ok:
            m["duration"] = round(val, 3)
            out.append(m)
    write_index(index, out)
    mg.update(stage_a_done=True, source="kaldi", segments=len(out), audio_format=args.audio_format)
    save_state(ws, st)


def mgb2_hf(ws: Path, args, st: dict, mg: dict):
    """Stream MGB-2 from Hugging Face; only the hours we need are downloaded."""
    import numpy as np
    import librosa
    from datasets import Audio, get_dataset_split_names, load_dataset
    from tqdm import tqdm

    repo = args.mgb2_repo
    # Try the repo as-is, then the auto-converted parquet branch (for script-based repos)
    revisions = [args.mgb2_revision] if args.mgb2_revision else [None, "refs/convert/parquet"]
    avail, revision, last_err = None, None, None
    for rev in revisions:
        try:
            avail = get_dataset_split_names(repo, revision=rev)
            revision = rev
            break
        except Exception as e:
            last_err = e
    if not avail:
        raise PipelineError(f"cannot open {repo}: {type(last_err).__name__}: {last_err}\n"
                            "  -> open the dataset page on huggingface.co, accept its terms if asked, then run "
                            "`hf auth login` (or set HF_TOKEN). If it still fails, get MGB-2 from "
                            "arabicspeech.org/mgb2 and use --mgb2-kaldi-dir.")
    print(f"[mgb2] {repo} (revision={revision or 'main'}) splits: {avail}")

    low = {s.lower(): s for s in avail}
    dev_hf = next((low[k] for k in ("dev", "validation", "valid", "val") if k in low), None)
    tasks = []   # (hf_split, [(logical_name, hours), ...])
    if "train" in low and (dev_hf or "test" in low):
        tasks.append((low["train"], [("train", args.mgb2_hours)]))
        if dev_hf:
            tasks.append((dev_hf, [("dev", args.mgb2_eval_hours)]))
        if "test" in low:
            tasks.append((low["test"], [("test", args.mgb2_eval_hours)]))
    else:
        only = low.get("train", avail[0])
        print(f"[mgb2] only split '{only}' available -> carving dev/test out of it "
              "(dev/test will share programs/speakers with train: results are optimistic and NOT "
              "comparable with published MGB-2 numbers)")
        tasks.append((only, [("dev", args.mgb2_eval_hours), ("test", args.mgb2_eval_hours),
                             ("train", args.mgb2_hours)]))

    index = ws / "index" / "mgb2.jsonl"
    done_splits = mg.setdefault("splits_done", {})
    for hf_split, targets in tasks:
        if done_splits.get(hf_split):
            print(f"[mgb2] split '{hf_split}' already done -> skipping")
            continue
        existing = [r for r in read_index(index) if r.get("hf_split") == hf_split]
        got = {name: 0.0 for name, _ in targets}
        for r in existing:
            if r["split"] in got:
                got[r["split"]] += r["duration"] / 3600
        todo_h = sum(max(h - got[n], 0) for n, h in targets)
        need_gb = todo_h * GB_PER_HOUR[args.audio_format] * 1.1
        print(f"[mgb2] '{hf_split}': targets {dict(targets)} h, have { {k: round(v, 2) for k, v in got.items()} } h "
              f"(~{need_gb:.1f} GB more)")
        if args.dry_run:
            continue
        if todo_h <= 0:
            done_splits[hf_split] = True
            save_state(ws, st)
            continue
        guard(ws, need_gb, args.budget_gb, f"mgb2 {hf_split}")
        ds = load_dataset(repo, split=hf_split, streaming=True, revision=revision)
        feats = ds.features or {}
        audio_col = next((k for k, v in feats.items() if isinstance(v, Audio)), "audio")
        try:
            ds = ds.cast_column(audio_col, Audio(decode=False))
        except Exception:
            pass
        ds = ds.shuffle(seed=args.seed, buffer_size=1000).skip(len(existing))
        n, new, exhausted = len(existing), [], True
        bar = tqdm(total=todo_h * 3600, unit="s", desc=f"mgb2 {hf_split}")
        for ex in ds:
            cur = next((nm for nm, h in targets if got[nm] < h), None)
            if cur is None:
                exhausted = False
                break
            text = next((ex[k] for k in TEXT_KEYS if k in ex and isinstance(ex[k], str)), None)
            if text is None:
                raise PipelineError(f"no text column found; columns are: {list(ex.keys())}. "
                                    f"Add the right name to TEXT_KEYS in the script.")
            a = ex[audio_col]
            try:
                if a.get("bytes"):
                    y, _ = librosa.load(io.BytesIO(a["bytes"]), sr=SR, mono=True)
                elif a.get("array") is not None:
                    y = np.asarray(a["array"], dtype="float32")
                    if a.get("sampling_rate", SR) != SR:
                        y = librosa.resample(y, orig_sr=a["sampling_rate"], target_sr=SR)
                else:
                    y, _ = librosa.load(a["path"], sr=SR, mono=True)
            except Exception:
                continue
            uid = f"mgb2_{hf_split}_{n:07d}"
            dst = ws / "audio" / "mgb2" / f"{uid}.{args.audio_format}"
            _write_audio(y, dst)
            rec = {"id": uid, "split": cur, "hf_split": hf_split, "audio": str(dst.relative_to(ws)),
                   "duration": round(len(y) / SR, 3), "text_raw": text,
                   "speaker": "", "gender": "", "age": "", "accent": ""}
            for k in META_KEYS:
                if k in ex and isinstance(ex[k], (str, int)):
                    rec["speaker" if k.startswith("speaker") else k] = str(ex[k])
            new.append(rec)
            n += 1
            got[cur] += rec["duration"] / 3600
            bar.update(rec["duration"])
            if len(new) >= 200:
                write_index(index, new, "a")
                new = []
        bar.close()
        write_index(index, new, "a")
        if exhausted or all(got[nm] >= h for nm, h in targets):
            done_splits[hf_split] = True
        save_state(ws, st)
    if not args.dry_run and tasks and all(done_splits.get(t[0]) for t in tasks):
        mg.update(stage_a_done=True, source="hf", repo=repo, audio_format=args.audio_format)
        save_state(ws, st)


def run_mgb2(ws: Path, args, st: dict, cfg: NormConfig):
    mg = st.setdefault("mgb2", {})
    index = ws / "index" / "mgb2.jsonl"
    if mg.get("stage_a_done") and index.exists():
        print(f"[mgb2] already processed ({len(read_index(index))} segments) -> no download")
    elif args.mgb2_kaldi_dir:
        mgb2_kaldi(ws, args, st, mg)
    else:
        if not args.dry_run:
            confirm(args, f"[mgb2] About to stream ~{args.mgb2_hours + 2 * args.mgb2_eval_hours} h from "
                          f"{args.mgb2_repo} (~{(args.mgb2_hours + 2 * args.mgb2_eval_hours) * GB_PER_HOUR[args.audio_format]:.1f} GB on disk).")
        mgb2_hf(ws, args, st, mg)
    if index.exists():
        finalize(ws, "mgb2", args, cfg)


# ----------------------------------------------------------------------------
# Cheap step: normalization + filtering -> final manifests (always safe to rerun)
# ----------------------------------------------------------------------------
def finalize(ws: Path, name: str, args, cfg: NormConfig):
    recs = read_index(ws / "index" / f"{name}.jsonl")
    if not recs:
        return
    out_dir = ws / "manifests"
    out_dir.mkdir(exist_ok=True)
    by_split = {}
    for r in recs:
        r["text_norm"] = normalize_arabic(r["text_raw"], cfg)
        r["audio_filepath"] = str(ws / r.pop("audio"))
        d = r["duration"]
        if not r["text_norm"] or d < args.min_dur or d > args.max_dur:
            continue
        r["n_words"] = len(r["text_norm"].split())
        r["n_chars"] = len(r["text_norm"])
        r["chars_per_sec"] = round(r["n_chars"] / d, 2)
        by_split.setdefault(r["split"], []).append(r)
    for split, rows in by_split.items():
        p = out_dir / f"{name}_{split}.jsonl"
        write_index(p, rows)
        hrs = sum(r["duration"] for r in rows) / 3600
        print(f"[{name}] {split}: {len(rows)} utts, {hrs:.2f} h -> {p.name}")


# ----------------------------------------------------------------------------
def print_status(ws: Path, args):
    st = load_state(ws)
    print(f"workspace: {ws}")
    for sub in ("raw", "audio", "index", "manifests"):
        print(f"  {sub:10s} {dir_size(ws / sub) / GB:6.2f} GB")
    used = dir_size(ws) / GB
    print(f"  TOTAL      {used:6.2f} GB of {args.budget_gb:.0f} GB budget | free on disk: "
          f"{shutil.disk_usage(ws).free / GB:.1f} GB")
    for name in ("cv", "mgb2"):
        s = st.get(name, {})
        print(f"  [{name}] processed={bool(s.get('stage_a_done'))} raw_deleted={bool(s.get('raw_deleted'))} {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default="arabic_data")
    ap.add_argument("--datasets", nargs="+", default=["cv", "mgb2"], choices=["cv", "mgb2"])
    ap.add_argument("--budget-gb", type=float, default=30)
    ap.add_argument("--audio-format", default="flac", choices=["flac", "wav"])
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--yes", action="store_true", help="skip download confirmation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--delete-raw", action="store_true", help="after processing, delete raw CV archive/extract")
    ap.add_argument("--redo", nargs="+", choices=["cv", "mgb2"], default=[],
                    help="discard processed audio+index of a dataset and rebuild it")
    ap.add_argument("--cv-id", default=CV_DEFAULT_ID)
    ap.add_argument("--cv-max-clips", type=int, default=0, help="max clips per split (0 = all)")
    ap.add_argument("--mgb2-repo", default=MGB2_HF_REPO)
    ap.add_argument("--mgb2-revision", default=None, help="HF revision (default: auto, tries main then refs/convert/parquet)")
    ap.add_argument("--mgb2-hours", type=float, default=50, help="hours of train data")
    ap.add_argument("--mgb2-eval-hours", type=float, default=5, help="hours each for dev and test")
    ap.add_argument("--mgb2-kaldi-dir", nargs="+", default=[], metavar="SPLIT=PATH")
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--keep-latin", action="store_true")
    ap.add_argument("--no-ta-marbuta-norm", action="store_true")
    ap.add_argument("--no-yaa-norm", action="store_true")
    args = ap.parse_args()

    ws = Path(args.workdir).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status(ws, args)
        return
    cfg = NormConfig(keep_latin=args.keep_latin, ta_marbuta=not args.no_ta_marbuta_norm, yaa=not args.no_yaa_norm)
    st = load_state(ws)

    for name in args.redo:
        print(f"[{name}] --redo: removing processed audio and index (raw files are kept)")
        shutil.rmtree(ws / "audio" / name, ignore_errors=True)
        (ws / "index" / f"{name}.jsonl").unlink(missing_ok=True)
        st.pop(name, None)
        save_state(ws, st)

    for name in args.datasets:
        try:
            (run_cv if name == "cv" else run_mgb2)(ws, args, st, cfg)
        except PipelineError as e:
            print(f"[{name}] STOPPED: {e}")
        except KeyboardInterrupt:
            print(f"\n[{name}] interrupted - progress is kept; rerun to resume.")
            sys.exit(1)
    print_status(ws, args)


if __name__ == "__main__":
    main()