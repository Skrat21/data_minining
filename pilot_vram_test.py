#!/usr/bin/env python3
"""Pilot script for Phase 2: Test Whisper model feasibility on CPU/GPU."""
from __future__ import annotations

import argparse
import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def test_model(model_name: str, batch_size: int = 1):
    """Test if a Whisper model can load and run on available hardware."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Testing {model_name} on {'CUDA' if use_cuda else 'CPU'}")
    
    try:
        print(f"  Loading processor and model...")
        processor = WhisperProcessor.from_pretrained(model_name, language="ar", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = model.to(device)
        model.eval()
        
        n_mels = model.config.num_mel_bins
        max_len = 3000
        
        print(f"  Creating dummy input (batch_size={batch_size})...")
        dummy_input = torch.randn(batch_size, n_mels, max_len, device=device, dtype=torch.float16 if use_cuda else torch.float32)
        dummy_labels = torch.tensor([[50258, 50359, 50363] + [0] * 20] * batch_size, device=device, dtype=torch.long)
        
        start = time.time()
        with torch.no_grad():
            outputs = model(input_features=dummy_input, labels=dummy_labels)
            loss = outputs.loss
        elapsed = time.time() - start
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if use_cuda else 0
        print(f"  ✓ SUCCESS - Loss: {loss.item():.4f}, Time: {elapsed:.2f}s, VRAM: {peak_vram:.2f}GB")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED - {type(e).__name__}: {str(e)[:100]}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Whisper model feasibility")
    parser.add_argument("--model", type=str, default="openai/whisper-small", help="Whisper model name")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    args = parser.parse_args()
    
    success = test_model(args.model, args.batch_size)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()