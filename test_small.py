from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import time

model_name = 'openai/whisper-small'
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')
print('Testing on', device)

processor = WhisperProcessor.from_pretrained(model_name, language='ar', task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()
model.to(device)

n_mels = model.config.num_mel_bins
max_len = 3000

dummy_input = torch.randn(1, n_mels, max_len, device=device, dtype=torch.float32)
dummy_labels = torch.tensor([[50258, 50359, 50363] + [0] * 20], device=device, dtype=torch.long)

start = time.time()
with torch.no_grad():
    outputs = model(input_features=dummy_input, labels=dummy_labels)
    loss = outputs.loss
elapsed = time.time() - start

peak = torch.cuda.max_memory_allocated() / (1024**3) if use_cuda else 0
print('Loss:', round(loss.item(), 4), 'Time:', round(elapsed, 2), 's', 'VRAM:', round(peak, 2), 'GB')