#!/usr/bin/env python3
"""Mesure le temps d'inférence CPU (Ryzen 7 8845HS, 8 threads, batch=1) des modèles
de contexte DINOv3-B : R1 (tuile seule) et R2 (tuile + contexte 1024x1024)."""
import os, sys, time, torch
ROOT = '/home/erazal/Documents/Mémoire'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'rapport'))
from src.models import merge_lora_state_dict
from transformers import DINOv3ViTConfig, DINOv3ViTModel
torch.set_num_threads(8)

def build(ckpt):
    sd = torch.load(os.path.join(ROOT, ckpt), map_location='cpu', weights_only=False)['model_state_dict']
    bs = {k[len('backbone.'):]: v for k, v in sd.items()
          if k.startswith('backbone.') and not k[len('backbone.'):].startswith('head.')}
    bs = merge_lora_state_dict(bs)
    cfg = DINOv3ViTConfig(image_size=224, patch_size=16, hidden_size=768,
                          num_hidden_layers=12, num_attention_heads=12,
                          num_register_tokens=4, intermediate_size=3072)
    # sdpa (inférence réelle, pas de poids d'attention)
    m = DINOv3ViTModel(cfg); m.load_state_dict(bs); m.eval()
    return m

def timeit(fn, reps=3):
    ts = []
    for _ in range(reps + 1):
        t0 = time.time(); fn(); ts.append(time.time() - t0)
    return min(ts[1:])

R1 = "results/context_distill/checkpoints/dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100_seed0_best.pth"
R2 = "results/context_distill/checkpoints/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed0_best.pth"

m1 = build(R1); m2 = build(R2)
tile = torch.randn(1, 3, 224, 224)
ctx  = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    t_tile = timeit(lambda: m1(pixel_values=tile))
    t_tile2 = timeit(lambda: m2(pixel_values=tile))
    t_ctx = timeit(lambda: m2(pixel_values=ctx))
print(f"R1 tuile 224      : {t_tile*1000:.0f} ms")
print(f"R2 tuile 224      : {t_tile2*1000:.0f} ms")
print(f"R2 contexte 1024  : {t_ctx*1000:.0f} ms")
print(f"R2 total          : {(t_tile2+t_ctx)*1000:.0f} ms")
