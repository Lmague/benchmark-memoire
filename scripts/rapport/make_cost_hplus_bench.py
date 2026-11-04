#!/usr/bin/env python3
"""Mesure le temps d'inférence CPU (Ryzen 7 8845HS, 8 threads, batch=1) de DINOv3 ViT-H+/16.
Reconstruit via timm (vit_huge_plus_patch16_dinov3) — poids téléchargés à la volée.
"""
import os, time, torch, timm
torch.set_num_threads(8)
t0 = time.time()
m = timm.create_model('vit_huge_plus_patch16_dinov3', pretrained=True)
m.eval()
print('model loaded in %.0fs, params %.1fM' % (time.time()-t0, sum(p.numel() for p in m.parameters())/1e6))
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    _ = m(x)  # warmup
    ts = []
    for _ in range(4):
        t = time.time(); _ = m(x); ts.append(time.time()-t)
print('H+ CPU ms (224x224, batch=1, 8 threads): %.0f ms  (best of 4)' % (min(ts)*1000))
print('GFLOPs (formule): 230.4')
