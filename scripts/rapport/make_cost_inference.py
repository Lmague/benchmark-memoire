#!/usr/bin/env python3
"""Table coût/performance pour TOUS les modèles du benchmark.

Params, FLOPs (formule ViT du registre), F1 canonique (11 cls), et temps d'inférence
CPU mesuré/estimé (Ryzen 7 8845HS, 8 threads, batch=1). Les modèles partageant une
architecture (ViT-S/B/L/H+, ResNet-50) ont le même coût d'inférence ; LoRA n'ajoute
aucune couche → coût = celui du backbone gelé.
"""
import os, sys, json
ROOT='/home/erazal/Documents/Mémoire'
sys.path.insert(0, os.path.join(ROOT,'scripts')); sys.path.insert(0, os.path.join(ROOT,'scripts','rapport'))
from make_lora_rank_tables import vit_gflops

# arch: (params_M, gflops, cpu_ms, estime?)
ARCH={
 "vit_s":  (21.6, vit_gflops(12,384,6,4,False),  24, False),
 "vit_b":  (85.7, vit_gflops(12,768,12,4,False), 82, False),
 "vit_l":  (303.1,vit_gflops(24,1024,16,4,False),646,False),
 "vit_hp": (840.6,vit_gflops(32,1280,20,6,True), 1062,False),
 "resnet50":(25.6, 4.1, 25, False),
}
CPU_MS={"vit_s":24,"vit_b":82,"vit_l":646,"vit_hp":1062,"resnet50":25}
# Modèles de contexte DINOv3-B (LoRA r=2, blocs 6-11) : coût propre.
# R1/R3 = tuile seule à l'inférence (1 forward) ; R2 = tuile + contexte 1024×1024 (2 forwards).
# GFLOPs = formule ViT (n_tokens) ; CPU ms = mesuré (Ryzen 7 8845HS, 8 threads, batch=1, sdpa).
CTX={
 "ctxdistill_dA_tL":  ("Contexte R1 (distil. A, tuile)",
   vit_gflops(12,768,12,4,False,197), 82),
 "ctxdistill_dA_tEMA":("Contexte R3 (distil. A, EMA)",
   vit_gflops(12,768,12,4,False,197), 82),
 "ctxdistill_dB_tL":  ("Contexte R2 (distil. B, fusion)",
   vit_gflops(12,768,12,4,False,197)+vit_gflops(12,768,12,4,False,4101), 3956),
}
def vitb_imagenet_ms(): return 118  # torchvision ViT-B/16 (ImageNet) mesuré
def arch_of(model):
    l=model
    if 'resnet50' in l or 'resnet' in l: return 'resnet50'
    if 'vits16' in l: return 'vit_s'
    if 'vith16plus' in l: return 'vit_hp'
    if 'vitl16' in l: return 'vit_l'
    if 'vitb16' in l or 'simdinov2_vitb16' in l: return 'vit_b'
    if 'simdinov2_vitl16' in l: return 'vit_l'
    return None

# F1 canonique + display depuis la table maître
merged=json.load(open(os.path.join(ROOT,'results/all_models_canonical_merged.json')))
F={m['model']:m for m in merged['models']}
# modèles à inclure (canoniques, non dépréciés/exclus) -> (label, arch)
INCLUDE=[
 ("dinov3_vits16_lvd","DINOv3 ViT-S/16 gelé","vit_s"),
 ("dinov3_vits16_lvd_lora_r8","DINOv3 ViT-S/16 LoRA r=8","vit_s"),
 ("resnet50_imagenet","ResNet-50 IN-1k","resnet50"),
 ("resnet50_fulft_sota","ResNet-50 IN-FT","resnet50"),
 ("vitb16_imagenet","ViT-B/16 IN-1k","vit_b"),
 ("dinov3_vitb16_lvd","DINOv3 ViT-B/16 gelé","vit_b"),
 ("dinov3_vitb16_lvd_lora_r8","DINOv3 ViT-B/16 LoRA r=8","vit_b"),
 ("dinov3_vitb16_lvd_full","DINOv3 ViT-B/16 Full","vit_b"),
 ("dinov3_vitb16_lvd_mhsa","DINOv3 ViT-B/16 MHSA","vit_b"),
 ("simdinov2_vitb16","SimDINOv2 ViT-B/16 gelé","vit_b"),
 ("simdinov2_vitb16_lora","SimDINOv2 ViT-B/16 LoRA r=8","vit_b"),
 ("dinov3_vitl16_lvd","DINOv3 ViT-L/16 gelé","vit_l"),
 ("dinov3_vitl16_lvd_lora","DINOv3 ViT-L/16 LoRA r=8","vit_l"),
 ("simdinov2_vitl16","SimDINOv2 ViT-L/16 gelé","vit_l"),
 ("simdinov2_vitl16_lora","SimDINOv2 ViT-L/16 LoRA r=8","vit_l"),
 ("dinov3_vitl16_sat","DINOv3 ViT-L/16 SAT","vit_l"),
 ("satmae_vitl16","SatMAE ViT-L/16","vit_l"),
 ("scalemae_vitl16","ScaleMAE ViT-L/16","vit_l"),
 ("dinov3_vith16plus_lvd","DINOv3 ViT-H+/16 gelé","vit_hp"),
 ("ctxdistill_dA_tL","Contexte R1 (distil. A, tuile)","vit_b"),
 ("ctxdistill_dA_tEMA","Contexte R3 (distil. A, EMA)","vit_b"),
 ("ctxdistill_dB_tL","Contexte R2 (distil. B, fusion)","vit_b"),
]

def fmt(x, d=3):
    return f"{x:.{d}f}".replace('.', ',') if x is not None else '---'

rows=[]
for key,label,arch in INCLUDE:
    f1=F.get(key,{}).get('f1_linear_probe')
    pm,gf,ms,est=ARCH[arch]
    if f1 is None: continue
    if key in CTX:  # modèle de contexte : coût propre (formule + mesuré)
        label = CTX[key][0]
        gf, ms = CTX[key][1], CTX[key][2]
        est = gf > 300   # seule la fusion + contexte (R2) diffère du backbone
    if key=="vitb16_imagenet": ms=118  # ViT-B torchvision (ImageNet) réel
    rows.append((f1, label, arch, pm, gf, ms, est))
rows.sort(reverse=True)

lines=["\\begin{longtable}{@{}lrrrrrr@{}}",
 "\\caption{Coût d'inférence et efficacité des modèles du benchmark (probe canonique, 11~classes). Params = total du modèle ; pour les variantes LoRA, seuls \\texttt{r8a16} sont entraînables ($\\sim$0,3~\\% des params). FLOPs = coût d'inférence (formule ViT, 197~tokens). Temps CPU mesuré sur \\textbf{AMD Ryzen~7 8845HS} (8 threads, batch=1) ; H+ mesuré. F1/params $\\times100$ et F1/GFLOP $\\times100$.}\\label{tab:cost_inference}\\\\",
 "\\toprule","Modèle & Params (M) & GFLOPs & F1 & CPU ms/img & F1/params & F1/GFLOP \\\\","\\midrule",
 "\\endfirsthead","\\toprule","Modèle & Params (M) & GFLOPs & F1 & CPU ms/img & F1/params & F1/GFLOP \\\\","\\midrule","\\endhead","\\bottomrule","\\endfoot"]
for f1,label,arch,pm,gf,ms,est in rows:
    f1p=100*f1/pm; f1g=100*f1/gf
    est_t=(" (est., + contexte 1024 à l'inf.)" if gf > 300 else (" (est.)" if est else ""))
    lines.append(f"{label} & {pm:.1f} & {gf:.1f} & \\textbf{{{f1:.4f}}} & {ms:.0f}{est_t} & {f1p:.3f} & {f1g:.3f} \\\\")
lines.append("\\bottomrule\\end{longtable}")
lines.append("\\par\\vspace{2pt}\\footnotesize LoRA n'ajoute aucune couche : l'inf\\'erence d'un mod\\`ele adapt\\'e vaut celle de son backbone gel\\'e. CPU = Ryzen 7 8845HS (8 threads, PyTorch CPU, batch=1). H+ estim\\'e via le rapport de FLOPs (230,4~G vs 61,4~G du ViT-L).")
tex="\n".join(lines)
os.makedirs(os.path.join(ROOT,'rapport_bouguessa/tables'),exist_ok=True)
open(os.path.join(ROOT,'rapport_bouguessa/tables/t_cost_inference.tex'),'w').write(tex)
print("t_cost_inference.tex écrit,", len(rows), "modèles")
