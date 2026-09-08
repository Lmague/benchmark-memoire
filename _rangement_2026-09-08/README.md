# _rangement_2026-09-08/ — Rangement du 2026-09-08 (sans rien jeter, sans rien casser)

Tout ce qui a été déplacé ici est **documenté**. Rien n'est supprimé.
Les chemins d'origine restés nécessaires sont des **symlinks** (fonctionnels).

## Contenu

| Sous-dossier | Contenu | Origine | Symlink laissé ? |
|---|---|---|---|
| `json_racine/` | 4 JSON épars (`analyse_summary`, `old_ft_reprobe_11cls`, `sota_screening_*`) | racine | oui (4 liens) |
| `baks/` | 4 `.bak` (`AGENTS.md.bak*`, `AGENT_MEMORY.md.bak`) | racine | non (référencés nulle part) |
| `papier_brouillons/` | 1 `.bak` du papier pré-rewrite 2026-08-27 | `paper_arctic_fm_benchmark/` | non |
| `zips_contexte/` | `context_512_valtest.zip` (3.8 G), `context_2048_valtest.zip` (3.7 G) | racine | oui (2 liens) |
| `divers_racine/` | `texput.log`, `test_agent_complexe.txt` | racine | non |

## Ce qui n'a PAS été déplacé (volontairement) et pourquoi

| Dossier | Taille | Pourquoi on n'y touche pas |
|---|---|---|
| `embeddings/` (333 fichiers) | ~12 G | chemins hardcodés dans `probe.py`, `extract.py`, tous les scripts + `registry.py` |
| `splits/`, `splits_11cls/`, `splits_spatial/` | — | `splits/` hardcodé partout ; `splits_spatial/` utilisé par datacurve |
| `configs/`, `src/` | — | importés par tous les CLIs et scripts |
| `checkpoints/` | 1.6 G | sorties `train.py` par défaut |
| `DINOv3_LoRA/`, `DINOv3_LoRA_8/`, `DINOv3_ExPLoRA_Like/`, `dinov3_vitb16_lvd_explora/` | ~8 G | embeddings/runs référencés par `registry.py` (familles `dinov3b_lora8`, ExPLoRA) |
| `LoRA_ablation/`, `lora_block_ablation/`, `lora_spatial_v2/` | ~17 G | campagnes d'ablation, runs listés dans les rapports |
| `ViT-B-16-IN-*/`, `ViTB_L_LoRA/`, `ResNet_50_Full/` | ~70 G | familles `sota_screening`, `lora_3models` du registre |
| `sota_screening/`, `runs/`, `runs_vits16/`, `ft_ssl_results/` | ~45 G | familles `ft_old`/`ft_ssl` du registre — canoniques pour le tableau maître |
| `results/` | ~14 G | sources canoniques du manuscrit (AGENTS.md §3) — ne jamais déplacer |
| `Dataset_Leo/` | 134 G | données source — ne pas déplacer |
| `_anciennes_experiences/` | 1.7 G | archives officielles (voir `ARCHIVE_INDEX.md`) |
| `paper_conference/`, `Papiers/`, `research_overview/`, `docs/` | — | documents de travail, laissés en place |

## Règle pour la suite

- Nouveau run → sous `results/<campagne>/`, jamais en vrac à la racine.
- Nouveau JSON épars → `results/`, jamais à la racine.
- Avant tout déplacement : `grep -rn "<nom>" scripts/ configs/ src/ *.py Makefile`.
