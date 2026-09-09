#!/usr/bin/env python3
"""Reconcile les inputs requis du head-sweep avec ce qui EST DÉJÀ sur Narval.

Contexte : `prepare_head_sweep_push.py` liste 522 fichiers (~12,7 Go) que le
tile sweep lit. Les runs ont été produits sur Narval : une grande partie de ces
embeddings y dort déjà ($SCRATCH/sota_screening, $SCRATCH/context_distill,
$SCRATCH/simB_stageA, ...). Les renvoyer serait du gaspillage.

Piège traité : les fichiers s'appellent tous train.npy/val.npy/test.npy — le
nom ne suffit PAS, et la taille non plus (deux modèles de même dim ont un
train.npy de taille identique mais de contenu différent). Fingerprint utilisé :
md5 des 64 PREMIERS et 64 DERNIERS Ko (le début du fichier ndarray diffère dès
la première classe ; risque de collision résiduelle négligeable).

Flux (lancé par check_narval_inputs.sh, côté local) :
  1. lit /tmp/head_sweep_manifest.txt (chemins relatifs dépôt) + leurs tailles ;
  2. lit /tmp/narval_npy.txt (l'inventaire "chemin|taille" produit par le ssh find) ;
  3. candidats = même basename + même taille ;
  4. fingerprint local des fichiers requis → /tmp/local_fp.txt ;
     fingerprint remote des candidats (une seule session ssh) ;
  5. écrit :
       /tmp/head_sweep_link.sh      — ln -s sur Narval (candidate → miroir
                                      $SCRATCH/head_sweep_inputs/<relpath>) ;
       /tmp/head_sweep_missing.txt  — manifeste rsync des fichiers absents ;
     et imprime le rapport (LIEN / DEJA_LA / MANQUANT + Mo économisés).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = "/lustre07/scratch/lmague/head_sweep_inputs"


def fingerprint(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
        try:
            f.seek(-65536, os.SEEK_END)
        except OSError:
            f.seek(0)
        h.update(f.read(65536))
    return h.hexdigest()


def main() -> None:
    manifest = [l.strip() for l in open("/tmp/head_sweep_manifest.txt") if l.strip()]
    inv = {}
    for line in open("/tmp/narval_npy.txt"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        path, size = line.rsplit("|", 1)
        inv.setdefault(os.path.basename(path), []).append((path, int(size)))

    # candidats : même basename + même taille
    cands = {}      # relpath -> liste de chemins remote plausibles
    missing_now = []
    for rel in manifest:
        lp = os.path.join(ROOT, rel)
        if not os.path.exists(lp):
            continue  # ex. json de résultats déjà épurés
        size = os.path.getsize(lp)
        hits = [p for p, s in inv.get(os.path.basename(rel), []) if s == size]
        if hits:
            cands[rel] = hits
        else:
            missing_now.append(rel)

    # fingerprints locaux des requis qui ont un candidat
    print(f"[reconcile] {len(cands)} fichiers ont un candidat sur Narval "
          f"(sur {len(manifest)}) — fingerprint local…", flush=True)
    local_fp = {rel: fingerprint(os.path.join(ROOT, rel)) for rel in cands}

    # fingerprints remote : une seule session ssh, liste de tous les candidats
    remote_files = sorted({p for hits in cands.values() for p in hits})
    script = "; ".join(
        f'printf "%s|" "{p}"; ( dd if="{p}" bs=65536 count=1 2>/dev/null; '
        f'tail -c 65536 "{p}" 2>/dev/null ) | md5sum | cut -d" " -f1'
        for p in remote_files)
    print(f"[reconcile] fingerprint remote de {len(remote_files)} candidats…", flush=True)
    out = subprocess.run(["ssh", sys.argv[1] if len(sys.argv) > 1 else "narval",
                          "bash -s"], input=script, capture_output=True, text=True)
    if out.returncode != 0:
        print("[reconcile] ERREUR ssh :", out.stderr[:400]); sys.exit(1)
    remote_fps = {}
    for line in out.stdout.splitlines():
        if "|" in line:
            p, fp = line.split("|", 1)
            remote_fps[p.strip()] = fp.strip()

    link_cmds, matched_bytes = [], 0
    truly_missing = []
    already = []
    for rel in manifest:
        lp = os.path.join(ROOT, rel)
        if not os.path.exists(lp):
            continue
        dest = f"{MIRROR}/{rel}"
        hits = cands.get(rel, [])
        winner = None
        if rel in local_fp:
            want = local_fp[rel]
            for h in hits:
                if remote_fps.get(h) == want:
                    winner = h
                    break
        if winner == dest:
            already.append(rel)
        elif winner:
            link_cmds.append((dest, winner, os.path.getsize(lp)))
        else:
            truly_missing.append(rel)

    with open("/tmp/head_sweep_link.sh", "w") as f:
        f.write("#!/bin/bash\n# liens miroir head_sweep_inputs (généré par "
                "head_sweep_reconcile.py)\nset -u\n")
        for dest, src, _ in link_cmds:
            f.write(f'mkdir -p "$(dirname "{dest}")" && ln -sfn "{src}" "{dest}"\n')

    with open("/tmp/head_sweep_missing.txt", "w") as f:
        f.write("\n".join(truly_missing) + ("\n" if truly_missing else ""))

    saved = sum(s for _, _, s in link_cmds)
    print(f"\n[reconcile] RAPPORT : {len(already)} déjà en place (miroir), "
          f"{len(link_cmds)} à lier sur place, {len(truly_missing)} à pousser")
    print(f"[reconcile] upload économisé : {saved/1e9:.2f} Go sur "
          f"{(saved + sum(os.path.getsize(os.path.join(ROOT,r)) for r in truly_missing))/1e9:.2f} Go requis")
    if truly_missing:
        groups = {}
        for r in truly_missing:
            key = "/".join(r.split("/")[:2])
            groups[key] = groups.get(key, 0) + 1
        print("[reconcile] manquants par zone :")
        for k, n in sorted(groups.items()):
            print(f"  {k}: {n} fichiers")
    print("[reconcile] → /tmp/head_sweep_link.sh  (à exécuter sur Narval)")
    print("[reconcile] → /tmp/head_sweep_missing.txt  (manifeste du rsync résiduel)")


if __name__ == "__main__":
    main()
