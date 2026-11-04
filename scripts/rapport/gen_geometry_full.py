#!/usr/bin/env python3
"""Régénère tables/t_geometry_full.tex — version réduite/lisible du tableau complet.

Retire les colonnes redondantes (RankMe σ ≈ Rang eff σ², StbRk ≈ PR, CHI ≈ Silhouette)
et espace les en-têtes ; la heatmap geom_metrics_heatmap.png garde le détail complet.

    python3 scripts/rapport/gen_geometry_full.py
"""
import csv, json, os, sys
ROOT = '/home/erazal/Documents/Mémoire'
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'rapport'))
from registry import CANONICAL_F1, OUT

# (col, en-tête, décimales)
COLS = [
    ('eff_rank_sigma2', 'Rang eff. $\\sigma^2$', 1),
    ('participation_ratio', 'PR', 1),
    ('alpha_spectral', '$\\alpha$', 2),
    ('aniso_cosine', 'Aniso', 3),
    ('fisher_ratio', 'Fisher', 3),
    ('knn_purity', 'kNN', 3),
    ('silhouette', 'Silh', 4),
    ('nc1_canonical', 'NC1', 2),
    ('dbi', 'DBI', 2),
    ('intrinsic_dim_twonn', 'TwoNN', 1),
    ('logme_train', 'LogME tr', 1),
]
def num(v, nd):
    return ('{:.%df}' % nd).format(float(v)).replace('.', ',')

rows = [r for r in csv.DictReader(open(os.path.join(OUT, 'geometry_models.csv')))
        if r['model'] in CANONICAL_F1]
rows.sort(key=lambda r: -CANONICAL_F1[r['model']][0])
disp = {m['model']: m['display']
        for m in json.load(open(os.path.join(ROOT, 'results/all_models_canonical_merged.json')))['models']}
head = '\\toprule
Modèle & F1 & ' + ' & '.join(l for _, l, _ in COLS) + ' \\\\
\\midrule
'
lines = ['\\setlength{\\tabcolsep}{4pt}\\scriptsize',
         '\\begin{longtable}{@{}lr' + 'r' * len(COLS) + '@{}}',
         '\\caption{Géométrie latente des %d modèles --- protocole unique : embeddings '
         'test, sous-échantillon 20\,000, seed~42, métriques calculées par seed puis '
         'moyennées. Trié par F1 décroissant. Les colonnes redondantes (RankMe σ, StbRk, '
         'CHI) et la heatmap normalisée gardent le détail complet.}\\label{tab:geomfull}\\'
         % len(rows), head, '\\endfirsthead', '\\toprule',
         'Modèle & F1 & ' + ' & '.join(l for _, l, _ in COLS) + ' \\\\', '\\midrule', '\\endhead',
         '\\bottomrule', '\\endfoot']
for r in rows:
    cells = [disp.get(r['model'], r['model']), num(CANONICAL_F1[r['model']][0], 4)]
    for col, _l, nd in COLS:
        v = num(r[col], nd)
        sd = r.get(col + '_std', '')
        if sd not in ('', None) and float(sd) > 0:
            v += '\\scriptsize{$\\pm$\\,%s}' % num(sd, nd)
        cells.append(v)
    lines.append(' & '.join(cells) + ' \\\\')
lines += ['\\bottomrule', '\\end{longtable}',
          '\\vspace{2pt}\\par\\footnotesize\\textit{Source : }'
          '\\texttt{results/rapport\\_data/geometry\\_models.csv} '
          '\\texttt{(scripts/rapport/geometry\\_test\\_fixed\\_n.py)}.']
open(os.path.join(ROOT, 'rapport_bouguessa/tables/t_geometry_full.tex'), 'w').write(
    '\\n'.join(lines))
print('t_geometry_full.tex écrit (%d modèles, %d colonnes)' % (len(rows), len(COLS)))
