# Known issues

## `bootstrap_ci.py` — schéma de classes (`--pass`) et fiabilité de `f1_macro_all`

### Le bug (corrigé le 2026-07-10)

`scripts/bootstrap_ci.py` avait `N_CLASSES = 12` codé en dur et re-fittait le probe
sur les features BRUTES renvoyées par `load_features()` **sans appliquer la réduction
de classes** que `probe.py` applique pour chaque passe. Il ne reproduisait donc
correctement QUE la passe `with_rhol` (12cls, aucun drop).

Pour les passes `without_rhol` (11cls) et surtout `8cls` (8cls), les embeddings sur
disque sont en schéma **source** (12cls pour les canoniques, 11cls pour les runs SOTA).
La passe cible s'obtient en retirant des classes via `src.latent.drop_class` (cascade
décroissante `src.utils.pass_drops`), ce qui **compacte** les indices restants.

Conséquence sur 8cls avec l'ancien code :

- Un simple `--n-classes 8` aurait calculé `f1_macro_all` avec `labels=range(8)` sur des
  features **12cls non réduites** → indices 0..7 = `ALDE, ARCA, BIRC, DRYI, LICH, MOSS,
  PETF, RHOL` : les **mauvaises** classes (inclut 3 classes censées être retirées +
  RHOL, exclut `SEDG, TUSS, WILL`). Les 8 vraies classes 8cls sont `[0,2,4,5,6,9,10,11]`
  en 12cls, et deviennent `0..7` seulement APRÈS la cascade de drops.
- De plus le classifieur lui-même aurait été fitté sur 11/12 classes au lieu de 8 →
  **`f1_macro_pres` aussi faux** (pas seulement `f1_macro_all`). C'était donc un bug
  plus profond qu'un simple paramètre de comptage.

### Le fix

Ajout de `--pass {with_rhol,without_rhol,8cls}` (défaut `with_rhol` = schéma A,
rétro-compatible). Le re-fit applique désormais la **même** cascade de drops
source-aware que `probe.py` (`pass_drops(tag, source_schema(model))`), et `n_classes`
est dérivé de la passe. `--pass` **remplace** l'idée d'un `--n-classes` seul (qui aurait
été silencieusement faux).

Rétro-compatibilité : `--pass with_rhol` → `drops=[]` (12cls) → chemin de code identique
à l'ancien (`n_test=17598`, 12 classes). Aucun changement pour les runs schéma A.

### Fiabilité de `f1_macro_all` par passe (à citer correctement)

- **`with_rhol` (12cls)** et **`without_rhol` (11cls)** : `f1_macro_all` inclut des
  classes **jamais évaluables** (RHOL absente du test → F1=0 ; ARCA/DRYI/RUBC ≈ 0 en
  probe). La moyenne macro est donc **biaisée à la baisse** — c'est un artefact connu,
  pas une vraie mesure de performance. **Citer `f1_macro_pres`** (métrique reportée du
  mémoire), pas `f1_macro_all`, pour ces passes.
- **`8cls` (8cls)** : les 8 classes sont **toutes présentes** dans le test
  (`labels présents == range(8)`), donc `f1_macro_all == f1_macro_pres` (vérifié :
  `diff == 0`). Les deux sont fiables et interchangeables pour 8cls.

### Validation locale (2026-07-10)

Sur `dinov3_vitl16_lvd` (embeddings cachés localement, `--force-c 0.01`,
`max_iter` réduit — donc valeur non canonique) : passe `8cls` → `n_test=17277`,
`obs_all == obs_pres` (diff 0). La reproduction du chiffre canonique exact
(`test.f1_macro_all = 0.6650` pour `dinov3_vitl16_lvd` 8cls) nécessite le `best_C` du
`8cls/probe_knn.json` du cluster et `max_iter=2000` ; à lancer sur le cluster
(embeddings/tiles non disponibles en local).
