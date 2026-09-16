# Makefile — Benchmark Arctic-TVC (passe nocturne 2)
#
# Cible principale : ``make paper`` — exécute le pipeline A→F end-to-end
# depuis un checkout propre, avec les embeddings dans ``embeddings/`` (lecture seule).
#
# Cibles :
#   paper       — A→F complet (probe + latent + significance + transfer + figures + tables)
#   probe       — A seul (probe + k-NN, 12 modèles)
#   latent      — B seul (RankMe + anisotropie + géométrie étendue)
#   sig         — C seul (bootstrap apparié n=1000)
#   transfer    — D seul (LogME + α-ReQ + NESum + corrélations)
#   figures     — E seul (figures paper + support)
#   tables      — F seul (fragments LaTeX)
#   clean       — supprime UNIQUEMENT les artefacts sous results/ et docs/figures/ (sauf .deprecated)
#   status      — vérifie embeddings + affiche résumé
#   reconcile   — recalcule les valeurs de contrôle (sanity-check vs CHANGELOG/RECONCILIATION)
#
# Garanties : seed=42 partout, déterministe bit-pour-bit, idempotent.
# Les chiffres canoniques exigent le BLAS mono-thread (AGENTS.md §4.8) : les 5
# variables sont donc exportées ici pour toutes les recettes du Makefile.

export OMP_NUM_THREADS    := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS    := 1
export NUMEXPR_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

PROJ     := $(shell pwd)
PY       := python3
LOG      := results/transfer/pipeline_run.log
NBOOT    ?= 1000

.PHONY: paper probe latent sig transfer figures tables clean status reconcile help

help:
	@echo "Cibles : paper | probe | latent | sig | transfer | figures | tables | status | reconcile | clean"

# ------------------------------------------------------------------ A → F
paper:
	$(PY) scripts/run_pipeline.py --log $(LOG) --n-bootstrap $(NBOOT)

probe:     ; $(PY) scripts/run_pipeline.py --only A --log $(LOG)
latent:    ; $(PY) scripts/run_pipeline.py --only B --log $(LOG)
sig:       ; $(PY) scripts/run_pipeline.py --only C --log $(LOG) --n-bootstrap $(NBOOT)
transfer:  ; $(PY) scripts/run_pipeline.py --only D --log $(LOG)
figures:   ; $(PY) scripts/run_pipeline.py --only E --log $(LOG)
tables:    ; $(PY) scripts/run_pipeline.py --only F --log $(LOG)

# ------------------------------------------------------------------ vérification
status:
	@echo "=== Embeddings présents ? ===" && \
	  ls embeddings/*_test.npy 2>/dev/null | wc -l | xargs printf "  test.npy:    %s\n" && \
	  ls embeddings/*_train.npy 2>/dev/null | wc -l | xargs printf "  train.npy:   %s\n" && \
	  ls embeddings/*_val.npy 2>/dev/null | wc -l | xargs printf "  val.npy:     %s\n"
	@echo "=== Artefacts canoniques ===" && \
	  for f in results/with_rhol/probe_knn_cgrid.json \
	           results/with_rhol/latent_metrics.json \
	           results/geometry_extended_12models.json \
	           results/significance_matrix_all12.json \
	           results/transfer/spectrum_metrics.json \
	           results/transfer/logme_scores.json \
	           results/correlations.json \
	           results/transfer/headline_pairs_paired_tests.json \
	           results/transfer/aso_matrix_eps_min.json \
	           results/figures_paper/headline_pair.png \
	           results/figures_paper/controlled_pair.png ; do \
	    [ -f $$f ] && printf "  OK  %s\n" $$f || printf "  MANQUE  %s\n" $$f ; \
	  done

reconcile:
	$(PY) scripts/recompute_control_values.py

# ------------------------------------------------------------------ nettoyage
clean:
	@echo "ATTENTION : va supprimer les artefacts sous results/ et docs/figures/."
	@echo "  Pas les embeddings. Pas les .deprecated."
	@read -p "Confirmer (o/N) ? " r && [ "$$r" = "o" ] || (echo "Annulé." && exit 1)
	rm -f $(LOG)
	rm -rf results/with_rhol/*.json results/with_rhol/probe_knn.json
	rm -rf results/without_rhol/*.json
	rm -f  results/significance_matrix.json results/significance_matrix.png
	rm -f  results/significance_corrected.json
	rm -f  results/geometry_extended.json
	rm -f  results/geometry_extended_12models.json
	rm -rf results/transfer/*.json results/transfer/*.csv
	rm -rf results/transfer/*.png results/transfer/*.pdf
	rm -rf results/figures/*
	rm -rf results/figures_paper/*
	rm -rf results/figures_support/*
	rm -f  results/correlations.json results/correlations_corrected.json
	rm -f  results/correlations_n9_vs_n12.csv
	rm -f  results/latent_*.json results/layerwise_probe.json
	@echo "Artefacts supprimés. Embeddings intacts. Prêt pour 'make paper'."
