# code/Makefile — package-level targets. Always loads the python module first: the venv's
# interpreter dynamically links against libpython3.10.so.1.0, which only resolves once the
# python/3.10.4 module has set LD_LIBRARY_PATH. This bit twice in P0; do not drop it.
SHELL := /bin/bash
REPO_ROOT := $(abspath ..)
VENV := $(REPO_ROOT)/envs/p3fcl
ACTIVATE := module load python/3.10.4 >/dev/null 2>&1; source $(VENV)/bin/activate

.PHONY: setup test smoke figures verify paper

setup:
	@bash $(REPO_ROOT)/env/bootstrap_kodiak.sh

test:
	@$(ACTIVATE); python -m pytest tests/ -q

smoke:
	@$(ACTIVATE); time python -m pytest tests/test_methods.py tests/test_claims.py -q

# Regenerates every figure/table this repo currently has a script for (P4's FIG01/FIG02/TAB03/TAB04).
# Grows as more analysis/*.py scripts land -- CLAUDE.md non-negotiable #3: these read results/*.csv
# and do no computation of their own.
figures:
	@$(ACTIVATE); cd $(REPO_ROOT) && \
	python analysis/fig01_decoupling.py && \
	python analysis/fig02_halflife.py && \
	python analysis/fig03_dose_response_pilot.py && \
	python analysis/fig04_semantic_vs_individual.py && \
	python analysis/fig05_eps_of_T.py && \
	python analysis/fig11_secure_agg.py && \
	python analysis/fig13_gram_inversion.py && \
	python analysis/fig16_roc.py && \
	python analysis/fig18_natural_federation.py && \
	python analysis/fig_a6_property_inference.py && \
	python analysis/fig17_seed_variance.py && \
	python analysis/tab02_units.py && \
	python analysis/tab03_leakage.py && \
	python analysis/tab04_halflife.py && \
	python analysis/tab10_taxonomy.py

# Provenance check (CLAUDE.md non-negotiable #3, build/00_BUILD_PLAN.md's "never cut" list): every
# numeric literal in paper/**/*.tex must trace to a results/*.csv row, a generated tables/*.tex
# value, or an explicit \providedvalue{} escape. Clean (0 exceptions) on an empty/content-free
# paper/, which is the P0 gate's own definition of passing.
verify:
	@$(ACTIVATE); python $(REPO_ROOT)/analysis/verify_provenance.py

# `latexmk` is not installed on this system (checked 2026-09-21: pdflatex/bibtex are, latexmk/biber
# are not) -- plain 4-pass pdflatex+bibtex instead, the standard sequence to resolve citations and
# cross-references without latexmk's automation. paper/main.tex is a structural scaffold (figure/
# table wiring only, no prose) until Opus fills in the actual manuscript content -- see
# build/STATE.md and paper/PAPER_BRIEFING.md.
paper:
	@$(ACTIVATE); cd $(REPO_ROOT)/paper && \
	pdflatex -interaction=nonstopmode main.tex && \
	bibtex main || true && \
	pdflatex -interaction=nonstopmode main.tex && \
	pdflatex -interaction=nonstopmode main.tex
