# Repo-root Makefile — the commands named in CLAUDE.md. setup/test/smoke delegate into code/
# (the package); figures/verify operate on the repo-root analysis/, results/, figs/, tables/
# directories, per CLAUDE.md's Layout.
SHELL := /bin/bash
.PHONY: setup test smoke figures verify

setup:
	$(MAKE) -C code setup

test:
	$(MAKE) -C code test

smoke:
	$(MAKE) -C code smoke

figures:
	@module load python/3.10.4 >/dev/null 2>&1; source envs/p3fcl/bin/activate; \
	for f in analysis/fig*.py analysis/tab*.py; do \
		[ -e "$$f" ] || continue; \
		echo "-- $$f --"; python "$$f" || exit 1; \
	done
	@echo "figures: all current analysis scripts completed."

verify:
	@module load python/3.10.4 >/dev/null 2>&1; source envs/p3fcl/bin/activate; \
	python code/scripts/check_fx9_consistency.py && python analysis/verify_provenance.py
