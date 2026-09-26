# code/Makefile — package-level targets. Always loads the python module first: the venv's
# interpreter dynamically links against libpython3.10.so.1.0, which only resolves once the
# python/3.10.4 module has set LD_LIBRARY_PATH. This bit twice in P0; do not drop it.
SHELL := /bin/bash
REPO_ROOT := $(abspath ..)
VENV := $(REPO_ROOT)/envs/p3fcl
ACTIVATE := module load python/3.10.4 >/dev/null 2>&1; source $(VENV)/bin/activate

.PHONY: setup test smoke figures verify

setup:
	@bash $(REPO_ROOT)/env/bootstrap.sh

test:
	@$(ACTIVATE); python -m pytest tests/ -q

smoke:
	@$(ACTIVATE); time python -m pytest tests/test_methods.py tests/test_claims.py -q

# Figures/tables and the provenance check are defined once, in the repo-root Makefile
# (every analysis/fig*.py and analysis/tab*.py, then check_fx9_consistency.py + verify_provenance.py).
figures:
	@$(MAKE) -C $(REPO_ROOT) figures

verify:
	@$(MAKE) -C $(REPO_ROOT) verify
