PY      := .venv/bin/python
SERPINT := .venv/bin/serpint

.PHONY: setup test lint format locations budget pilot-dry pilot inspect-pilot collect-dry collect inspect

setup:            ## create the venv and install pinned deps + the package
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e . --no-deps

test:             ## run the unit tests (no network, no API calls)
	.venv/bin/pytest

lint:             ## ruff lint + format check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

format:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

locations:        ## resolve cities to SerpAPI canonical locations (free)
	$(SERPINT) locations

budget:           ## ledger spend + SerpAPI account balance
	$(SERPINT) budget

pilot-dry:        ## planned pilot calls and their cost; makes no requests
	$(SERPINT) collect pilot --dry-run

pilot:            ## 5 cities x 5 queries (cells are reused by the full grid)
	$(SERPINT) collect pilot

inspect-pilot:    ## reports/pilot_fields.md from the cached pilot responses
	$(SERPINT) inspect pilot

collect-dry:      ## planned full-grid calls and their cost; makes no requests
	$(SERPINT) collect grid --dry-run

collect:          ## full 20 x 20 grid; resumable, skips cached cells
	$(SERPINT) collect grid

inspect:
	$(SERPINT) inspect grid
