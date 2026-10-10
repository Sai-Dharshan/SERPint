PY      := .venv/bin/python
SERPINT := .venv/bin/serpint

.PHONY: setup test lint format locations budget pilot-dry pilot inspect-pilot core-dry core inspect-core collect-dry collect inspect parse-core parse

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

pilot:            ## 5 cities x 5 queries = 25 searches (all reused by core and grid)
	$(SERPINT) collect pilot

inspect-pilot:    ## reports/pilot_fields.md from the cached pilot responses
	$(SERPINT) inspect pilot

core-dry:         ## planned core calls; pilot cells show as cached
	$(SERPINT) collect core --dry-run

core:             ## 16 cities x 15 queries = 240 searches; fits one free-tier month
	$(SERPINT) collect core

inspect-core:
	$(SERPINT) inspect core

collect-dry:      ## planned full-grid calls and their cost; makes no requests
	$(SERPINT) collect grid --dry-run

collect:          ## full 20 x 20 grid (tops up core with 160 more); resumable
	$(SERPINT) collect grid

inspect:
	$(SERPINT) inspect grid

parse-core:       ## visible text of the core pages -> data/processed/, reports/core_parse.md
	$(SERPINT) parse core

parse:
	$(SERPINT) parse grid
