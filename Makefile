# Consensoor Makefile
# Spec test targets for running Ethereum consensus spec tests

SPEC_VERSION ?= v1.7.0-alpha.1
SPEC_TESTS_DIR := tests/spec-tests
DEFAULT_PRESET := mainnet
PYTEST_PARALLEL := -n auto

# Supported forks and presets
FORKS := phase0 altair bellatrix capella deneb electra fulu gloas
PRESETS := minimal mainnet

.PHONY: all clean help docker check-tests fetch-tests-minimal fetch-tests-mainnet \
	$(foreach f,$(FORKS),test-$(f)-minimal test-$(f)-mainnet) \
	test-all-minimal test-all-mainnet

help:
	@echo "Consensoor Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make test <fork> [preset]      - Run spec tests (default: mainnet)"
	@echo "  make test all [preset]         - Run all spec tests"
	@echo ""
	@echo "Examples:"
	@echo "  make test phase0               - Run phase0 tests (mainnet)"
	@echo "  make test phase0 minimal       - Run phase0 tests (minimal)"
	@echo "  make test phase0 mainnet       - Run phase0 tests (mainnet)"
	@echo "  make test electra              - Run electra tests (mainnet)"
	@echo "  make test all                  - Run all tests (mainnet)"
	@echo "  make test all minimal          - Run all tests (minimal)"
	@echo "  make test gloas                - Run gloas (ePBS) tests"
	@echo ""
	@echo "Other targets:"
	@echo "  make fetch-tests-minimal       - Download minimal preset tests (~470MB)"
	@echo "  make fetch-tests-mainnet       - Download mainnet preset tests (~680MB)"
	@echo "  make check-tests               - Check if tests are downloaded"
	@echo "  make docker                    - Build Docker image"
	@echo "  make clean                     - Clean test artifacts"
	@echo ""
	@echo "Default preset: $(DEFAULT_PRESET)"
	@echo "Spec version: $(SPEC_VERSION)"

# Create tests directory
$(SPEC_TESTS_DIR):
	mkdir -p $(SPEC_TESTS_DIR)

# Fetch minimal preset tests
fetch-tests-minimal: $(SPEC_TESTS_DIR)
	@echo "Checking for consensus spec tests $(SPEC_VERSION) (minimal preset)..."
	@if [ ! -f "$(SPEC_TESTS_DIR)/minimal.tar.gz" ]; then \
		echo "Downloading minimal spec tests (~470MB)..."; \
		curl -L --progress-bar -o "$(SPEC_TESTS_DIR)/minimal.tar.gz" \
			"https://github.com/ethereum/consensus-specs/releases/download/$(SPEC_VERSION)/minimal.tar.gz"; \
	else \
		echo "Using cached: $(SPEC_TESTS_DIR)/minimal.tar.gz"; \
	fi
	@if [ ! -d "$(SPEC_TESTS_DIR)/tests/minimal" ]; then \
		echo "Extracting..."; \
		cd $(SPEC_TESTS_DIR) && tar -xzf minimal.tar.gz; \
	fi
	@echo "Ready: $(SPEC_TESTS_DIR)/tests/minimal"

# Fetch mainnet preset tests
fetch-tests-mainnet: $(SPEC_TESTS_DIR)
	@echo "Checking for consensus spec tests $(SPEC_VERSION) (mainnet preset)..."
	@if [ ! -f "$(SPEC_TESTS_DIR)/mainnet.tar.gz" ]; then \
		echo "Downloading mainnet spec tests (~680MB)..."; \
		curl -L --progress-bar -o "$(SPEC_TESTS_DIR)/mainnet.tar.gz" \
			"https://github.com/ethereum/consensus-specs/releases/download/$(SPEC_VERSION)/mainnet.tar.gz"; \
	else \
		echo "Using cached: $(SPEC_TESTS_DIR)/mainnet.tar.gz"; \
	fi
	@if [ ! -d "$(SPEC_TESTS_DIR)/tests/mainnet" ]; then \
		echo "Extracting..."; \
		cd $(SPEC_TESTS_DIR) && tar -xzf mainnet.tar.gz; \
	fi
	@echo "Ready: $(SPEC_TESTS_DIR)/tests/mainnet"

# Check if tests are downloaded
check-tests:
	@echo "Spec tests status:"
	@if [ -f "$(SPEC_TESTS_DIR)/minimal.tar.gz" ]; then \
		echo "  minimal: $$(ls -lh $(SPEC_TESTS_DIR)/minimal.tar.gz | awk '{print $$5}')"; \
	else \
		echo "  minimal: not downloaded"; \
	fi
	@if [ -f "$(SPEC_TESTS_DIR)/mainnet.tar.gz" ]; then \
		echo "  mainnet: $$(ls -lh $(SPEC_TESTS_DIR)/mainnet.tar.gz | awk '{print $$5}')"; \
	else \
		echo "  mainnet: not downloaded"; \
	fi

# Main test target - handles "make test <fork> [preset]"
test:
	@FORK="$(word 2,$(MAKECMDGOALS))"; \
	PRESET="$(word 3,$(MAKECMDGOALS))"; \
	if [ -z "$$FORK" ]; then \
		echo "Usage: make test <fork> [preset]"; \
		echo "  fork: phase0|altair|bellatrix|capella|deneb|electra|fulu|gloas|all"; \
		echo "  preset: minimal|mainnet (default: $(DEFAULT_PRESET))"; \
		exit 1; \
	fi; \
	if [ -z "$$PRESET" ]; then \
		PRESET="$(DEFAULT_PRESET)"; \
	fi; \
	$(MAKE) --no-print-directory _run-test FORK=$$FORK PRESET=$$PRESET

# Clean Python cache (always run before tests)
clean-cache:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Internal target to run tests
_run-test: clean-cache fetch-tests-$(PRESET)
	@if [ "$(FORK)" = "all" ]; then \
		echo "Running ALL spec tests ($(PRESET)) [parallel: $(PYTEST_PARALLEL)]..."; \
		python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=$(PRESET) \
			--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/$(PRESET) 2>&1; \
	elif [ "$(FORK)" = "gloas" ] && [ ! -d "$(SPEC_TESTS_DIR)/tests/$(PRESET)/gloas" ]; then \
		echo "Running gloas spec tests ($(PRESET)) [parallel: $(PYTEST_PARALLEL)]..."; \
		echo "Note: Gloas (ePBS) not in mainline spec tests yet."; \
		echo "Running local gloas tests..."; \
		python3 -m pytest tests/spec/test_gloas.py -v $(PYTEST_PARALLEL) --preset=$(PRESET) 2>&1 || true; \
	else \
		echo "Running $(FORK) spec tests ($(PRESET)) [parallel: $(PYTEST_PARALLEL)]..."; \
		python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=$(PRESET) \
			--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/$(PRESET) \
			-k "$(FORK)" 2>&1; \
	fi

# Define no-op targets for fork and preset names (to prevent "no rule" errors)
$(FORKS) all minimal mainnet:
	@:

# Explicit targets for direct invocation (alternative syntax: make test-phase0-mainnet)
test-phase0-minimal: fetch-tests-minimal
	@echo "Running phase0 spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "phase0"

test-phase0-mainnet: fetch-tests-mainnet
	@echo "Running phase0 spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "phase0"

test-altair-minimal: fetch-tests-minimal
	@echo "Running altair spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "altair"

test-altair-mainnet: fetch-tests-mainnet
	@echo "Running altair spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "altair"

test-bellatrix-minimal: fetch-tests-minimal
	@echo "Running bellatrix spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "bellatrix"

test-bellatrix-mainnet: fetch-tests-mainnet
	@echo "Running bellatrix spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "bellatrix"

test-capella-minimal: fetch-tests-minimal
	@echo "Running capella spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "capella"

test-capella-mainnet: fetch-tests-mainnet
	@echo "Running capella spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "capella"

test-deneb-minimal: fetch-tests-minimal
	@echo "Running deneb spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "deneb"

test-deneb-mainnet: fetch-tests-mainnet
	@echo "Running deneb spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "deneb"

test-electra-minimal: fetch-tests-minimal
	@echo "Running electra spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "electra"

test-electra-mainnet: fetch-tests-mainnet
	@echo "Running electra spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "electra"

test-fulu-minimal: fetch-tests-minimal
	@echo "Running fulu spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "fulu"

test-fulu-mainnet: fetch-tests-mainnet
	@echo "Running fulu spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "fulu"

test-gloas-minimal: fetch-tests-minimal
	@echo "Running gloas spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	@if [ -d "$(SPEC_TESTS_DIR)/tests/minimal/gloas" ]; then \
		python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
			--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal -k "gloas"; \
	else \
		echo "Note: Gloas (ePBS) not in mainline spec tests yet."; \
		echo "Running local gloas tests..."; \
		python3 -m pytest tests/spec/test_gloas.py -v $(PYTEST_PARALLEL) --preset=minimal || true; \
	fi

test-gloas-mainnet: fetch-tests-mainnet
	@echo "Running gloas spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	@if [ -d "$(SPEC_TESTS_DIR)/tests/mainnet/gloas" ]; then \
		python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
			--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet -k "gloas"; \
	else \
		echo "Note: Gloas (ePBS) not in mainline spec tests yet."; \
		echo "Running local gloas tests..."; \
		python3 -m pytest tests/spec/test_gloas.py -v $(PYTEST_PARALLEL) --preset=mainnet || true; \
	fi

test-all-minimal: fetch-tests-minimal
	@echo "Running ALL spec tests (minimal) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=minimal \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/minimal

test-all-mainnet: fetch-tests-mainnet
	@echo "Running ALL spec tests (mainnet) [parallel: $(PYTEST_PARALLEL)]..."
	python3 -m pytest tests/spec/ -v $(PYTEST_PARALLEL) --preset=mainnet \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/mainnet

# Docker build
docker:
	docker build -t consensoor:latest .

# Clean test artifacts
clean:
	rm -rf $(SPEC_TESTS_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Prevent make from treating fork/preset names as files
.PHONY: $(FORKS) all minimal mainnet test _run-test clean-cache
