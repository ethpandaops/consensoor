# Consensoor Makefile

SPEC_VERSION ?= nightly
SPEC_TESTS_DIR := tests/spec-tests
DOWNLOAD_REFTESTS := tests/download_reftests.sh

.PHONY:          \
	docker       \
	fetch-tests  \
	help         \
	test

help:
	@echo "Consensoor Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make test                 # Run minimal tests for all forks"
	@echo "  make test preset=mainnet  # Run mainnet tests for all forks"
	@echo "  make test fork=electra    # Run minimal tests for electra"
	@echo "  make test fork=electra preset=mainnet"
	@echo ""
	@echo "Presets: minimal, mainnet"
	@echo "Forks: phase0, altair, bellatrix, capella, deneb, electra, fulu, gloas"
	@echo ""
	@echo "Other targets:"
	@echo "  make fetch-tests  # Download reference tests"
	@echo "  make docker       # Build Docker image"

fetch-tests:
	@$(DOWNLOAD_REFTESTS) "$(SPEC_TESTS_DIR)" "$(SPEC_VERSION)"

test: CORES := $(or $(cores),auto)
test: PRESET := $(or $(preset),minimal)
test: MAYBE_FORK := $(if $(fork),-k "$(fork)")
test: fetch-tests
	@python3 -m pytest           \
		tests/spec/              \
		--verbose                \
		--numprocesses $(CORES)  \
		--preset=$(PRESET)       \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/$(PRESET) \
		$(MAYBE_FORK)

docker:
	docker build -t consensoor:latest .
