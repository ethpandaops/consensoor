# Consensoor Makefile

SPEC_VERSION ?= nightly
SPEC_TESTS_DIR := tests/spec-tests
DOWNLOAD_REFTESTS := tests/download_reftests.sh

VALID_PRESETS := minimal mainnet
VALID_FORKS := phase0 altair bellatrix capella deneb electra fulu gloas

# Parse targets: separate known targets from test parameters.
# Words like 'minimal', 'mainnet', fork names, and 'all' are test params.
MAKECMDGOALS ?=
TEST_PARAMS := $(filter all $(VALID_PRESETS) $(VALID_FORKS),$(MAKECMDGOALS))
REAL_TARGETS := $(filter-out $(TEST_PARAMS),$(MAKECMDGOALS))

# Extract preset and fork from test params
PARAM_PRESET := $(firstword $(filter $(VALID_PRESETS),$(TEST_PARAMS)))
PARAM_FORK := $(firstword $(filter $(VALID_FORKS),$(TEST_PARAMS)))

# 'all' means all forks (no filter), which is the default
HAS_ALL := $(filter all,$(TEST_PARAMS))

# Check for unknown params (not a preset, not a fork, not 'all')
UNKNOWN_PARAMS := $(filter-out all $(VALID_PRESETS) $(VALID_FORKS),$(TEST_PARAMS))

.PHONY:              \
	docker           \
	fetch-tests      \
	help             \
	test             \
	all              \
	$(VALID_PRESETS) \
	$(VALID_FORKS)

# No-op targets so make doesn't error on preset/fork/all words
all $(VALID_PRESETS) $(VALID_FORKS):
	@true

help:
	@echo "Consensoor Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make test                     # Run minimal tests for all forks"
	@echo "  make test mainnet             # Run mainnet tests for all forks"
	@echo "  make test electra             # Run minimal tests for electra"
	@echo "  make test mainnet electra     # Run mainnet tests for electra"
	@echo "  make test all                 # Run minimal tests for all forks"
	@echo "  make test mainnet all         # Run mainnet tests for all forks"
	@echo ""
	@echo "Presets: $(VALID_PRESETS)"
	@echo "Forks:   $(VALID_FORKS)"
	@echo ""
	@echo "Other targets:"
	@echo "  make fetch-tests  # Download reference tests"
	@echo "  make docker       # Build Docker image"

fetch-tests:
	@$(DOWNLOAD_REFTESTS) "$(SPEC_TESTS_DIR)" "$(SPEC_VERSION)"

test: CORES := $(or $(cores),auto)
test: PRESET := $(or $(PARAM_PRESET),$(preset),minimal)
test: FORK := $(or $(PARAM_FORK),$(fork))
test: MAYBE_FORK := $(if $(or $(PARAM_FORK),$(fork)),-k "$(or $(PARAM_FORK),$(fork))")
test: fetch-tests
	@if [ -n "$(UNKNOWN_PARAMS)" ]; then \
		echo "Error: Unknown parameter(s): $(UNKNOWN_PARAMS)"; \
		echo "Run 'make help' for usage."; \
		exit 1; \
	fi
	@python3 -m pytest           \
		tests/spec/              \
		--verbose                \
		--numprocesses $(CORES)  \
		--preset=$(PRESET)       \
		--spec-tests-dir=$(SPEC_TESTS_DIR)/tests/$(PRESET) \
		$(MAYBE_FORK)

docker:
	docker build -t consensoor:latest .


# Catch unknown targets
%:
	@$(MAKE) --no-print-directory help
	@echo "Error: Unknown target '$@'" && exit 1
