BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

SRC_DIR := $(BASE_DIR)/src
DATA_DIR := $(BASE_DIR)/data

DATASET_DIR := $(DATA_DIR)/datasets
DATASET_DIR_3D := $(DATASET_DIR)/3d
DATASET_DIR_2D := $(DATASET_DIR)/2d

RESULTS_DIR := $(DATA_DIR)/results
RESULTS_DIR_SIZES := $(RESULTS_DIR)/sizes
RESULTS_DIR_NEIGHBORS := $(RESULTS_DIR_SIZES)/neighbors
RESULTS_DIR_SCALEFACTORS := $(RESULTS_DIR_SIZES)/scale_factors


CHECKPOINTS_DIR := $(DATA_DIR)/checkpoints
CHECKPOINTS_TRIAL_BASENAME := trial_

IDX := $(shell ls -d "$(CHECKPOINTS_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)"* 2>/dev/null | \
        sed "s|.*$(CHECKPOINTS_TRIAL_BASENAME)||" | \
        sort -n | \
        tail -1)

# Set default if empty
IDX ?= 0
NEXT_IDX = $(shell echo $$(($(IDX) + 1)))
CHECKPOINTS_DIR_NEXT := $(CHECKPOINTS_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)$(NEXT_IDX)

PATHS_FILE = $(SRC_DIR)/paths.json

.PHONY: config gendata run

config:
	@echo "Configuring idx for checkpoints at path $(CHECKPOINTS_DIR)..."
	@echo "greatest index is $(IDX)"
	@echo "Creating directories..."
	mkdir -p $(DATASET_DIR_3D)
	mkdir -p $(DATASET_DIR_2D)
	@echo "Creating paths file..."
	@echo '{' > $(PATHS_FILE)
	@echo '"dataset2d": "$(DATASET_DIR_2D)",' >> $(PATHS_FILE)
	@echo '"dataset3d": "$(DATASET_DIR_3D)",' >> $(PATHS_FILE)
	@echo '"checkpoints_next": "$(CHECKPOINTS_DIR_NEXT)",' >> $(PATHS_FILE)
	@echo '"results_sizes": "$(RESULTS_DIR_SIZES)",' >> $(PATHS_FILE)
	@echo '"results_neighbors": "$(RESULTS_DIR_NEIGHBORS)",' >> $(PATHS_FILE)
	@echo '"results_scalefactors": "$(RESULTS_DIR_SCALEFACTORS)"' >> $(PATHS_FILE)
	@echo '}' >> $(PATHS_FILE)

gendata: config
	@echo "Generating Swiss roll dataset..."
	$(PYTHON) $(SRC_DIR)/dataset_generation.py --paths $(PATHS_FILE)

run: config
	@echo "Running manifold sculpting..."
	$(PYTHON) $(SRC_DIR)/run.py --paths $(PATHS_FILE)