BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

SRC_DIR := $(BASE_DIR)/src
DATA_DIR := $(BASE_DIR)/data
FIG_DIR := $(BASE_DIR)/figs

DATASET_DIR := $(DATA_DIR)/datasets
DATASET_DIR_3D := $(DATASET_DIR)/3d
DATASET_DIR_2D := $(DATASET_DIR)/2d

RESULTS_DIR := $(DATA_DIR)/results
RESULTS_DIR_SIZES := $(RESULTS_DIR)/sizes
RESULTS_DIR_NEIGHBORS := $(RESULTS_DIR_SIZES)/neighbors
RESULTS_DIR_SCALEFACTORS := $(RESULTS_DIR_SIZES)/scale_factors


CHECKPOINTS_DIR := $(DATA_DIR)/checkpoints
CHECKPOINTS_TRIAL_BASENAME := trial_
CHECKPOINTS_FIG_DIR := $(FIG_DIR)/checkpoints

IDX := $(shell ls -d "$(CHECKPOINTS_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)"* 2>/dev/null | \
        sed "s|.*$(CHECKPOINTS_TRIAL_BASENAME)||" | \
        sort -n | \
        tail -1)

# Set default if empty
IDX ?= 0
NEXT_IDX = $(shell echo $$(($(IDX) + 1)))
CHECKPOINTS_DIR_CURRENT := $(CHECKPOINTS_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)$(IDX)
CHECKPOINTS_DIR_NEXT := $(CHECKPOINTS_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)$(NEXT_IDX)
CHECKPOINTS_FIG_DIR_NEXT := $(CHECKPOINTS_FIG_DIR)/$(CHECKPOINTS_TRIAL_BASENAME)$(NEXT_IDX)

PATHS_FILE = $(SRC_DIR)/paths.json

.PHONY: config gendata run sigma sizes neighs gif help

config:
	@echo "Creating paths file..."
	@echo '{' > $(PATHS_FILE)
	@echo '	"dataset2d": "$(DATASET_DIR_2D)",' >> $(PATHS_FILE)
	@echo '	"dataset3d": "$(DATASET_DIR_3D)",' >> $(PATHS_FILE)
	@echo '	"checkpoints": "$(CHECKPOINTS_DIR_CURRENT)",' >> $(PATHS_FILE)
	@echo '	"checkpoints_next": "$(CHECKPOINTS_DIR_NEXT)",' >> $(PATHS_FILE)
	@echo '	"checkpoints_fig_next": "$(CHECKPOINTS_FIG_DIR_NEXT)",' >> $(PATHS_FILE)
	@echo '	"results_sizes": "$(RESULTS_DIR_SIZES)",' >> $(PATHS_FILE)
	@echo '	"results_neighbors": "$(RESULTS_DIR_NEIGHBORS)",' >> $(PATHS_FILE)
	@echo '	"results_scalefactors": "$(RESULTS_DIR_SCALEFACTORS)"' >> $(PATHS_FILE)
	@echo '}' >> $(PATHS_FILE)

gendata: config
	@echo "Generating Swiss roll dataset..."
	$(PYTHON) $(SRC_DIR)/dataset_generation.py --paths $(PATHS_FILE)

run: config
	@echo "Running manifold sculpting..."
	$(PYTHON) $(SRC_DIR)/run.py --paths $(PATHS_FILE)

gif: config
	@echo "Generating GIF from checkpoints..."
	$(PYTHON) $(SRC_DIR)/generate_gif.py --paths $(PATHS_FILE)

sigma: config
	$(PYTHON) $(SRC_DIR)/mse_vs_sigma.py --paths $(PATHS_FILE)

sizes: config
	$(PYTHON) $(SRC_DIR)/mse_vs_sizes.py --paths $(PATHS_FILE)

neighs: config
	$(PYTHON) $(SRC_DIR)/mse_vs_neighbors.py --paths $(PATHS_FILE)

help:
	@echo "Makefile commands:"
	@echo "  config   - Configure paths and directories"
	@echo "  gendata  - Generate Swiss roll dataset"
	@echo "  run      - Run manifold sculpting"
	@echo "  sigma    - Run manifold sculpting vs sigma"
	@echo "  sizes    - Run manifold sculpting vs dataset sizes"
	@echo "  neighs   - Run manifold sculpting vs number of neighbors"
	@echo "  help     - Show this help message"