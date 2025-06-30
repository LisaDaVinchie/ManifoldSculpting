BASE_DIR := $(shell pwd)
PYTHON := $(shell which python3)

SRC_DIR := $(BASE_DIR)/src
DATA_DIR := $(BASE_DIR)/data

DATASET_DIR := $(DATA_DIR)/datasets
DATASET_DIR_3D := $(DATASET_DIR)/3d
DATASET_DIR_2D := $(DATASET_DIR)/2d

PATHS_FILE = $(SRC_DIR)/paths.json

.PHONY: gendata

config :
	@echo "Creating directories..."
	mkdir -p $(DATASET_DIR_3D)
	mkdir -p $(DATASET_DIR_2D)
	@echo "Creating paths file..."
	@echo '{' > $(PATHS_FILE)
	@echo '"dataset2d": "$(DATASET_DIR_2D)",' >> $(PATHS_FILE)
	@echo '"dataset3d": "$(DATASET_DIR_3D)"' >> $(PATHS_FILE)
	@echo '}' >> $(PATHS_FILE)

gendata: config
	@echo "Generating Swiss roll dataset..."
	$(PYTHON) $(SRC_DIR)/dataset_generation.py --paths $(PATHS_FILE)