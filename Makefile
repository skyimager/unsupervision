SHELL := $(shell which bash)

CONDA_BIN = $(shell which conda)
CONDA_ROOT = $(shell $(CONDA_BIN) info --base)
CONDA_ENV_NAME ?= "experiment"
CONDA_ENV_PREFIX = $(shell conda env list | grep $(CONDA_ENV_NAME) | sort | awk '{$$1=""; print $$0}' | tr -d '*\| ')
CONDA_ACTIVATE := source $(CONDA_ROOT)/etc/profile.d/conda.sh ; conda activate $(CONDA_ENV_NAME) && PATH=${CONDA_ENV_PREFIX}/bin:${PATH};	

RUN_OS := LINUX
ifeq ($(OS),Windows_NT)
	RUN_OS = WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		RUN_OS = LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		RUN_OS = OSX
	endif
endif

##################
# Env Management #
##################

environment:
	$(CONDA_BIN) remove -n $(CONDA_ENV_NAME) --all -y --force-remove
	$(CONDA_BIN) env create -n $(CONDA_ENV_NAME) -f environment.yml

update_environment:
	$(CONDA_BIN) env update -n $(CONDA_ENV_NAME) -f environment.yml

export_environment:
	$(CONDA_BIN) env export -n $(CONDA_ENV_NAME) | grep -v "^prefix: \|^name: " > environment-exported.yml
	cat environment-exported.yml

##################
# Notebook tasks #
##################
NOTEBOOK_PORT?= 8888

jupyter-install:
	$(CONDA_ACTIVATE) pip install notebook==6.* jupyter_contrib_nbextensions
	$(CONDA_ACTIVATE) jupyter contrib nbextension install --user	

jupyter-start:
	rm -rf nohup.out
	nohup jupyter notebook --port $(NOTEBOOK_PORT) --ip=* --no-browser --allow-root &
	sleep 5
	jupyter notebook list

jupyter-stop:
	jupyter notebook stop $(NOTEBOOK_PORT)
