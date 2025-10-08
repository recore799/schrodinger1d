#!/bin/bash

# Script to launch JupyterLab with the correct project path

# Set PYTHONPATH to the project root directory
export PYTHONPATH="$(pwd)"

# Launch JupyterLab, pointing it to the notebooks directory
jupyter lab --notebook-dir=./notebooks
