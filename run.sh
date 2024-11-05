#!/bin/bash

if ! command -v python &> /dev/null; then
    echo "Python is not found."
    exit 1
fi

if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

export PYSPARK_PYTHON=venv/bin/python

export PYSPARK_DRIVER_PYTHON=venv/bin/python

if [ ! -d "model" ]; then
    python train_model.py
fi

wave run app

deactivate
