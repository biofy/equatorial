#!/bin/bash

pip install -r requirements.txt
python3 exercises/scripts/fetch_data.py

jupyter lab \
    --ip=0.0.0.0 \
    --port=8889 \
    --allow-root \
    --no-browser \
    --NotebookApp.token='' \
    --notebook-dir=/workspace/repos \
    --NotebookApp.allow_origin='*'
