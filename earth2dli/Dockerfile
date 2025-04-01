FROM nvcr.io/nvidia/modulus/modulus:25.03

RUN mkdir -p /workspace/earth2dli
COPY requirements.txt /workspace/earth2dli/requirements.txt
RUN pip install --upgrade ipywidgets jupyterlab==4.2.0 jupyterlab-widgets jupyter-archive jupyter-resource-usage
RUN pip install --upgrade pip && pip install -r /workspace/earth2dli/requirements.txt

COPY exercises /workspace/earth2dli/exercises
COPY README.md /workspace/earth2dli/README.md

RUN python3 /workspace/earth2dli/exercises/scripts/fetch_data.py
