FROM python:3.10-slim

WORKDIR /app

# copy ONLY requirements first (important for cache)
COPY requirements.txt /app/requirements.txt

# system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip

# install correct numpy first
RUN pip install numpy==1.26.4

# install PyTorch CPU
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# install PyG wheels
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
RUN pip install torch-geometric

# install transformers COMPATIBLE VERSION
RUN pip install transformers==4.41.2

# install remaining dependencies
RUN pip install fastapi uvicorn[standard] pydantic networkx scikit-learn

# NOW copy your project code
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]