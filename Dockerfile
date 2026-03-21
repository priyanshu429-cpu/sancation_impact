FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install numpy==1.26.4
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

RUN pip install torch-geometric
RUN pip install transformers==4.41.2
RUN pip install fastapi uvicorn[standard] pydantic networkx scikit-learn

COPY . /app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]