FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Only needed if you want to keep the files unchanged.
WORKDIR /workspace
# Ensure BOTH specialized walker scripts, utils, and config are copied
COPY wang_landau_walkers.py utils.py config.py /workspace
# Install build tools for any C++ extensions or specialized layers
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
ENV CC=gcc
ENV CXX=g++

# Note: CMD is not critical here as it's overridden by the SLURM script
#CMD ["python", "wang_landau_walkers_A.py", "-u"]
