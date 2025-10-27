
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime


WORKDIR /workspace
COPY wang_landau_walkers.py utils.py config.py /workspace


CMD ["python", "wang_landau_walkers.py", "-u"]
