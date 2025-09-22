AESTHETIC 2.0 (fresh)
- URL/local ingest, hybrid k-medoids + facility-location selector
- Modular pipeline
- Optional ML via requirements-ml-optional.txt (Torch, Transformers, etc.)

Quick start
-----------
1) C:\Python\Python312\python.exe -m venv .venv
   .\.venv\Scripts\Activate.ps1
2) pip install -r requirements.txt
3) Ensure FFmpeg in PATH or set in config.yaml
4) python app.py

ML/LLM
------
Install optional deps when needed:
   pip install -r requirements-ml-optional.txt
Models and LLM helpers live in aesthetic/models (registry, loaders, stubs).
