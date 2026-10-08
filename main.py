"""
Sentry QC - FastAPI Backend
---------------------------
Main application entrypoint.
Exports `app` from the modular application defined in `app.main`.

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  # or
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  # or
  python main.py
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
