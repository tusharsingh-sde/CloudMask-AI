"""
Convenience entrypoint alias exposing the FastAPI application instance.
Allows running:
    uvicorn main:app --reload
as well as:
    uvicorn api:app --reload
"""

import os
import uvicorn
from api import app

__all__ = ["app"]

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run("main:app", host=host, port=port, log_level=log_level, reload=False)
