import logging

from fastapi import FastAPI

from app.api.v1.routes import upload

app = FastAPI(
    title="File Upload Service",
)

# Inclure les routes
app.include_router(upload.router, tags=["upload"])
