# app/main.py
from fastapi import FastAPI
from app.routers import upload
from app.embeddings import worker
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()
app.include_router(upload.router)

@app.on_event("startup")
async def startup_event():
    worker.start_workers()

@app.on_event("shutdown")
async def shutdown_event():
    await worker.stop_workers()