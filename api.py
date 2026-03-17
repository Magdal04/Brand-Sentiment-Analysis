from fastapi import FastAPI
from run_pipeline import run_pipeline
import json

app = FastAPI()

@app.post("/run")
def trigger_pipeline():
    run_pipeline()

    with open("data/processed/ai_payload.json") as f:
        report = json.load(f)

    return {
        "status": "success",
        "report": report,   
    }
