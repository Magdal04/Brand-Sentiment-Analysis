import os
import json
import logging
import threading
import uuid
from datetime import datetime
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from run_pipeline import run_pipeline


logger = logging.getLogger(__name__)

app = FastAPI(title="Brand Sentiment Analysis API")


class PipelineInput(BaseModel):
    """Expected n8n payload (application/json request body)."""

    brand: str = Field(None, description="The brand being analyzed")
    youtube_queries: Union[str, List[str]] = Field(
        ..., description="YouTube query or list of queries"
    )
    news_queries: Union[str, List[str]] = Field(
        ..., description="News query or list of queries"
    )


def _to_list(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


# Minimal in-memory job store (single container / single process).
JOBS: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run")
async def execute(data: PipelineInput):
    """Queue the pipeline and return immediately (prevents n8n infinite loading).

    Poll /status/{job_id} and then /result/{job_id}.
    """

    youtube_queries = _to_list(data.youtube_queries)
    news_queries = _to_list(data.news_queries)
    if not youtube_queries and not news_queries:
        raise HTTPException(
            status_code=400,
            detail="youtube_queries and/or news_queries must be non-empty",
        )

    # Make the request-provided queries visible to collectors.
    os.environ["BRAND"] = data.brand or ""
    os.environ["YOUTUBE_QUERIES"] = ",".join(youtube_queries)
    os.environ["NEWS_QUERIES"] = ",".join(news_queries)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat(),
        "finished_at": None,
        "status": "queued",
        "error": None,
        "youtube_queries": youtube_queries,
        "news_queries": news_queries,
    }

    def _worker():
        JOBS[job_id]["status"] = "running"
        try:
            logger.info(
                "Job %s starting. youtube=%s news=%s",
                job_id,
                youtube_queries,
                news_queries,
            )
            run_pipeline(skip_collection=False)
            JOBS[job_id]["status"] = "succeeded"
        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
            logger.exception("Job %s failed", job_id)
        finally:
            JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()

    threading.Thread(target=_worker, daemon=True).start()

    return {
        "status": "accepted",
        "job_id": job_id,
        "poll": {
            "status": f"/status/{job_id}",
            "result": f"/result/{job_id}",
        },
    }


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }


@app.get("/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    if job["status"] in {"queued", "running"}:
        raise HTTPException(status_code=202, detail="Job still running")
    if job["status"] == "failed":
        raise HTTPException(status_code=500, detail=job.get("error") or "Job failed")

    try:
        with open("data/processed/ai_payload.json", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Result file not found")

    return {
        "job_id": job_id,
        "status": "completed",
        "report": report,
    }
