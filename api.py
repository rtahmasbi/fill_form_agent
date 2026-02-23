"""
api.py — FastAPI interface for the Form-Filling Agent
======================================================

Flow
----
1. POST /upload              → upload passport + G-28 files, returns job_id
2. POST /extract/{job_id}    → runs extrcat_info.py on each file, merges JSON,
                               writes user_info4.txt, returns merged dict
3. POST /fill/{job_id}       → calls main.py with the saved user_info4.txt
4. GET  /status/{job_id}     → poll logs / status of the fill job
5. GET  /jobs                → list all jobs

Run
---
    uvicorn api:app --reload --port 8000

Then open http://localhost:8000      for the UI
or  http://localhost:8000/docs       for Swagger.

Requirements
------------
    pip install fastapi uvicorn python-multipart openai playwright
    playwright install chromium
"""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ── paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent
JOBS_DIR  = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

MAIN_PY       = BASE_DIR / "main.py"
EXTRCAT_PY    = BASE_DIR / "extrcat_info.py"
TARGET_URL    = "https://mendrika-alma.github.io/form-submission/"
HEADLESS      = "false"   # set to "true" to hide the browser window

# ── in-memory job store ───────────────────────────────────────────────────────

jobs: Dict[str, dict] = {}   # job_id → metadata


def _job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    d.mkdir(exist_ok=True)
    return d


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Form-Filling Agent API",
    description="Upload passport + G-28, extract info with GPT-4o, auto-fill the web form.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────

ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}


def _validate_file(file: UploadFile):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )


async def _save_upload(file: UploadFile, dest: Path) -> Path:
    content = await file.read()
    dest.write_bytes(content)
    return dest


def _run_extraction(file_path: Path) -> str:
    """
    Run extrcat_info.py on a single file and return raw stdout.
    Raises RuntimeError if the process exits non-zero.
    """
    result = subprocess.run(
        [sys.executable, str(EXTRCAT_PY), "--input_file", str(file_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"extrcat_info.py failed for {file_path.name}:\n{result.stderr}"
        )
    return result.stdout.strip()


def _parse_extraction(raw: str) -> dict:
    """
    Try to parse the raw string from extrcat_info.py into a dict.
    Handles: JSON string, markdown-fenced JSON, Python dict repr.
    Falls back to {"raw_text": raw} so we never lose data.
    """
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    import ast
    try:
        result = ast.literal_eval(clean)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {"raw_text": raw}


def _merge_dicts(outputs: list[dict]) -> dict:
    """Merge a list of dicts into one, later keys overwrite earlier ones."""
    merged: dict = {}
    for d in outputs:
        if isinstance(d, dict):
            merged.update(d)
    return merged


def _dict_to_user_info_txt(data: dict) -> str:
    """
    Flatten extraction dict to plain text that main.py's LLM agent can read.

    Example output:
        surname: Doe
        given_names: John
        passport_number: A1234567
        ...
    """
    lines = []

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            lines.append(f"{prefix.rstrip('.')}: {', '.join(str(i) for i in obj)}")
        else:
            lines.append(f"{prefix.rstrip('.')}: {obj}")

    _flatten(data)
    return "\n".join(lines)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload", summary="Upload passport and/or G-28 documents")
async def upload(
    passport: Optional[UploadFile] = File(None, description="Passport image (JPG/PNG/WEBP)"),
    g28:      Optional[UploadFile] = File(None, description="G-28 form (PDF or image)"),
):
    """
    Upload one or both documents.
    Returns a **job_id** to use in all subsequent calls.
    At least one file is required.
    """
    if passport is None and g28 is None:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one file (passport or g28).",
        )

    job_id  = str(uuid.uuid4())[:8]
    job_dir = _job_dir(job_id)
    files_saved = []

    if passport:
        _validate_file(passport)
        dest = job_dir / f"passport{Path(passport.filename).suffix.lower()}"
        await _save_upload(passport, dest)
        files_saved.append(str(dest))

    if g28:
        _validate_file(g28)
        dest = job_dir / f"g28{Path(g28.filename).suffix.lower()}"
        await _save_upload(g28, dest)
        files_saved.append(str(dest))

    jobs[job_id] = {
        "job_id":         job_id,
        "created_at":     datetime.utcnow().isoformat(),
        "files":          files_saved,
        "status":         "uploaded",
        "extracted":      None,
        "user_info_path": None,
        "fill_log":       None,
    }

    return {"job_id": job_id, "files_saved": files_saved}


@app.post("/extract/{job_id}", summary="Extract info from uploaded documents")
async def extract(job_id: str):
    """
    Runs **extrcat_info.py** (via GPT-4o) on every uploaded file, merges the
    results into a single dict, and writes **user_info4.txt** inside the job
    folder for main.py to consume.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if not job["files"]:
        raise HTTPException(status_code=400, detail="No files to extract from.")

    try:
        job["status"] = "extracting"

        # Run extraction for each file (offloaded to thread so FastAPI stays async)
        raw_outputs = []
        for file_path in job["files"]:
            raw = await asyncio.to_thread(_run_extraction, Path(file_path))
            raw_outputs.append(raw)

        # Parse + merge
        parsed  = [_parse_extraction(r) for r in raw_outputs]
        merged  = _merge_dicts(parsed)

        # Write user_info4.txt
        user_info_txt = _dict_to_user_info_txt(merged)
        txt_path      = _job_dir(job_id) / "user_info4.txt"
        txt_path.write_text(user_info_txt, encoding="utf-8")

        # Write extracted.json for inspection
        json_path = _job_dir(job_id) / "extracted.json"
        json_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

        job["extracted"]      = merged
        job["user_info_path"] = str(txt_path)
        job["status"]         = "extracted"

        return {
            "job_id":        job_id,
            "extracted":     merged,
            "user_info_txt": user_info_txt,
        }

    except Exception as e:
        job["status"] = f"error: {e}"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fill/{job_id}", summary="Launch main.py to fill the web form")
async def fill(job_id: str):
    """
    Launches:

        python main.py \\
            --url https://mendrika-alma.github.io/form-submission/ \\
            --user_info <job_dir>/user_info4.txt \\
            --headless false

    The process runs in the background.
    Poll **GET /status/{job_id}** for live log output.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if job["status"] != "extracted":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot fill: job status is '{job['status']}'. Run POST /extract first.",
        )

    user_info_path = job.get("user_info_path")
    if not user_info_path or not Path(user_info_path).exists():
        raise HTTPException(
            status_code=400,
            detail="user_info4.txt missing. Run POST /extract first.",
        )

    log_path          = _job_dir(job_id) / "fill.log"
    job["status"]     = "filling"
    job["fill_log"]   = str(log_path)

    async def _run_fill():
        cmd = [
            sys.executable, str(MAIN_PY),
            "--url",       TARGET_URL,
            "--user_info", user_info_path,
            "--headless",  HEADLESS,
        ]
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_f,
                stderr=log_f,
            )
            await proc.wait()

        jobs[job_id]["status"] = (
            "done" if proc.returncode == 0
            else f"fill_failed (exit {proc.returncode})"
        )

    asyncio.create_task(_run_fill())

    return {
        "job_id":  job_id,
        "message": "Form fill started. Poll GET /status/{job_id} for live output.",
        "log":     str(log_path),
    }


@app.get("/status/{job_id}", summary="Get job status and live log tail")
async def status(job_id: str, tail: int = 50):
    """Returns job metadata and the last `tail` lines of the fill log."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job   = {k: v for k, v in jobs[job_id].items() if k != "extracted"}
    lines: list[str] = []

    log_path = job.get("fill_log")
    if log_path and Path(log_path).exists():
        all_lines = Path(log_path).read_text(encoding="utf-8").splitlines()
        lines     = all_lines[-tail:]

    return {**job, "log_tail": lines}


@app.get("/jobs", summary="List all jobs (without extracted payload)")
async def list_jobs():
    return [
        {k: v for k, v in j.items() if k != "extracted"}
        for j in jobs.values()
    ]


# ── browser UI ────────────────────────────────────────────────────────────────

UI_FILE = BASE_DIR / "index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    """Serve the standalone index.html UI from the same directory as api.py."""
    if not UI_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found. Place it in the same folder as api.py.",
        )
    return HTMLResponse(content=UI_FILE.read_text(encoding="utf-8"))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
