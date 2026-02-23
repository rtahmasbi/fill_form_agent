"""
api.py — FastAPI interface for the Form-Filling Agent
======================================================

Flow
----
1. POST /upload              → upload passport + G-28 files, returns job_id
2. POST /extract/{job_id}    → runs extrcat_info.py on each file, merges JSON,
                               writes user_info4.txt, returns merged dict
3. POST /fill/{job_id}       → launches main.py, streams stdout/stderr to fill.log,
                               detects the APPROVAL_REQUIRED sentinel, pauses
4. GET  /status/{job_id}     → poll status; when awaiting_approval, returns
                               fill_summary so the UI can show it
5. POST /approve/{job_id}    → send decision=yes|no, resumes main.py via stdin
6. GET  /jobs                → list all jobs

How the approval handshake works
---------------------------------
main.py prints a sentinel line:
    __APPROVAL_REQUIRED__
followed by the fill summary, then blocks on input().

api.py watches stdout for that sentinel, sets status="awaiting_approval",
captures the summary, and keeps the process alive (stdin pipe open).
When POST /approve is called it writes "yes\n" or "no\n" to stdin.

Run
---
    uvicorn api:app --reload --port 8000

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

BASE_DIR   = Path(__file__).parent
JOBS_DIR   = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

MAIN_PY    = BASE_DIR / "main.py"
EXTRCAT_PY = BASE_DIR / "extrcat_info.py"
TARGET_URL = "https://mendrika-alma.github.io/form-submission/"
HEADLESS   = "false"

# Sentinel that main.py prints just before blocking on input()
APPROVAL_SENTINEL = "__APPROVAL_REQUIRED__"

# ── in-memory job store ───────────────────────────────────────────────────────

jobs: Dict[str, dict] = {}
# Holds live asyncio.subprocess.Process objects keyed by job_id
_procs: Dict[str, asyncio.subprocess.Process] = {}

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
    result = subprocess.run(
        [sys.executable, str(EXTRCAT_PY), "--input_file", str(file_path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"extrcat_info.py failed for {file_path.name}:\n{result.stderr}")
    return result.stdout.strip()

def _parse_extraction(raw: str) -> dict:
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
    merged: dict = {}
    for d in outputs:
        if isinstance(d, dict):
            merged.update(d)
    return merged

def _dict_to_user_info_txt(data: dict) -> str:
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
    if passport is None and g28 is None:
        raise HTTPException(status_code=400, detail="Upload at least one file (passport or g28).")

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
        "fill_summary":   None,   # captured from main.py before approval prompt
    }
    return {"job_id": job_id, "files_saved": files_saved}


@app.post("/extract/{job_id}", summary="Extract info from uploaded documents")
async def extract(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if not job["files"]:
        raise HTTPException(status_code=400, detail="No files to extract from.")
    try:
        job["status"] = "extracting"
        raw_outputs = []
        for file_path in job["files"]:
            raw = await asyncio.to_thread(_run_extraction, Path(file_path))
            raw_outputs.append(raw)

        parsed        = [_parse_extraction(r) for r in raw_outputs]
        merged        = _merge_dicts(parsed)
        user_info_txt = _dict_to_user_info_txt(merged)

        txt_path = _job_dir(job_id) / "user_info4.txt"
        txt_path.write_text(user_info_txt, encoding="utf-8")

        json_path = _job_dir(job_id) / "extracted.json"
        json_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

        job["extracted"]      = merged
        job["user_info_path"] = str(txt_path)
        job["status"]         = "extracted"

        return {"job_id": job_id, "extracted": merged, "user_info_txt": user_info_txt}

    except Exception as e:
        job["status"] = f"error: {e}"
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fill/{job_id}", summary="Launch main.py — pauses at approval step")
async def fill(job_id: str):
    """
    Starts main.py as a subprocess with stdin/stdout pipes.
    Streams stdout lines to fill.log AND watches for the
    __APPROVAL_REQUIRED__ sentinel.  When detected the job
    transitions to status='awaiting_approval' and the process
    is kept alive waiting for POST /approve/{job_id}.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if job["status"] != "extracted":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot fill: status is '{job['status']}'. Run /extract first.",
        )

    user_info_path = job.get("user_info_path")
    if not user_info_path or not Path(user_info_path).exists():
        raise HTTPException(status_code=400, detail="extracted info are missing. Run /extract first.")

    log_path        = _job_dir(job_id) / "fill.log"
    job["status"]   = "filling"
    job["fill_log"] = str(log_path)

    async def _stream_and_watch():
        cmd = [
            sys.executable, str(MAIN_PY),
            "--url",       TARGET_URL,
            "--user_info", user_info_path,
            "--headless",  HEADLESS,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,   # keep stdin open for approval
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, # merge stderr into stdout
        )
        _procs[job_id] = proc

        summary_lines: list[str] = []
        capturing_summary = False

        with open(log_path, "w", encoding="utf-8") as log_f:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break  # process closed stdout

                line = line_bytes.decode("utf-8", errors="replace")
                log_f.write(line)
                log_f.flush()

                stripped = line.strip()

                # ── detect approval sentinel ──────────────────────────────
                if stripped == APPROVAL_SENTINEL:
                    capturing_summary = True
                    continue

                # ── detect the interactive prompt line ────────────────────
                # main.py prints "Your decision (yes/no): " then blocks
                if capturing_summary and "Your decision" in stripped:
                    # Done capturing — surface to UI
                    jobs[job_id]["fill_summary"] = "\n".join(summary_lines).strip()
                    jobs[job_id]["status"]        = "awaiting_approval"
                    # Don't break — keep reading (process is blocked on stdin)
                    capturing_summary = False
                    continue

                if capturing_summary:
                    summary_lines.append(stripped)

        # Process exited (after approval or cancellation)
        rc = await proc.wait()
        _procs.pop(job_id, None)
        if jobs[job_id]["status"] not in ("approved", "declined"):
            jobs[job_id]["status"] = "done" if rc == 0 else f"fill_failed (exit {rc})"

    asyncio.create_task(_stream_and_watch())

    return {
        "job_id":  job_id,
        "message": "Form fill started. Poll GET /status/{job_id} — will pause for your approval.",
    }


@app.post("/approve/{job_id}", summary="Approve or decline form submission")
async def approve(job_id: str, decision: str = "yes"):
    """
    Send **decision=yes** to submit the form, or **decision=no** to cancel.
    Only valid when job status is `awaiting_approval`.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not awaiting approval (status: '{job['status']}').",
        )

    proc = _procs.get(job_id)
    if proc is None or proc.stdin is None:
        raise HTTPException(status_code=500, detail="Process not found or stdin unavailable.")

    decision_clean = decision.strip().lower()
    if decision_clean not in ("yes", "y", "no", "n"):
        raise HTTPException(status_code=400, detail="decision must be 'yes' or 'no'.")

    # Write the decision to main.py's stdin (replaces the terminal input() call)
    proc.stdin.write((decision_clean + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()

    if decision_clean in ("yes", "y"):
        jobs[job_id]["status"] = "approved"
        msg = "Approval sent — form is being submitted."
    else:
        jobs[job_id]["status"] = "declined"
        msg = "Declined — form submission cancelled."

    return {"job_id": job_id, "decision": decision_clean, "message": msg}


@app.get("/status/{job_id}", summary="Get job status and live log tail")
async def status(job_id: str, tail: int = 60):
    """
    Returns job metadata and the last `tail` lines of the fill log.
    When status is `awaiting_approval`, also returns `fill_summary`
    so the UI can display the form field values for review.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job   = {k: v for k, v in jobs[job_id].items() if k != "extracted"}
    lines: list[str] = []

    log_path = job.get("fill_log")
    if log_path and Path(log_path).exists():
        all_lines = Path(log_path).read_text(encoding="utf-8").splitlines()
        lines     = all_lines[-tail:]

    return {**job, "log_tail": lines}


@app.get("/jobs", summary="List all jobs")
async def list_jobs():
    return [
        {k: v for k, v in j.items() if k != "extracted"}
        for j in jobs.values()
    ]


# ── browser UI ────────────────────────────────────────────────────────────────

UI_FILE = BASE_DIR / "index.html"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html not found next to api.py.")
    return HTMLResponse(content=UI_FILE.read_text(encoding="utf-8"))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
