"""Local-only FastAPI server and streaming workflow events."""
import logging
import os
import queue
import threading
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langsmith import tracing_context

from .settings import ROOT, tracing_enabled
from .retrieval import build_indexes, index_ready, PDFS
from .workflow import execute

app = FastAPI(title='TenantFlow', version='1.0.0')
app.mount('/assets', StaticFiles(directory=ROOT / 'frontend'), name='assets')
WORK_LOCK = threading.Lock()


class RunRequest(BaseModel):
    query: str = Field(min_length=3, max_length=6000)
    mode: str = Field(default='demo', pattern='^(demo|live)$')
    demo_retry: bool = True


@app.get('/')
def home():
    return FileResponse(ROOT / 'frontend' / 'index.html')


@app.get('/api/status')
def status():
    return {'openai_configured': bool(os.getenv('OPENAI_API_KEY')), 'tracing_enabled': tracing_enabled(), 'project': os.getenv('LANGSMITH_PROJECT'), 'index_ready': index_ready(), 'pdfs': {kind: (ROOT / 'data' / value[0]).exists() for kind, value in PDFS.items()}}


def stream_job(worker):
    if not WORK_LOCK.acquire(blocking=False):
        raise HTTPException(409, 'A run or index build is already in progress. Wait for it to finish.')
    events = queue.Queue()

    def run():
        try:
            events.put(worker(events.put))
        except Exception:
            # Provider errors may include account or endpoint details; keep them
            # in the local server log, not the browser response.
            logging.exception('TenantFlow operation failed')
            events.put({'type': 'error', 'message': 'The operation failed. Check the local terminal for details. Verify API credentials, endpoint/model support, and PDF index status.'})
        finally:
            events.put(None)
            WORK_LOCK.release()

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield json.dumps({'type': 'heartbeat'}) + '\n'
                continue
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False) + '\n'

    return StreamingResponse(generate(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


@app.post('/api/run')
def run_workflow(body: RunRequest):
    query = body.query.strip()
    if len(query) < 3:
        raise HTTPException(422, 'Enter a question with at least three non-whitespace characters.')
    if body.mode == 'live':
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(400, 'Set OPENAI_API_KEY in .env and restart the server.')
        if not index_ready():
            raise HTTPException(400, 'Build the PDF index before starting a live run.')
    return stream_job(lambda emit: execute(query, body.mode, body.demo_retry, emit))


@app.post('/api/index')
def index():
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(400, 'Set OPENAI_API_KEY in .env and restart the server.')
    if not all((ROOT / 'data' / item[0]).exists() for item in PDFS.values()):
        raise HTTPException(400, 'Add tenants_rights.pdf and previous_judgements.pdf to data/.')

    def worker(emit):
        emit({'type': 'step', 'step': 'index', 'title': 'Loading PDFs, splitting text, and building persistent embeddings…'})
        with tracing_context(enabled=tracing_enabled()):
            counts = build_indexes()
        return {'type': 'indexed', **counts}

    return stream_job(worker)
