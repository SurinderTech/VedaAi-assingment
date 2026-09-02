import asyncio
import modal

app = modal.App("vedaai-backend")

# ── Persistent Volume ──────────────────────────────────────────────────────────
# Stores uploaded files + store_metadata.json across all container restarts.
# This is the only Modal-native storage needed — no external database required.
# With max_containers=1, in-memory dicts handle intra-session requests;
# the Volume is the safety net that makes the workflow reliable after any restart.
volume = modal.Volume.from_name("vedaai-storage", create_if_missing=True)
VOLUME_PATH = "/vedaai_data"

# ── Container Image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # poppler-utils → pdf2image PDF rendering fallback (pdftoppm binary)
    # libglib2.0-0  → OpenCV / image library dependency on Debian
    # libgl1        → OpenCV headless (cv2) import dependency
    .apt_install(["poppler-utils", "libglib2.0-0", "libgl1"])
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("app", remote_path="/root/app")
)


# ── ASGI Function ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    memory=4096,
    timeout=1800,
    secrets=[modal.Secret.from_name("vedaai-secrets")],
    volumes={VOLUME_PATH: volume},
    # min_containers=1: one container is always running and warm.
    # Maximises in-memory cache hits and eliminates cold-start latency.
    # Combined with max_containers=1 this means exactly one warm container
    # handles all traffic at all times.
    min_containers=0,
    scaledown_window=300,
    # Single container guarantees all requests share the same Python process
    # and the same in-memory _assessments / _statuses / _files dicts.
    max_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    import os

    # ── Storage bootstrap ──────────────────────────────────────────────────────
    # Both store.py and routes.py read VEDAAI_UPLOAD_DIR at import time.
    # Setting it here (before any app imports) routes all file I/O to the
    # persistent Volume so uploaded files and store_metadata.json survive
    # container restarts.
    upload_dir = f"{VOLUME_PATH}/uploads"
    os.environ["VEDAAI_UPLOAD_DIR"] = upload_dir
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(f"{upload_dir}/snapshots", exist_ok=True)

    # Import the FastAPI application (triggers store.py module init, which
    # calls _load_from_disk() and reads any existing JSON from the Volume).
    from app.main import app as fastapi_application

    # ── ASGI wrapper: commit Volume after every HTTP request ───────────────────
    # Why this is needed:
    #   Modal Volume writes are immediately visible within the same container
    #   but are only DURABLE (survive restarts) after volume.commit().
    #   FastAPI BackgroundTasks (run_pipeline) execute INSIDE the ASGI lifecycle
    #   call — before the call returns — so this single commit captures the
    #   fully-processed state including OCR results, grading, and status updates.
    async def asgi_with_commit(scope, receive, send):
        await fastapi_application(scope, receive, send)
        if scope.get("type") == "http":
            try:
                await asyncio.to_thread(volume.commit)
            except Exception as exc:
                # Non-fatal: in-memory state is still correct; only
                # cross-restart durability is affected.
                print(f"[VolumeCommit] Non-fatal commit error: {exc}")

    return asgi_with_commit