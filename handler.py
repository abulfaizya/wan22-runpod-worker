import os, json, glob, uuid, base64, tempfile, subprocess, time, shutil
from pathlib import Path
import runpod

# ---------- Optional: validate incoming images ----------
try:
    from PIL import Image  # pip install Pillow
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ---------- S3 config ----------
S3_ENABLED = all(k in os.environ for k in ["S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_BUCKET"])
if S3_ENABLED:
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
        region_name=os.getenv("S3_REGION", "auto"),
    )
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_PRESIGN = os.getenv("S3_PRESIGN", "1") == "1"
    S3_EXPIRE = int(os.getenv("S3_EXPIRE_SECS", "86400"))
    S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE")

# ---------- WAN paths ----------
WAN_DIR = Path("/workspace/Wan2.2")
CKPT_TI2V5B = WAN_DIR / "Wan2.2-TI2V-5B"
OUT_ROOT = WAN_DIR / "outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def _write_temp_image(image_b64: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="wan22_"))
    img_path = tmpdir / "input.png"
    raw = base64.b64decode(image_b64)
    img_path.write_bytes(raw)
    if PIL_AVAILABLE:
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"Invalid image data: {e}")
    return img_path

def _pick_latest_mp4() -> Path:
    candidates = list(OUT_ROOT.rglob("*.mp4")) + list(WAN_DIR.glob("*.mp4"))
    if not candidates:
        raise RuntimeError("No MP4 found after generation.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _upload_s3(local_path: Path) -> str:
    key = f"wan22/{uuid.uuid4()}.mp4"
    s3.upload_file(str(local_path), S3_BUCKET, key)
    if S3_PRESIGN:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=S3_EXPIRE,
        )
    if S3_PUBLIC_BASE:
        return f"{S3_PUBLIC_BASE.rstrip('/')}/{key}"
    ep = os.getenv("S3_ENDPOINT", "").rstrip("/")
    return f"{ep}/{S3_BUCKET}/{key}"

def _run_wan(task: str, size: str, prompt: str, image_path: Path | None):
    task_lower = task.lower()
    if task_lower == "ti2v-5b":
        ckpt_dir = CKPT_TI2V5B
        cli_task = "ti2v-5B"
        extra = ["--offload_model", "True", "--convert_model_dtype", "--t5_cpu"]
    elif task_lower.startswith("i2v"):
        ckpt_dir = WAN_DIR / "Wan2.2-I2V-A14B"
        cli_task = "i2v-A14B"
        extra = []
        if image_path is None:
            raise RuntimeError("image_b64 is required for Image→Video tasks.")
    else:
        raise RuntimeError(f"Unsupported task '{task}'. Try 'ti2v-5B' or 'i2v-A14B'.")

    cmd = [
        "python3", str(WAN_DIR / "generate.py"),
        "--task", cli_task,
        "--size", size,
        "--ckpt_dir", str(ckpt_dir),
        "--prompt", prompt,
    ] + extra
    if image_path is not None:
        cmd += ["--image", str(image_path)]

    env = os.environ.copy()
    env["WAN_OUTPUT_DIR"] = str(OUT_ROOT)
    print("Running WAN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(WAN_DIR), check=True, env=env)

    return _pick_latest_mp4()

def handler(job):
    try:
        payload = job.get("input", {}) or {}
        task   = str(payload.get("task", "ti2v-5B"))
        size   = str(payload.get("size", "1280*704"))
        prompt = str(payload.get("prompt", "")).strip()
        image_b64 = payload.get("image_b64")

        if not prompt:
            return {"status": "ERROR", "error": "Missing prompt."}

        # progress: starting
        runpod.serverless.progress_update(job, {"pct": 2, "msg": "Starting job"})

        tmp_img = None
        try:
            if image_b64:
                runpod.serverless.progress_update(job, {"pct": 6, "msg": "Decoding image"})
                tmp_img = _write_temp_image(image_b64)

            runpod.serverless.progress_update(job, {"pct": 12, "msg": "Loading / preparing models"})
            runpod.serverless.progress_update(job, {"pct": 25, "msg": "Generating video… (this can take minutes)"})
            video_path = _run_wan(task=task, size=size, prompt=prompt, image_path=tmp_img)

            runpod.serverless.progress_update(job, {"pct": 85, "msg": "Uploading result"})
            if S3_ENABLED:
                url = _upload_s3(video_path)
                runpod.serverless.progress_update(job, {"pct": 98, "msg": "Finishing"})
                return {"status": "COMPLETED", "task": task, "size": size, "video_url": url}

            b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
            runpod.serverless.progress_update(job, {"pct": 98, "msg": "Finishing"})
            return {"status": "COMPLETED", "task": task, "size": size, "video_base64": b64}

        finally:
            if tmp_img:
                shutil.rmtree(tmp_img.parent, ignore_errors=True)

    except subprocess.CalledProcessError as e:
        return {"status": "ERROR", "error": f"WAN CLI failed ({e.returncode}). Check logs."}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

runpod.serverless.start({"handler": handler})
