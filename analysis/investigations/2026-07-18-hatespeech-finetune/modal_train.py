"""Modal app: run any investigation script on an A100 with out/ on a Volume.

    modal run modal_train.py --cmd "python 09_dapt.py --full ..."
    modal run --detach modal_train.py --cmd "bash run_d_batch.sh"

Volume `hatespeech-finetune` mounts at out/; scripts are baked into the
image (redeploy picks up local edits automatically). HF model cache lives
on the volume so weights download once.
"""

import modal

app = modal.App("hatespeech-finetune")
vol = modal.Volume.from_name("hatespeech-finetune", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pandas>=2",
        "pyarrow>=18",
        "scikit-learn>=1.4",
        "torch>=2.4",
        "transformers>=4.56",
        "accelerate>=1.0",
        "matplotlib>=3.8",
        "huggingface_hub>=0.30",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=["out/**", "out", "__pycache__/**", "*.ipynb", ".DS_Store"],
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/root/app/out": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=12 * 3600,
)
def run(cmd: str) -> int:
    import os
    import subprocess
    import threading
    import time

    os.environ["HF_HOME"] = "/root/app/out/hf-cache"

    def committer():
        while True:
            time.sleep(300)
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    rc = subprocess.run(["bash", "-c", cmd], cwd="/root/app").returncode
    vol.commit()
    return rc


@app.local_entrypoint()
def main(cmd: str, spawn: bool = False):
    if spawn:
        call = run.spawn(cmd)
        print(f"spawned: {call.object_id}")
        return
    rc = run.remote(cmd)
    if rc != 0:
        raise SystemExit(rc)
