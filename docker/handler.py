import subprocess
import time
import shutil
import os
import runpod
from pathlib import Path
import sys
from pathlib import Path

# ==========================================================
# VARIABLES
# ==========================================================
# to test in docker before serverless as is faster : cd /workspace/MuseTalk/docker && python3 handler.py
# git clone --branch serverless https://github.com/seb2oo/MuseTalk.git /workspace/MuseTalk
TESTS_IN_DOCKER = True


# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_DIR = Path("/workspace/MuseTalk")
MODELS_DIR = PROJECT_DIR / "models"

HF_CACHE_DIR = Path(
    "/runpod-volume/huggingface-cache/hub"
)

MUSE_TALK_MODEL_ID = "TMElyralab/MuseTalk"





# ==========================================================
# DIRECTORIES
# ==========================================================

print("==========================================")
print(" MuseTalk - Download models")
print("==========================================")

print("")
print("Project directory:")
print(f"  {PROJECT_DIR}")

print("")
print("Models directory:")
print(f"  {MODELS_DIR}")


# ----------------------------------------------------------
# Création des répertoires
# ----------------------------------------------------------

(MODELS_DIR / "musetalkV15").mkdir(
    parents=True,
    exist_ok=True,
)

(MODELS_DIR / "syncnet").mkdir(
    parents=True,
    exist_ok=True,
)

(MODELS_DIR / "dwpose").mkdir(
    parents=True,
    exist_ok=True,
)

(MODELS_DIR / "face-parse-bisent").mkdir(
    parents=True,
    exist_ok=True,
)

(MODELS_DIR / "sd-vae").mkdir(
    parents=True,
    exist_ok=True,
)

(MODELS_DIR / "whisper").mkdir(
    parents=True,
    exist_ok=True,
)





# ============================================================
# HELPERS
# ============================================================

def run_command(command):
    print(f"[COMMAND] {command}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR) + ":" + env.get("PYTHONPATH", "")

    subprocess.run(
        command,
        shell=True,
        check=True,
        env=env,
    )

# ============================================================
# PULL GIT
# ============================================================

def pull_git():
    print("Pulling Fish Speech repository...")

    run_command(
        f"cd '{PROJECT_DIR}' && git pull origin serverless"
    )



# ============================================================
# CACHED MODEL (this projet has several models... we place the bigger one into the cache...)
# ============================================================


def find_cached_model(model_id):

    org, name = model_id.split("/", 1)

    model_dir = (
        HF_CACHE_DIR
        / f"models--{org}--{name}"
    )

    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        raise RuntimeError(
            f"HF cache not found: {snapshots_dir}"
        )

    snapshots = [
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir()
    ]

    if not snapshots:
        raise RuntimeError(
            f"No cached snapshot found for {model_id}"
        )

    if len(snapshots) > 1:
        print(
            f"Found {len(snapshots)} cached snapshots."
        )

    return snapshots[0]

if not TESTS_IN_DOCKER :
    MUSE_TALK_CACHE_PATH = find_cached_model(
        MUSE_TALK_MODEL_ID
    )

    print("MuseTalk cached model found:")
    print(MUSE_TALK_CACHE_PATH)




# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap():

    print("=" * 70)
    print("MUSETALK SERVERLESS BOOTSTRAP")
    print("=" * 70)

    # --------------------------------------------------------
    # CLONE MUSETALK
    # --------------------------------------------------------

    if not os.path.exists(
        os.path.join(PROJECT_DIR, ".git")
    ):

        print("MuseTalk repository not found.")

        if os.path.exists(PROJECT_DIR):
            print(
                f"Removing incomplete directory: {PROJECT_DIR}"
            )

            run_command(
                f"rm -rf '{PROJECT_DIR}'"
            )

        print("Cloning MuseTalk repository...")

        run_command(
            "git clone -b serverless "
            "https://github.com/seb2oo/MuseTalk.git "
            f"'{PROJECT_DIR}'"
        )
        print("1")

    else:

        print(
            "MuseTalk repository already exists."
        )

    

    print("2")
    # Make MuseTalk available to this Python process
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    # --------------------------------------------------------
    # DOWNLOAD CHECKPOINT
    # --------------------------------------------------------

    if not TESTS_IN_DOCKER:
  
        DESTINATION = MODELS_DIR

        print(f"Copying MuseTalk model to: {DESTINATION}")

        # can take time.. to monitor and in other case we give the CHECKPOINT_PATH direytly to the code if possible ie : unet_model_path = Path(CHECKPOINT_PATH) / "musetalkV15" / "unet.pth"
        shutil.copytree(
            MUSE_TALK_CACHE_PATH,
            DESTINATION,
            dirs_exist_ok=True
        )

        print("MuseTalk model copied successfully.")
    else:
        # ==========================================================
        # MuseTalk
        # ==========================================================

        print("")
        print("==========================================")
        print("Downloading MuseTalk...")
        print("==========================================")

        MUSE_TALK_DIR = MODELS_DIR / "musetalkV15"

        run_command([
            "huggingface-cli",
            "download",
            "TMElyralab/MuseTalk",
            "--local-dir",
            str(MUSE_TALK_DIR),
        ])



    # ==========================================================
    # SD VAE
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading SD VAE...")
    print("==========================================")

    SD_VAE_DIR = MODELS_DIR / "sd-vae"

    run_command([
        "huggingface-cli",
        "download",
        "stabilityai/sd-vae-ft-mse",
        "config.json",
        "--local-dir",
        str(SD_VAE_DIR),
    ])

    run_command([
        "huggingface-cli",
        "download",
        "stabilityai/sd-vae-ft-mse",
        "diffusion_pytorch_model.bin",
        "--local-dir",
        str(SD_VAE_DIR),
    ])


    # ==========================================================
    # Whisper
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading Whisper...")
    print("==========================================")

    WHISPER_DIR = MODELS_DIR / "whisper"

    run_command([
        "huggingface-cli",
        "download",
        "openai/whisper-tiny",
        "config.json",
        "--local-dir",
        str(WHISPER_DIR),
    ])

    run_command([
        "huggingface-cli",
        "download",
        "openai/whisper-tiny",
        "pytorch_model.bin",
        "--local-dir",
        str(WHISPER_DIR),
    ])

    run_command([
        "huggingface-cli",
        "download",
        "openai/whisper-tiny",
        "preprocessor_config.json",
        "--local-dir",
        str(WHISPER_DIR),
    ])


    # ==========================================================
    # DWPose
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading DWPose...")
    print("==========================================")

    DWPOSE_DIR = MODELS_DIR / "dwpose"

    run_command([
        "huggingface-cli",
        "download",
        "yzd-v/DWPose",
        "--local-dir",
        str(DWPOSE_DIR),
        "--include",
        "dw-ll_ucoco_384.pth",
    ])


    # ==========================================================
    # LatentSync
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading LatentSync...")
    print("==========================================")

    SYNCNET_DIR = MODELS_DIR / "syncnet"

    run_command([
        "huggingface-cli",
        "download",
        "ByteDance/LatentSync",
        "--local-dir",
        str(SYNCNET_DIR),
        "--include",
        "latentsync_syncnet.pt",
    ])


    # ==========================================================
    # Face Parse
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading face-parse-bisent...")
    print("==========================================")

    FACE_PARSE_DIR = MODELS_DIR / "face-parse-bisent"

    run_command([
        "huggingface-cli",
        "download",
        "ManyOtherFunctions/face-parse-bisent",
        "79999_iter.pth",
        "--local-dir",
        str(FACE_PARSE_DIR),
    ])

    run_command([
        "huggingface-cli",
        "download",
        "ManyOtherFunctions/face-parse-bisent",
        "resnet18-5c106cde.pth",
        "--local-dir",
        str(FACE_PARSE_DIR),
    ])


    # ==========================================================
    # S3FD
    # ==========================================================

    print("")
    print("==========================================")
    print("Downloading S3FD...")
    print("==========================================")

    S3FD_DIR = Path(
        "/root/.cache/torch/hub/checkpoints"
    )

    S3FD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    S3FD_PATH = (
        S3FD_DIR
        / "s3fd-619a316812.pth"
    )

    run_command([
        "wget",
        "-O",
        str(S3FD_PATH),
        "https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth",
    ])


    # ==========================================================
    # DONE
    # ==========================================================

    print("")
    print("==========================================")
    print(" Model download completed")
    print("==========================================")

    print("")
    print("Models are located in:")
    print(f"  {MODELS_DIR}")






## handler final
def handler(job):

    """
    Expected request:

    {
        "input": {
            "text": "Bonjour, ceci est un test.",
            "language": "fr",
            "reload_anything":false,
            "del_cache":false
        }
    }

    Supported languages:
        fr = French
        en = English
        es = Spanish
        de = German
    """

    job_input = job.get(
        "input",
        {},
    )

    text = job_input.get(
        "text"
    )

    language = job_input.get(
        "language"
    )

    reload = job_input.get(
            "reload_anything",False
        )

    del_cache = job_input.get(
                "del_cache",False
            )
    

    if not text:

        raise ValueError(
            "Missing required input: 'text'"
        )

    if not language:

        raise ValueError(
            "Missing required input: 'language'"
        )

    if not isinstance(reload, bool):
        raise ValueError(
            "Input 'reload_anything' must be a boolean"
        )

    if not isinstance(del_cache, bool):
        raise ValueError(
            "Input 'del_cache' must be a boolean"
        )
    
    
    if reload:
        pull_git()
    ## deleted at the end is better
    # if del_cache:
    #     print("cache is erased")
    #     run_command("rm -rf /runpod-volume/torchinductor-cache/*")
    #     cache_diagnostics()


    return {}


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

if not TESTS_IN_DOCKER:
    if __name__ == "__main__":

        runpod.serverless.start(
            {
                "handler": handler
            }
        )



print("TEST BOOTSTRAP")
bootstrap()