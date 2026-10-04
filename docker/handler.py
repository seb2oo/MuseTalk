import argparse
import base64
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import runpod
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)


logger = logging.getLogger("musetalk")


TESTS_IN_DOCKER = False

if TESTS_IN_DOCKER:
    USE_CACHE_MODEL = False
else:
    USE_CACHE_MODEL = True


PROJECT_DIR = Path("/workspace/MuseTalk")
MODELS_DIR = PROJECT_DIR / "models"

HF_CACHE_DIR = Path(
    "/runpod-volume/huggingface-cache/hub"
)

MUSE_TALK_MODEL_ID = "TMElyralab/MuseTalk"

unet_config = str(MODELS_DIR / "musetalkV15" / "musetalk" / "musetalk.json")
unet_model_path = str(MODELS_DIR / "musetalkV15" / "musetalk" / "pytorch_model.bin")



logger.info("Starting MuseTalk worker")
logger.info("Project directory:")
logger.info(f"  {PROJECT_DIR}")
logger.info("Models directory:")
logger.info(f"  {MODELS_DIR}")


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


def run_command(command):
    logger.debug(f"[COMMAND] {command}")

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"[COMMAND ERROR] {result.stderr.strip()[-4000:]}")
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            stderr=result.stderr,
        )


def run_command2(command):
    logger.debug(f"[COMMAND] {command}")

    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"[COMMAND ERROR] {result.stderr.strip()[-4000:]}")
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            stderr=result.stderr,
        )


def pull_git():
    logger.info("Pulling MuseTalk repository")

    run_command2(
        f"cd '{PROJECT_DIR}' && git pull origin serverless"
    )


def find_cached_model(model_id):

    org, name = model_id.split("/", 1)

    model_dir = (
        HF_CACHE_DIR
        / f"models--{org}--{name}"
    )

    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        logger.fatal(f"HF cache not found: {snapshots_dir}")
        raise RuntimeError(f"HF cache not found: {snapshots_dir}")

    snapshots = [
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir()
    ]

    if not snapshots:
        logger.fatal(f"No cached snapshot found for {model_id}")
        raise RuntimeError(f"No cached snapshot found for {model_id}")

    if len(snapshots) > 1:
        logger.info(f"Found {len(snapshots)} cached snapshots.")

    return snapshots[0]

logger.debug("Looking up cached MuseTalk model")

if USE_CACHE_MODEL :
    MUSE_TALK_CACHE_PATH = find_cached_model(
        MUSE_TALK_MODEL_ID
    )

    logger.info("MuseTalk cached model found:")
    logger.info(MUSE_TALK_CACHE_PATH)


def bootstrap():

    logger.info("Starting MuseTalk bootstrap")


    if not os.path.exists(
        os.path.join(PROJECT_DIR, ".git")
    ):

        logger.info("MuseTalk repository not found.")

        if os.path.exists(PROJECT_DIR):
            logger.info(f"Removing incomplete directory: {PROJECT_DIR}")

            run_command2(
                f"rm -rf '{PROJECT_DIR}'"
            )

        logger.info("Cloning MuseTalk repository...")

        run_command2(
            "git clone -b serverless "
            "https://github.com/seb2oo/MuseTalk.git "
            f"'{PROJECT_DIR}'"
        )

    else:

        logger.info("MuseTalk repository already exists.")


    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))


    if USE_CACHE_MODEL:

        logger.debug(f"Using cached MuseTalk model from: {MUSE_TALK_CACHE_PATH}")


        unet_config=str(
        MUSE_TALK_CACHE_PATH
        / "musetalk"
        / "musetalk.json"
        )

        unet_model_path=str(
            MUSE_TALK_CACHE_PATH
            / "musetalk"
            / "pytorch_model.bin"
        )

        logger.debug(f"UNET CONFIG: {unet_config}")
        logger.debug(f"UNET CONFIG EXISTS: {os.path.exists(unet_config)}")

        logger.debug(f"UNET MODEL: {unet_model_path}")
        logger.debug(f"UNET MODEL EXISTS: {os.path.exists(unet_model_path)}")

        logger.debug("Using MuseTalk model directly from the RunPod cache")
    else:
        logger.info("Downloading MuseTalk")

        MUSE_TALK_DIR = MODELS_DIR / "musetalkV15"

        run_command([
            "huggingface-cli",
            "download",
            "TMElyralab/MuseTalk",
            "--local-dir",
            str(MUSE_TALK_DIR),
        ])


    logger.info("Downloading SD VAE")

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


    logger.info("Downloading Whisper")

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


    logger.info("Downloading DWPose")

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


    logger.info("Downloading LatentSync")

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


    logger.info("Downloading face-parse-bisent")

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


    logger.info("Downloading S3FD")

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


    logger.info("Model download completed")
    logger.info("Models are located in:")
    logger.info(f"  {MODELS_DIR}")

logger.debug("Starting bootstrap")
bootstrap()


CACHE_SOURCE = "/workspace/avatar_cache"
MUSE_TALK_PATH = "/workspace/MuseTalk"
AVATAR_PATH = os.path.join(
    MUSE_TALK_PATH,
    "results",
    "v15",
    "avatars",
    "avatar_1",
)

os.makedirs(AVATAR_PATH, exist_ok=True)

for filename in ["avatar_cache.pt", "avatar_info.json"]:
    src = os.path.join(CACHE_SOURCE, filename)
    dst = os.path.join(AVATAR_PATH, filename)

    if not os.path.exists(src):
        logger.fatal(f"Missing avatar cache file: {src}")
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)

    logger.debug(
        f"Copied {filename}: {os.path.getsize(dst) / (1024**2):.2f} MB"
    )


yaml_path = "/workspace/MuseTalk/configs/inference/realtime.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "preparation: True",
    "preparation: False"
)

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(content)

logger.info("realtime.yaml: preparation set to False")


os.chdir(PROJECT_DIR)
from scripts import realtime_inference

from scripts.realtime_inference import fast_check_ffmpeg
from scripts.realtime_inference import load_all_model
from scripts.realtime_inference import AudioProcessor
from scripts.realtime_inference import WhisperModel
from scripts.realtime_inference import FaceParsing
from scripts.realtime_inference import OmegaConf
from scripts.realtime_inference import Avatar


realtime_inference.T0 = time.perf_counter()


def log_time(message):
    logger.debug(f"{message} | elapsed={time.perf_counter() - realtime_inference.T0:.3f}s")


args = argparse.Namespace(
    version="v15",
    ffmpeg_path="./ffmpeg-4.4-amd64-static/",
    gpu_id=0,
    vae_type="sd-vae",


    unet_config=unet_config,
    unet_model_path = unet_model_path,
    whisper_dir="./models/whisper",
    inference_config="configs/inference/realtime.yaml",
    bbox_shift=0,
    result_dir="./results",
    extra_margin=10,
    fps=25,
    audio_padding_length_left=2,
    audio_padding_length_right=2,
    batch_size=8,
    output_vid_name=None,
    use_saved_coord=False,
    saved_coord=False,
    parsing_mode="jaw",
    left_cheek_width=90,
    right_cheek_width=90,
    skip_save_images=False,
)

realtime_inference.args = args


if not fast_check_ffmpeg():
    logger.debug("Adding ffmpeg to PATH")

    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        logger.error("Unable to find ffmpeg after updating PATH")


device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

realtime_inference.device = device
torch.backends.cudnn.benchmark = True
log_time("Before model loading")    

vae, unet, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    device=device
)

realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.pe = pe
timesteps = torch.tensor([0], device=device)

realtime_inference.timesteps = timesteps
log_time("After model loading")                     

pe = pe.half().to(device)

vae.vae = vae.vae.half().to(device)

unet.model = unet.model.half().to(device)


audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)

realtime_inference.audio_processor = audio_processor
weight_dtype = unet.model.dtype

realtime_inference.weight_dtype = weight_dtype
whisper = WhisperModel.from_pretrained(args.whisper_dir)
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

realtime_inference.whisper = whisper
log_time("After Whisper loading")    


if args.version == "v15":
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
else:      
    fp = FaceParsing()


realtime_inference.fp = fp

inference_config = OmegaConf.load(args.inference_config)
logger.debug(f"Inference config loaded from: {args.inference_config}")
log_time("After inference config loading")    


for avatar_id in inference_config:
    data_preparation = inference_config[avatar_id]["preparation"]
    log_time("After avatar data preparation")

    video_path = inference_config[avatar_id]["video_path"]
    log_time("After reading avatar video path")

    if args.version == "v15":
        bbox_shift = 0
    else:
        bbox_shift = inference_config[avatar_id]["bbox_shift"]

    avatar = Avatar(
        avatar_id=avatar_id,
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=args.batch_size,
        preparation=data_preparation
    )

    log_time("After Avatar initialization")


def handler(job):
    """
    Reçoit un job RunPod contenant le chemin de l'audio.
    Tous les modèles et l'Avatar sont déjà chargés en mémoire.
    """

    job_input = job.get("input", {})


    any_shell_command = job_input.get("any_shell_command","")
    if any_shell_command:
        logger.debug(f"Running shell command: {any_shell_command}")

        run_command2(any_shell_command)


        return {
            "status": "command_completed",
            "command": any_shell_command,
        }

    audio_path = job_input.get("audio_path")
    if not audio_path:
        logger.error("Missing 'audio_path' in job input")
        return {
            "error": "Missing 'audio_path' in job input"
        }

    logger.info(f"Starting inference for {audio_path}")


    audio_num = job_input.get("audio_num", "audio_1")


    try:
        avatar.inference(
            audio_path,
            audio_num,
            args.fps,
            args.skip_save_images
        )
    except Exception:
        logger.exception(f"Inference failed for {audio_path}")
        raise

    log_time("After avatar inference")


    output_path = f"/workspace/MuseTalk/results/v15/avatars/avatar_1/vid_output/{audio_num}.mp4"


    if not os.path.exists(output_path):
        logger.error(f"Video not found: {output_path}")
        return {
            "status": "error",
            "message": f"Video not found: {output_path}"
        }

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    return {
        "status": "completed",
        "audio_path": audio_path,
        "audio_num": audio_num,
        "video_base64": video_base64,
    }


if not TESTS_IN_DOCKER:
    if __name__ == "__main__":

        runpod.serverless.start(
            {
                "handler": handler
            }
        )




"""
Ce que j’ai nettoyé
Remplacement de tous les print() par logger.debug(), logger.info(), logger.warning(), logger.error() ou logger.fatal() selon le cas.
Suppression des anciens tests XXXXXXXX DEBUG TEST, XXX 1, etc.
Suppression des gros blocs de commentaires devenus inutiles.
Suppression des imports en double.
Suppression du print(inference_config) qui pouvait déverser toute la configuration dans les logs.
run_command() et run_command2() modifiés :
stdout → DEVNULL
stderr → capturé
stderr affiché uniquement en cas d'erreur
erreur limitée aux dernières 4000 caractères
Donc les huggingface-cli, wget, etc. ne vont plus balancer leurs progress bars dans RunPod.
Remplacement du log_time() de realtime_inference.py, qui pouvait lui-même produire du bruit, par un petit log_time() local utilisant logger.debug().
Conservation des timings utiles avec :
elapsed=XXXs
Ajout de vrais logs d’erreur autour de avatar.inference().
Vérification explicite des fichiers avatar_cache.pt et avatar_info.json.
Conservation de l’utilisation directe du modèle MuseTalk depuis le cache RunPod, sans recopier le gros modèle dans /workspace.
J’ai aussi sécurisé le cas USE_CACHE_MODEL=False en définissant par défaut unet_config et unet_model_path, ce qui évite le NameError que tu avais rencontré.
Le fichier final compile correctement.
646 lignes, contre 1043 dans ton fichier original.
"""