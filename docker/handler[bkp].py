import subprocess
import time
import shutil
import os
import runpod
from pathlib import Path
import sys
from pathlib import Path
import torch
import sys
import base64
import shutil




import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

# test pour voir si le filtrage fonctionne sous serverless et apparement c'est pas le cas ..
logger = logging.getLogger("musetalk")
logger.setLevel(logging.DEBUG)

logger.debug("XXXXXXXX DEBUG TEST")
logger.info("XXXXXXXX INFO TEST")
logger.warning("XXXXXXXX WARNING TEST")
logger.error("XXXXXXXX ERROR TEST")




# ==========================================================
# COPY AVATAR FROM VOLUME RUNPOD TO WORKSPACE
## FINALEMENT ON NE LE FAIT PAS CAR ON VA AJOUTER LE CACHE.PT DIRECTEMENT DANS L'IMAGE ... MAIS ATENTION CECI PEUT AUSSI RALENTIR LE
## TEMPS DE CHARGEMENT DU DOCKER ... A VOIR ...
## ET CA NOUS EVITE AUSSI LE FAIT DE FAIRE UN CODE QUI CREE ET STOCKE LE AVATAR.PT UNE FOIS EN CACHE 
# ==========================================================

# VOLUME_DIR = "/runpod-volume"

# AVATAR_DIR = (
#     "/workspace/MuseTalk/results/v15/avatars/avatar_1"
# )

# os.makedirs(AVATAR_DIR, exist_ok=True)

# for filename in [
#     "avatar_cache.pt",
#     "avatar_info.json",
# ]:
#     src = os.path.join(VOLUME_DIR, filename)
#     dst = os.path.join(AVATAR_DIR, filename)

#     if not os.path.exists(src):
#         raise FileNotFoundError(
#             f"Missing avatar file on RunPod Volume: {src}"
#         )

#     print(f"Copying {src} -> {dst}")

#     shutil.copy2(src, dst)

#     print(
#         f"Copied {filename}: "
#         f"{os.path.getsize(dst) / (1024**2):.2f} MB"
#     )

# print("Avatar files copied successfully.")

# ==========================================================
# VARIABLES
# ==========================================================

# to test in docker before serverless as is faster : cd /workspace/MuseTalk/docker && python3 handler.py
# pip install runpod (sera inclue dans la version docker final )
# git clone --branch serverless https://github.com/seb2oo/MuseTalk.git /workspace/MuseTalk

# if False means : serverless
# if True means : Docker (for internal test before serverless deployment)
TESTS_IN_DOCKER = False

if TESTS_IN_DOCKER:
    USE_CACHE_MODEL = False
else:
    USE_CACHE_MODEL = True


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

logger.debug("XXX 1 - handler.py started")
logger.debug("XXX 1 - handler.py started")
logger.debug("XXX 1 - handler.py started")

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

## CETTE COMMANDE CREEAIT DES SOUCIS AVEC LA COMMANDE huggingface-cli ! CERTAINEMENT DU PYTHON PATH...
# def run_command(command):
#     print(f"[COMMAND] {command}")

#     env = os.environ.copy()
#     env["PYTHONPATH"] = str(PROJECT_DIR) + ":" + env.get("PYTHONPATH", "")

#     subprocess.run(
#         command,
#         shell=True,
#         check=True,
#         env=env,
#     )

def run_command(command):
    print(f"[COMMAND] {command}")

    subprocess.run(
        command,
        check=True,
    )

def run_command2(command):
    print(f"[COMMAND] {command}")

    subprocess.run(
        command,
        shell=True,
        check=True,
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

print("XXX 2 - before HF cache lookup")
logger.debug("XXX 2")

if USE_CACHE_MODEL :
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

            run_command2(
                f"rm -rf '{PROJECT_DIR}'"
            )

        print("Cloning MuseTalk repository...")

        run_command2(
            "git clone -b serverless "
            "https://github.com/seb2oo/MuseTalk.git "
            f"'{PROJECT_DIR}'"
        )
        print("1")
        logger.debug("XXX 1 - handler.py started")
        logger.debug("XXX 1 - handler.py started")

    else:

        print(
            "MuseTalk repository already exists."
        )

    

    print("2")
    # Make MuseTalk available to this Python process
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    # --------------------------------------------------------
    # DOWNLOAD CHECKPOINT
    # --------------------------------------------------------

    if USE_CACHE_MODEL:
  
        DESTINATION = MODELS_DIR

        print(f"Copying MuseTalk model to: {DESTINATION}")

        # can take time.. to monitor and in other case we give the CHECKPOINT_PATH direytly to the code if possible ie : unet_model_path = Path(CHECKPOINT_PATH) / "musetalkV15" / "unet.pth"
        ## et effectivement c'est possible si on regarde le code plus bas  : unet_model_path="./models/musetalkV15/musetalk/pytorch_model.bin",
        ## et jsutement on ne va pas faire le shuttil car prend trop de temsp 
        # shutil.copytree(
        #     MUSE_TALK_CACHE_PATH,
        #     DESTINATION,
        #     dirs_exist_ok=True
        # )

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

        print("UNET CONFIG:", unet_config)
        print("UNET CONFIG EXISTS:", os.path.exists(unet_config))

        print("UNET MODEL:", unet_model_path)
        print("UNET MODEL EXISTS:", os.path.exists(unet_model_path))

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

print("XXX 3 - before bootstrap()")
bootstrap()

# ==========================================================
# ADDING CACHE
# ==========================================================


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

    shutil.copy2(src, dst)

    print(
        f"Copied {filename}: "
        f"{os.path.getsize(dst) / (1024**2):.2f} MB"
    )




# ==========================================================
# S'ASSURER QUE LE FICHIER REALTIME EST BIEN EN PREPARATION FALSE ETANT DONNE QUE LAVATAR.PT EST DEJA DANS L'IMAGE
# ==========================================================

yaml_path = "/workspace/MuseTalk/configs/inference/realtime.yaml"

with open(yaml_path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "preparation: True",
    "preparation: False"
)

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(content)

print("realtime.yaml: preparation set to False")


"""
handler.py
    │
    ├── import realtime_inference
    │
    ├── initialize_musetalk()       ← UNE FOIS
    │       │
    │       ├── load UNet
    │       ├── load VAE
    │       ├── load PE
    │       ├── load Whisper
    │       ├── FaceParsing
    │       └── Avatar/cache
    │
    └── handler(job)
            │
            └── avatar.inference(...)
"""


## import GENERAL, DONC EXECUTION DE CE FICHIER SANS FAIRE LE IF NAME = MAIN .. ETC ...
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_DIR)
from scripts import realtime_inference

from scripts.realtime_inference import fast_check_ffmpeg
from scripts.realtime_inference import log_time
from scripts.realtime_inference import load_all_model
from scripts.realtime_inference import AudioProcessor
from scripts.realtime_inference import WhisperModel
from scripts.realtime_inference import FaceParsing
from scripts.realtime_inference import OmegaConf
from scripts.realtime_inference import Avatar


# ============================================================
# INITIALISATION DU WORKER
# ============================================================

# Ici tu reprends la partie "setup" de realtime_inference.py
# qui était dans if __name__ == "__main__"

# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.T0 = time.perf_counter()

import argparse

# os.chdir a fait que ce chemins sont valides !
args = argparse.Namespace(
    version="v15",
    ffmpeg_path="./ffmpeg-4.4-amd64-static/",
    gpu_id=0,
    vae_type="sd-vae",
    # unet_config="./models/musetalkV15/musetalk/musetalk.json",
    # unet_model_path="./models/musetalkV15/musetalk/pytorch_model.bin",
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
# renvoyer les args a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.args = args


# Configure ffmpeg path
if not fast_check_ffmpeg():
    print("Adding ffmpeg to PATH")
    # Choose path separator based on operating system
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# Set computing device
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.device = device
torch.backends.cudnn.benchmark = True
log_time("XXXYYY before load_all_model")# OK
# Load model weights
vae, unet, pe = load_all_model(
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    device=device
)
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.pe = pe
timesteps = torch.tensor([0], device=device)
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.timesteps = timesteps
log_time("XXXYYY after load_all_model")# NOK here to improve

pe = pe.half().to(device)
# pe = pe.half().to(device).eval()
vae.vae = vae.vae.half().to(device)
# vae.vae = vae.vae.to(memory_format=torch.channels_last)
unet.model = unet.model.half().to(device)
# unet.model = unet.model.half().to(device).eval()

# vae.vae = torch.compile(
# vae.vae,
# mode="reduce-overhead"
# )


# Initialize audio processor and Whisper model
audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.audio_processor = audio_processor
weight_dtype = unet.model.dtype
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.weight_dtype = weight_dtype
whisper = WhisperModel.from_pretrained(args.whisper_dir)
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.whisper = whisper
log_time("XXXYYY after whisper load")# OK

# Initialize face parser with configurable parameters based on version
if args.version == "v15":
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
else:  # v1
    fp = FaceParsing()
    
# renvoyer les variable global a realtime inférence !! HYPER IMPORTANT --> expliqué à la fin de ce fichier 
realtime_inference.fp = fp

inference_config = OmegaConf.load(args.inference_config)
print(inference_config)
log_time("XXXYYY after interference config")# OK


## TOUTE CETTE PARTIE ICI VA ÊTRE GERER DANS LE HANDLER MNT (OU EN TOUT CAS UNE PARTIE)
# for avatar_id in inference_config:
#     data_preparation = inference_config[avatar_id]["preparation"]
#     log_time("XXXYYY after data preparation") # OK
#     video_path = inference_config[avatar_id]["video_path"]
#     log_time("XXXYYY after video_path")# OK
#     if args.version == "v15":
#         bbox_shift = 0
#     else:
#         bbox_shift = inference_config[avatar_id]["bbox_shift"]
#     avatar = Avatar(
#         avatar_id=avatar_id,
#         video_path=video_path,
#         bbox_shift=bbox_shift,
#         batch_size=args.batch_size,
#         preparation=data_preparation)
#     log_time("XXXYYY after Avatar, reading image ?") # NOK, has reading images two times ...

#     audio_clips = inference_config[avatar_id]["audio_clips"]
#     for audio_num, audio_path in audio_clips.items():
#         print("Inferring using:", audio_path)
#         avatar.inference(audio_path,
#                         audio_num,
#                         args.fps,
#                         args.skip_save_images)
#     log_time("XXXYYY after avatar inference")


## ATTENTION SI :D:\Dev\04_MuseTalkImprovement\MuseTalk\configs\inference\realtime.yaml est a False ou True alors soit ca passe, soit ca casse
## etant donne que ca ne s'execute que une fois il faut que realtime.yaml soit a False Mais que l'avatar.pt existe deja
## donc placer le avatar.pt deja dnas le bon dossier , basé sur realtime.yaml, il faudrait le placé là :
## MuseTalk/results/v15/avatars/avatar_1/avatar_cache.pt et d'après ce que je vois dans realtime_inference.py il faudrait aussi placer le fichier .json

## GROS PROBLEME GIT NE SUPPORTE QUE 1OOMB DE FILE SIZE ! LE.PT FAIT PRESQUE 1GB ! IL VA DONC FALLOIR UTILISE RUNPOD VOLUME


for avatar_id in inference_config:
    data_preparation = inference_config[avatar_id]["preparation"]
    log_time("XXXYYY after data preparation")

    video_path = inference_config[avatar_id]["video_path"]
    log_time("XXXYYY after video_path")

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

    log_time("XXXYYY after Avatar, reading image ?")


print("I SHOULD BE HAPPY")

## handler final
def handler(job):
    """
    Reçoit un job RunPod contenant le chemin de l'audio.
    Tous les modèles et l'Avatar sont déjà chargés en mémoire.
    """

    job_input = job.get("input", {})

    # --------------------------------------------------------
    # Commande shell de test
    # --------------------------------------------------------
    any_shell_command = job_input.get("any_shell_command","")
    if any_shell_command:
        print(f"[HANDLER] Running shell command: {any_shell_command}")

        run_command2(any_shell_command)

        # On sort immédiatement du handler
        return {
            "status": "command_completed",
            "command": any_shell_command,
        }

    audio_path = job_input.get("audio_path")
    if not audio_path:
        return {
            "error": "Missing 'audio_path' in job input"
        }

    print(f"[HANDLER] Inferring using: {audio_path}")

    # On utilise l'avatar déjà initialisé au démarrage
    audio_num = job_input.get("audio_num", "audio_1")


    avatar.inference(
        audio_path,
        audio_num,
        args.fps,
        args.skip_save_images
    )

    log_time("XXXYYY after avatar inference")

    # "avatar_1 et audio num vienne du fichier realtime.yaml... rendre ca pt plus  solide par la suite serait une bonne idée"
    output_path = f"/workspace/MuseTalk/results/v15/avatars/avatar_1/vid_output/{audio_num}.mp4"


    if not os.path.exists(output_path):
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


# {
#   "input": {
#     "audio_path": "/workspace/MuseTalk/data/audio/seb_audio.wav",
#     "audio_num": "audio_1",
#     "any_shell_command": "ls -lah /workspace/MuseTalk"
#   }
# }

# {
#   "input": {
#     "audio_path": "/workspace/MuseTalk/data/audio/seb_audio.wav",
#     "audio_num": "audio_1",
#     "any_shell_command": ""
#   }
# }





"""
5. Pourquoi realtime_inference.args = args ?

C'est probablement le point le plus important.

Tu pourrais avoir dans handler.py :

args = argparse.Namespace(...)

mais ça crée :

handler.py
    └── args

Alors que ton Avatar est défini dans :

realtime_inference.py
    └── Avatar

Et lorsque Avatar cherche :

args.version

il cherche dans le namespace global de realtime_inference.py.

Donc on lui donne explicitement :

realtime_inference.args = args

Ce qui donne :

handler.py
    │
    └── args
          │
          ▼
realtime_inference.args
          │
          ▼
       Avatar
          │
          ├── args.version
          ├── args.batch_size
          ├── args.result_dir
          └── ...
6. Même chose pour device, vae, unet, etc.

C'est exactement le même principe.

Dans ton realtime_inference.py, Avatar.inference() utilise par exemple :

device
vae
unet
whisper
audio_processor
pe
timesteps
weight_dtype

Ces variables sont normalement créées dans :

if __name__ == "__main__":

Mais maintenant ce bloc ne sera plus exécuté.

Donc dans handler.py, après les avoir créées, tu dois les mettre dans le module :

realtime_inference.device = device
realtime_inference.vae = vae
realtime_inference.unet = unet
realtime_inference.whisper = whisper
realtime_inference.audio_processor = audio_processor
realtime_inference.pe = pe
realtime_inference.timesteps = timesteps
realtime_inference.weight_dtype = weight_dtype
"""



"""
Pour ton projet

Si tu utilises beaucoup de choses de realtime_inference.py, fais :

import realtime_inference

et utilise :

realtime_inference.Avatar
realtime_inference.video2imgs
realtime_inference.fast_check_ffmpeg

Si tu n'as besoin que d'une fonction :

from realtime_inference import video2imgs

et ensuite :

video2imgs(...)

Dans ton cas, je garderais import realtime_inference, parce que ton handler va avoir besoin de Avatar + plusieurs variables globales du module. 
Ça évite de multiplier les imports et rend l'organisation plus claire.
"""