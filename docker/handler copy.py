import os
import subprocess
import time
import shutil

# t0 = time.perf_counter()
# ## debug test to see if writting in local app is faster ! this of de docker file will be overwrite : ENV TORCHINDUCTOR_CACHE_DIR=/runpod-volume/torchinductor-cache
# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/app/torchinductor-cache"
# os.makedirs(
#     "/app/torchinductor-cache",
#     exist_ok=True,
# )

# ## sur serveur standard prend 180 secondesss , probably because cache is made of tousand of small files....
# subprocess.run(
#     [
#         "cp",
#         "-a",
#         "/runpod-volume/torchinductor-cache/.",
#         "/app/torchinductor-cache/",
#     ],
#     check=True,
# )
# print(
#     f"Time to transfert cache locally "
#     f"{time.perf_counter() - t0:.2f}s"
# )





## create a fake molotique file same size of the actual cache and see if i take long time also 
## OK ON A DONC CONFIRMER QUE LE RPOBLEME VIENT DES MILLIER DE FICHIER
# fake_file = "/app/fake_180mb.bin"
# size_mb = 180

# t0 = time.perf_counter()

# with open(fake_file, "wb") as f:
#     remaining = size_mb * 1024 * 1024
#     chunk = b"\0" * (1024 * 1024)

#     while remaining > 0:
#         n = min(remaining, len(chunk))
#         f.write(chunk[:n])
#         remaining -= n

#     f.flush()
#     os.fsync(f.fileno())

# print(
#     f"Created {fake_file} ({size_mb} MB) in "
#     f"{time.perf_counter() - t0:.2f}s"
# )


# # dummy test monolitique file
# t0 = time.perf_counter()
# subprocess.run(
#     [
#         "cp",
#         "/app/fake_180mb.bin",
#         "/runpod-volume/fake_180mb.bin",
#     ],
#     check=True,
# )

# print(
#     f"FAKE FILE: local -> volume: "
#     f"{time.perf_counter() - t0:.2f}s"
# )


# t0 = time.perf_counter()
# subprocess.run(
#     [
#         "cp",
#         "/runpod-volume/fake_180mb.bin",
#         "/app/fake_180mb_back.bin",
#     ],
#     check=True,
# )

# print(
#     f"FAKE FILE: volume -> local: "
#     f"{time.perf_counter() - t0:.2f}s"
# )



## hidden problem with that ...
# CACHE_DIR = "/app/torchinductor-cache"
# CACHE_TAR = "/runpod-volume/torchinductor-cache.tar"

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

# # Nettoyer complètement le cache local
# if os.path.exists(CACHE_DIR):
#     shutil.rmtree(CACHE_DIR)
# os.makedirs(CACHE_DIR, exist_ok=True)


# if os.path.exists(CACHE_TAR):
#     t0 = time.perf_counter()

#     subprocess.run(
#         [
#             "tar",
#             "-xf",
#             CACHE_TAR,
#             "-C",
#             CACHE_DIR,
#         ],
#         check=True,
#     )

#     print(
#         f"Time to extract TorchInductor cache: "
#         f"{time.perf_counter() - t0:.2f}s"
#     )

# ============================================================
# TORCHINDUCTOR CACHE RESTORE
# ============================================================

CACHE_DIR = "/app/torchinductor-cache"
CACHE_TAR = "/runpod-volume/torchinductor-cache.tar"

print("=== CACHE RESTORE START ===", flush=True)

print(f"CACHE_DIR  = {CACHE_DIR}", flush=True)
print(f"CACHE_TAR  = {CACHE_TAR}", flush=True)

# Important: before importing torch
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR

print("Checking archive...", flush=True)

if os.path.exists(CACHE_TAR):

    size = os.path.getsize(CACHE_TAR)

    print(
        f"Archive exists: YES ({size / 1024 / 1024:.2f} MB)",
        flush=True,
    )

    # --------------------------------------------------------
    # TEST 1: Can tar read the archive?
    # --------------------------------------------------------

    print("TEST 1: running tar -tf ...", flush=True)

    t0 = time.perf_counter()

    result = subprocess.run(
        [
            "tar",
            "-tf",
            CACHE_TAR,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.perf_counter() - t0

    print(
        f"tar -tf finished in {elapsed:.2f}s",
        flush=True,
    )

    print(
        f"tar -tf return code: {result.returncode}",
        flush=True,
    )

    if result.stderr:
        print(
            f"tar -tf stderr:\n{result.stderr}",
            flush=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "TorchInductor cache archive is invalid."
        )

    print("TEST 1 PASSED", flush=True)

    # --------------------------------------------------------
    # TEST 2: Extract archive
    # --------------------------------------------------------

    print("Cleaning local cache...", flush=True)

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Local cache ready.", flush=True)

    print("TEST 2: extracting archive...", flush=True)

    t0 = time.perf_counter()

    result = subprocess.run(
        [
            "tar",
            "-xf",
            CACHE_TAR,
            "-C",
            CACHE_DIR,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.perf_counter() - t0

    print(
        f"tar extraction finished in {elapsed:.2f}s",
        flush=True,
    )

    print(
        f"tar extraction return code: {result.returncode}",
        flush=True,
    )

    if result.stderr:
        print(
            f"tar extraction stderr:\n{result.stderr}",
            flush=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            "TorchInductor cache extraction failed."
        )

    print("TEST 2 PASSED", flush=True)

else:

    print(
        "Archive does NOT exist. Starting with empty local cache.",
        flush=True,
    )

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    os.makedirs(CACHE_DIR, exist_ok=True)

print("=== CACHE RESTORE END ===", flush=True)





import base64
import io

import numpy as np
import soundfile as sf
os.makedirs("/runpod-volume/torchinductor-cache", exist_ok=True)




import torch
import runpod

import sys
import contextlib

from pathlib import Path
import logging

logging.getLogger("torch._functorch._aot_autograd.autograd_cache").setLevel(logging.INFO)

# import time # not needed, log serveless do it


# ============================================================
# CONFIG
# ============================================================

INIT_CACHE = True

REPO_PATH = "/app/fish-speech"

CHECKPOINT_PATH = "/app/checkpoints/s2-pro"


REFERENCE_WAV_FR = "/app/fish-speech/input/fr.wav"
REFERENCE_TOKENS_FR = "/app/fish-speech/output/reconstructed_fr.npy"

REFERENCE_WAV_EN = "/app/fish-speech/input/en.wav"
REFERENCE_TOKENS_EN = "/app/fish-speech/output/reconstructed_en.npy"

REFERENCE_WAV_ES = "/app/fish-speech/input/es.wav"
REFERENCE_TOKENS_ES = "/app/fish-speech/output/reconstructed_es.npy"

REFERENCE_WAV_DE = "/app/fish-speech/input/de.wav"
REFERENCE_TOKENS_DE = "/app/fish-speech/output/reconstructed_de.npy"

REFERENCE_TOKENS = [REFERENCE_TOKENS_FR,REFERENCE_TOKENS_EN,REFERENCE_TOKENS_ES,REFERENCE_TOKENS_DE]
REFERENCE_WAV= [REFERENCE_WAV_FR,REFERENCE_WAV_EN,REFERENCE_WAV_ES,REFERENCE_WAV_DE]

DEVICE = "cuda"
PRECISION = torch.bfloat16

COMPILE = True


PROMPT_TEXT_FR = (
    "Le rire d'un proche a le pouvoir d'effacer mes soucis. "
    "Il éclate comme une lumière claire et me remplit de joie. "
    "Dans ces instants, tout semble plus léger, "
    "et je retrouve confiance en l'avenir."
)

PROMPT_TEXT_EN = (
    "This system converts written text into natural-sounding speech. "
    "Each word is processed, analyzed for context, and generated with the correct intonation. "
    "The goal is to make digital voices sound as close to human conversation as possible."
)

PROMPT_TEXT_ES = (
    "Camino por la playa temprano en la mañana. "
    "El sonido de las olas acompaña mis pasos y las gaviotas vuelan sobre mi cabeza. "
    "A veces me detengo a recoger una concha brillante. "
    "Estos instantes me llenan de calma antes de que empiece el día."
)


PROMPT_TEXT_DE = (
    "Dieses System wandelt geschriebenen Text in natürlich klingende Sprache um. "
    "Jede Eingabe wird analysiert, der Kontext berücksichtigt und die richtige Betonung erzeugt. "
    "Ziel ist es, digitale Stimmen so menschlich wie möglich klingen zu lassen."
)


TEMPERATURE = 0.7
TOP_P = 0.7
TOP_K = 30
REPETITION_PENALTY = 1.1

CHUNK_LENGTH = 300


# --------------------------------------------------------
# TORCHINDUCTOR CACHE DIAGNOSTIC
# --------------------------------------------------------
def cache_diagnostics():

    cache_dir = os.environ.get(
        "TORCHINDUCTOR_CACHE_DIR",
        "/runpod-volume/torchinductor-cache",
    )

    print("=" * 70)
    print("TORCHINDUCTOR CACHE DIAGNOSTICS")
    print("=" * 70)

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    print()
    print("ENVIRONMENT")
    print("-" * 70)

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA version (PyTorch): "
        f"{torch.version.cuda}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"Compute capability: "
            f"{torch.cuda.get_device_capability(0)}"
        )

        print(
            f"GPU count: "
            f"{torch.cuda.device_count()}"
        )

    # ========================================================
    # CACHE ENVIRONMENT VARIABLES
    # ========================================================

    print()
    print("CACHE ENVIRONMENT VARIABLES")
    print("-" * 70)

    print(
        f"TORCHINDUCTOR_CACHE_DIR: "
        f"{os.environ.get('TORCHINDUCTOR_CACHE_DIR', '<not set>')}"
    )

    print(
        f"TRITON_CACHE_DIR: "
        f"{os.environ.get('TRITON_CACHE_DIR', '<not set>')}"
    )

    print(
        f"TORCH_HOME: "
        f"{os.environ.get('TORCH_HOME', '<not set>')}"
    )

    # ========================================================
    # CACHE DIRECTORY
    # ========================================================

    print()
    print("CACHE DIRECTORY")
    print("-" * 70)

    print(
        f"Cache directory: "
        f"{cache_dir}"
    )

    if not os.path.exists(cache_dir):

        print(
            "Cache exists: NO"
        )

        print("=" * 70)

        return

    # ========================================================
    # CACHE STATISTICS
    # ========================================================

    total_size = 0
    file_count = 0
    directory_count = 0

    extensions = {}

    for root, dirs, files in os.walk(cache_dir):

        directory_count += len(dirs)

        for filename in files:

            filepath = os.path.join(
                root,
                filename,
            )

            try:

                size = os.path.getsize(
                    filepath
                )

                total_size += size
                file_count += 1

                extension = (
                    Path(filename).suffix
                    or "<no extension>"
                )

                extensions[extension] = (
                    extensions.get(extension, 0)
                    + 1
                )

            except OSError:

                pass

    print(
        "Cache exists: YES"
    )

    print(
        f"Cache files: "
        f"{file_count}"
    )

    print(
        f"Cache directories: "
        f"{directory_count}"
    )

    print(
        f"Cache size: "
        f"{total_size / (1024 * 1024):.2f} MB"
    )

    # ========================================================
    # FILE TYPES
    # ========================================================

    if extensions:

        print()
        print("FILE TYPES")
        print("-" * 70)

        for extension, count in sorted(
            extensions.items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            print(
                f"{extension}: "
                f"{count}"
            )

    # ========================================================
    # TOP-LEVEL CONTENT
    # ========================================================

    # print()
    # print("TOP-LEVEL CONTENT")
    # print("-" * 70)

    # try:

    #     entries = sorted(
    #         os.listdir(cache_dir)
    #     )

    #     if not entries:

    #         print(
    #             "(empty)"
    #         )

    #     else:

    #         for entry in entries:

    #             path = os.path.join(
    #                 cache_dir,
    #                 entry,
    #             )

    #             if os.path.isdir(path):

    #                 print(
    #                     f"[DIR]  {entry}"
    #                 )

    #             else:

    #                 try:

    #                     size = (
    #                         os.path.getsize(path)
    #                         / (1024 * 1024)
    #                     )

    #                     print(
    #                         f"[FILE] {entry} "
    #                         f"({size:.2f} MB)"
    #                     )

    #                 except OSError:

    #                     print(
    #                         f"[FILE] {entry}"
    #                     )

    # except OSError as e:

    #     print(
    #         f"Unable to list cache: {e}"
    #     )

    # ========================================================
    # RECENT CACHE FILES
    # ========================================================

    print()
    print("RECENT CACHE FILES")
    print("-" * 70)

    recent_files = []

    for root, dirs, files in os.walk(cache_dir):

        for filename in files:

            filepath = os.path.join(
                root,
                filename,
            )

            try:

                mtime = os.path.getmtime(
                    filepath
                )

                size = os.path.getsize(
                    filepath
                )

                recent_files.append(
                    (
                        mtime,
                        filepath,
                        size,
                    )
                )

            except OSError:

                pass

    recent_files.sort(
        reverse=True
    )

    if not recent_files:

        print(
            "(no files)"
        )

    else:

        for mtime, filepath, size in recent_files[:10]:

            relative_path = os.path.relpath(
                filepath,
                cache_dir,
            )

            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(mtime),
            )

            print(
                f"{timestamp} | "
                f"{size / 1024:.1f} KB | "
                f"{relative_path}"
            )

    print("=" * 70)

cache_diagnostics()

# ============================================================
# HELPERS
# ============================================================

def run_command(command):
    print(f"[COMMAND] {command}")

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_PATH + ":" + env.get("PYTHONPATH", "")

    subprocess.run(
        command,
        shell=True,
        check=True,
        env=env,
    )


# ============================================================
# CACHED MODEL
# ============================================================

MODEL_ID = "fishaudio/s2-pro"

def find_cached_model(model_id):

    org, name = model_id.split("/", 1)

    cache_dir = Path(
        "/runpod-volume/huggingface-cache/hub"
    )

    model_dir = cache_dir / f"models--{org}--{name}"
    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        raise RuntimeError(
            f"HF cache not found: {snapshots_dir}"
        )

    snapshots = [
        p for p in snapshots_dir.iterdir()
        if p.is_dir()
    ]

    if not snapshots:
        raise RuntimeError(
            f"No cached snapshot found for {model_id}"
        )

    if len(snapshots) > 1:
        print(f"Found {len(snapshots)} cached snapshots.")

    return str(snapshots[0])


CHECKPOINT_PATH = find_cached_model(MODEL_ID)

print("S2-Pro cached model found:")
print(CHECKPOINT_PATH)

# ============================================================
# PULL GIT
# ============================================================

# run_command(
#     f"rm -rf '{REPO_PATH}'"
# )
def pull_git():
    print("Pulling Fish Speech repository...")

    run_command(
        f"cd '{REPO_PATH}' && git pull origin docker"
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap():

    print("=" * 70)
    print("FISH SPEECH SERVERLESS BOOTSTRAP")
    print("=" * 70)

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    # no need anymore because we use cached model now
    # os.makedirs(
    #     "/app/checkpoints/s2-pro",
    #     exist_ok=True,
    # )


    # --------------------------------------------------------
    # CLONE FISH SPEECH
    # --------------------------------------------------------

    if not os.path.exists(
        os.path.join(REPO_PATH, ".git")
    ):

        print("Fish Speech repository not found.")

        if os.path.exists(REPO_PATH):
            print(
                f"Removing incomplete directory: {REPO_PATH}"
            )

            run_command(
                f"rm -rf '{REPO_PATH}'"
            )

        print("Cloning Fish Speech repository...")

        run_command(
            "git clone -b docker "
            "https://github.com/seb2oo/fish-speech.git "
            f"'{REPO_PATH}'"
        )
        print("1")

    else:

        print(
            "Fish Speech repository already exists."
        )

    
    # --------------------------------------------------------
    # CREATE RUNTIME OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        "/app/fish-speech/output",
        exist_ok=True,
    )

    print("2")
    # Make Fish Speech available to this Python process
    if REPO_PATH not in sys.path:
        sys.path.insert(0, REPO_PATH)

    # --------------------------------------------------------
    # DOWNLOAD CHECKPOINT
    # --------------------------------------------------------
    print("3")
    codec_path = os.path.join(
        CHECKPOINT_PATH,
        "codec.pth",
    )

    if not os.path.exists(codec_path):
        ## not needed because we use the cache
        # print("Downloading S2-Pro checkpoint...")
        # run_command(
        #     "hf download fishaudio/s2-pro "
        #     "--local-dir /app/checkpoints/s2-pro"
        # )
        raise RuntimeError(
            f"codec.pth not found in cached S2-Pro model: {CHECKPOINT_PATH}"
        )

    else:

        print(
            "S2-Pro checkpoint already exists."
        )
    
    
    # --------------------------------------------------------
    # CREATE REFERENCE TOKENS
    # --------------------------------------------------------

    ## WAS USEFULL ONLY WHEN WE TRIED ONLY ON LANGUAGE
    # if not os.path.exists(REFERENCE_TOKENS):

    #     print(
    #         "Reference tokens not found."
    #     )

    #     if not os.path.exists(REFERENCE_WAV):
    #         raise FileNotFoundError(
    #             f"Reference WAV not found: {REFERENCE_WAV}"
    #         )

    #     print(
    #         "Encoding reference voice with DAC..."
    #     )

    #     run_command(
    #         "python3 -m fish_speech.models.dac.inference "
    #         f"-i '{REFERENCE_WAV}' "
    #         f"-o '{os.path.join('/app/fish-speech/output', 'reconstructed.wav')}' "
    #         f"--checkpoint-path '{os.path.join(CHECKPOINT_PATH, 'codec.pth')}' "
    #         "-d cuda"
    #     )

    #     # DAC inference creates reconstructed.npy next
    #     # to the output WAV.

    #     if not os.path.exists(REFERENCE_TOKENS):

    #         raise RuntimeError(
    #             "DAC finished but reconstructed.npy "
    #             "was not created."
    #         )

    #     print(
    #         "Reference tokens created."
    #     )

    # else:

    #     print(
    #         "Reference tokens already exist. "
    #         "Skipping DAC encoding."
    #     )

    print("4")
    j=0
    for ref in REFERENCE_TOKENS: # if all(os.path.exists(ref) for ref in REFERENCE_TOKENS):
        if not os.path.exists(ref):
            j+=1
    print("5")
    if j == 0 :
        print(
            "Reference tokens already exist. "
            "Skipping DAC encoding."
        )
    else:
        print(
            "Reference tokens not (ALL?) found, we delete the existing ones if exists."
        )
        # Clean output directory
        run_command("rm -rf /app/fish-speech/output/*")

        print("7")
        for ref in REFERENCE_WAV:
            if not os.path.exists(ref):
                raise FileNotFoundError(
                    f"Reference WAV not found: {REFERENCE_WAV}"
            )

        print(
            "Encoding reference voice with DAC..."
        )

        # this command is slow ! Maybe better to use the one from fill_pytorch_compile2.py or already place the neccessary file into the git
        for ref in REFERENCE_WAV:
            language = Path(ref).stem
            output_path = os.path.join(
            "/app/fish-speech/output",
            f"reconstructed_{language}.wav",
            )
            run_command(
                    "python3 -m fish_speech.models.dac.inference "
                    f"-i '{ref}' "
                    f"-o '{output_path}' "
                    f"--checkpoint-path '{os.path.join(CHECKPOINT_PATH, 'codec.pth')}' "
                    "-d cuda"
                )
        # DAC inference creates reconstructed.npy next
        # to the output WAV.

        j=0
        for ref in REFERENCE_TOKENS:
            if not os.path.exists(ref):
                j+=1

        if j == 0 :
            print(
                "Reference tokens created."
            )
        else:
            raise RuntimeError(
                "DAC finished but reconstructed.npy "
                "was not created."
            )

    
    # --------------------------------------------------------
    # run patch for pytorch ! 
    # --------------------------------------------------------
    subprocess.run([
    sys.executable,
    "/app/fish-speech/patch_enable_fx_cache_aot.py"
    ], check=True)

    subprocess.run([
        sys.executable,
        "/app/fish-speech/patch_force_aot_cache_save.py"
    ], check=True)

    subprocess.run([
        sys.executable,
        "/app/fish-speech/patch_retry_aot_on_tensorify_restart.py"
    ], check=True)

    print("PyTorch patches sucessfully applied")


# ============================================================
# STARTUP
# ============================================================

startup_start = time.perf_counter()

bootstrap()

print("=" * 70)
print("IMPORTING FISH SPEECH")
print("=" * 70)

from fish_speech.models.text2semantic.inference import (
    init_model,
    load_codec_model,
    generate_long,
    decode_to_audio,
)


# ============================================================
# LOAD S2-PRO
# ============================================================

print("=" * 70)
print("LOADING S2-PRO")
print("=" * 70)

t0 = time.perf_counter()

model, decode_one_token = init_model(
    CHECKPOINT_PATH,
    DEVICE,
    PRECISION,
    compile=COMPILE,
)

print(
    f"S2-Pro loaded in "
    f"{time.perf_counter() - t0:.2f}s"
)


# ============================================================
# LOAD DAC
# ============================================================

print("=" * 70)
print("LOADING DAC")
print("=" * 70)

t0 = time.perf_counter()

codec = load_codec_model(
    os.path.join(
        CHECKPOINT_PATH,
        "codec.pth",
    ),
    DEVICE,
    PRECISION,
)

print(
    f"DAC loaded in "
    f"{time.perf_counter() - t0:.2f}s"
)


# ============================================================
# LOAD REFERENCE TOKENS
# ============================================================

print("=" * 70)
print("LOADING REFERENCE VOICE")
print("=" * 70)

reference_codes_fr = torch.from_numpy(
    np.load(REFERENCE_TOKENS_FR)
)
print(
    f"Reference codes shape: "
    f"{reference_codes_fr.shape}"
)

reference_codes_en = torch.from_numpy(
    np.load(REFERENCE_TOKENS_EN)
)
print(
    f"Reference codes shape: "
    f"{reference_codes_en.shape}"
)

reference_codes_es = torch.from_numpy(
    np.load(REFERENCE_TOKENS_ES)
)
print(
    f"Reference codes shape: "
    f"{reference_codes_es.shape}"
)

reference_codes_de = torch.from_numpy(
    np.load(REFERENCE_TOKENS_DE)
)
print(
    f"Reference codes shape: "
    f"{reference_codes_de.shape}"
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    torch.cuda.synchronize()

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
    )

    print(
        f"GPU memory reserved: "
        f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
    )


# ============================================================
# WORKER READY
# ============================================================

startup_time = time.perf_counter() - startup_start

print("=" * 70)
print("WORKER READY")
print("=" * 70)

print(
    f"Total startup time: "
    f"{startup_time:.2f}s"
)

print(
    f"torch.compile: "
    f"{COMPILE}"
)

print("=" * 70)


# ============================================================
# GENERATE AUDIO
# ============================================================

def generate_audio(text, language):
    global INIT_CACHE

    print("=" * 70)
    print("GENERATING AUDIO")
    print("=" * 70)


    match language:
        case "fr":
            PROMPT_TEXT = PROMPT_TEXT_FR
            reference_codes = reference_codes_fr

        case "en":
            PROMPT_TEXT = PROMPT_TEXT_EN
            reference_codes = reference_codes_en

        case "es":
            PROMPT_TEXT = PROMPT_TEXT_ES
            reference_codes = reference_codes_es

        case "de":
            PROMPT_TEXT = PROMPT_TEXT_DE
            reference_codes = reference_codes_de

        case _:
            raise ValueError(f"Unknown language: {language}")

    print(
        f"Text: {text}"
    )

    generation_start = time.perf_counter()

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    # Fish Speech / TorchInductor can produce a huge amount
    # of internal output during torch.compile.
    #
    # We intentionally hide stdout/stderr here so that:
    #
    # - "Compiling function..."
    # - Triton warnings
    # - 32449-step progress bars
    # - internal TorchInductor messages
    #
    # do not pollute the RunPod logs.
    #
    # Our own logs remain visible outside this block.

    with open(os.devnull, "w") as devnull:

        with contextlib.redirect_stdout(devnull), \
             contextlib.redirect_stderr(devnull):

            generator = generate_long(
                model=model,
                device=DEVICE,
                decode_one_token=decode_one_token,

                text=text,

                num_samples=1,

                max_new_tokens=0,

                top_p=TOP_P,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,

                compile=False,# we don't use the mega cache as test from fill_pytorch_compile2.py are not sucessfful

                iterative_prompt=True,
                chunk_length=CHUNK_LENGTH,

                prompt_text=[
                    PROMPT_TEXT
                ],

                prompt_tokens=[
                    reference_codes
                ],
            )

            generated_codes = []

            for response in generator:

                if response.action == "sample":

                    generated_codes.append(
                        response.codes
                    )

    if not generated_codes:

        raise RuntimeError(
            "Fish Speech generated no audio codes."
        )

    generated_codes = torch.cat(
        generated_codes,
        dim=1,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    num_tokens = generated_codes.shape[1]

    print(
        f"Generated tokens: {num_tokens}"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    if generation_time > 0:

        print(
            f"Generation speed: "
            f"{num_tokens / generation_time:.2f} tok/s"
        )


    # --------------------------------------------------------
    # DAC DECODE
    # --------------------------------------------------------

    decode_start = time.perf_counter()

    audio = decode_to_audio(
        generated_codes.to(DEVICE),
        codec,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    decode_time = (
        time.perf_counter()
        - decode_start
    )

    print(
        f"DAC decode time: "
        f"{decode_time:.2f}s"
    )


    # --------------------------------------------------------
    # WAV IN MEMORY
    # --------------------------------------------------------

    wav_buffer = io.BytesIO()

    audio_numpy = (
        audio
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    sf.write(
        wav_buffer,
        audio_numpy,
        codec.sample_rate,
        format="WAV",
    )

    wav_bytes = wav_buffer.getvalue()

    # --------------------------------------------------------
    # TORCHINDUCTOR CACHE DIAGNOSTIC
    # --------------------------------------------------------

    cache_diagnostics()

    
    total_time = (
        generation_time
        + decode_time
    )

    duration = (
        len(audio_numpy)
        / codec.sample_rate
    )

    # --------------------------------------------------------
    # Save in the cache
    # --------------------------------------------------------

    
    ## rsync is missing from my docker .. therfore I used cp-a below to gain time
    # run_command(
    # "rsync -a /app/torchinductor-cache/ /runpod-volume/torchinductor-cache/"
    # )


    ## sur server normal prend 300 secondes .... maybe too many files...
    # t0 = time.perf_counter()
    # subprocess.run(
    # [
    #     "cp",
    #     "-a",
    #     "/app/torchinductor-cache/.",
    #     "/runpod-volume/torchinductor-cache/",
    # ],
    # check=True,
    # )
    # print("time to synchronise in the cache ::",time.perf_counter()-t0)

    if INIT_CACHE :
        INIT_CACHE=  False
        t0 = time.perf_counter()

        tmp_tar = "/runpod-volume/torchinductor-cache.tar.tmp"

        subprocess.run(
            [
                "tar",
                "-cf",
                tmp_tar,
                "-C",
                CACHE_DIR,
                ".",
            ],
            check=True,
        )

        os.replace(tmp_tar, CACHE_TAR)

        print(
            f"Time to create TorchInductor cache archive: "
            f"{time.perf_counter() - t0:.2f}s"
        )

    

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print(
        f"Audio duration: "
        f"{duration:.2f}s"
    )

    print(
        f"Total inference time: "
        f"{total_time:.2f}s"
    )

    print("=" * 70)

    return {
        "audio_base64": base64.b64encode(
            wav_bytes
        ).decode("utf-8"),

        "sample_rate": codec.sample_rate,

        "duration": duration,

        "num_tokens": num_tokens,

        "generation_time": generation_time,

        "decode_time": decode_time,

        "total_time": total_time,
    }


# ============================================================
# RUNPOD HANDLER
# ============================================================

## handler basic du début ne générant que qqch pour le francais 
# def handler(job):

#     """
#     Expected request:

#     {
#         "input": {
#             "text": "Bonjour, ceci est un test."
#         }
#     }

#     For this diagnostic test, the same worker performs
#     3 consecutive generations.
#     """

#     job_input = job.get(
#         "input",
#         {},
#     )

#     text = job_input.get(
#         "text"
#     )

#     if not text:

#         raise ValueError(
#             "Missing required input: 'text'"
#         )


#     # --------------------------------------------------------
#     # TEST TEXTS
#     # --------------------------------------------------------

#     test_texts = [
#         text,

#         "Ceci est la deuxième génération effectuée "
#         "par le même worker Fish Speech.",

#         "Et ceci est la troisième génération. "
#         "Le modèle devrait maintenant être complètement chaud.",
#     ]


#     # --------------------------------------------------------
#     # RUN 3 GENERATIONS
#     # --------------------------------------------------------

#     print("=" * 70)
#     print("RUNNING 3 CONSECUTIVE GENERATIONS")
#     print("=" * 70)

#     results = []

#     for i, test_text in enumerate(test_texts, start=1):

#         print()
#         print("=" * 70)
#         print(f"GENERATION {i}/3")
#         print("=" * 70)

#         result = generate_audio(test_text)

#         results.append(result)

#         print(
#             f"Generation {i}/3 finished in "
#             f"{result['total_time']:.2f}s"
#         )


#     # --------------------------------------------------------
#     # SUMMARY
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print("3-GENERATION TEST SUMMARY")
#     print("=" * 70)

#     for i, result in enumerate(results, start=1):

#         print(
#             f"Generation {i}: "
#             f"{result['num_tokens']} tokens | "
#             f"{result['generation_time']:.2f}s generation | "
#             f"{result['generation_time'] / result['num_tokens']:.2f}s/token | "
#             f"{result['num_tokens'] / result['generation_time']:.2f} tok/s | "
#             f"{result['duration']:.2f}s audio"
#         )

#     print("=" * 70)


#     # --------------------------------------------------------
#     # RETURN
#     # --------------------------------------------------------

#     # Return the audio from the LAST generation.
#     # The statistics of all 3 generations are returned
#     # so we can compare warm-up vs steady-state performance.

#     return {
#         "audio_base64": results[-1]["audio_base64"],

#         "sample_rate": results[-1]["sample_rate"],

#         "duration": results[-1]["duration"],

#         "num_tokens": results[-1]["num_tokens"],

#         "generation_time": results[-1]["generation_time"],

#         "decode_time": results[-1]["decode_time"],

#         "total_time": results[-1]["total_time"],

#         "test_generations": [
#             {
#                 "generation": i + 1,
#                 "duration": result["duration"],
#                 "num_tokens": result["num_tokens"],
#                 "generation_time": result["generation_time"],
#                 "decode_time": result["decode_time"],
#                 "total_time": result["total_time"],
#             }
#             for i, result in enumerate(results)
#         ],
#     }

## handler de test pour les 3 langues
# def handler(job):

#     """
#     Expected request:

#    {
#         "input": {
#             "text": "Bonjour, ceci est un test.",
#             "language": "fr",
#             "reload_anything":false,
#             "del_cache":false
#         }
#     }

#     Supported languages:
#         fr = French
#         en = English
#         es = Spanish
#         de = German

#     For this diagnostic test, the same worker performs
#     3 consecutive generations in the selected language.
#     """

#     job_input = job.get(
#         "input",
#         {},
#     )

#     text = job_input.get(
#         "text"
#     )

#     language = job_input.get(
#         "language"
#     )

#     reload = job_input.get(
#             "reload_anything",False
#         )

#     del_cache = job_input.get(
#                 "del_cache",False
#             )
    

#     if not text:

#         raise ValueError(
#             "Missing required input: 'text'"
#          )

#     if not language:

#         raise ValueError(
#             "Missing required input: 'language'"
#         )

#     if not isinstance(reload, bool):
#         raise ValueError(
#             "Input 'reload_anything' must be a boolean"
#         )

#     if not isinstance(del_cache, bool):
#         raise ValueError(
#             "Input 'del_cache' must be a boolean"
#         )
    
    
#      if reload:
#          pull_git()
#     ## deleted at the end is better
#     # if del_cache:
#     #     print("cache is erased")
#     #     run_command("rm -rf /runpod-volume/torchinductor-cache/*")
#     #     cache_diagnostics()


#     # --------------------------------------------------------
#     # VALIDATE LANGUAGE
#     # --------------------------------------------------------

#     supported_languages = {
#         "fr": "French",
#         "en": "English",
#         "es": "Spanish",
#         "de": "German",
#     }

#     if language not in supported_languages:

#         raise ValueError(
#             f"Unsupported language: '{language}'. "
#             f"Supported languages: {list(supported_languages.keys())}"
#         )


#     # --------------------------------------------------------
#     # TEST TEXTS
#     # --------------------------------------------------------

#     test_texts_by_language = {

#         "fr": [
#             text,

#             "Ceci est la deuxième génération effectuée "
#             "par le même worker Fish Speech.",

#             "Et ceci est la troisième génération. "
#             "Le modèle devrait maintenant être complètement chaud.",
#         ],

#         "en": [
#             text,

#             "This is the second generation performed "
#             "by the same Fish Speech worker.",

#             "And this is the third generation. "
#             "The model should now be completely warmed up.",
#         ],

#         "es": [
#             text,

#             "Esta es la segunda generación realizada "
#             "por el mismo worker de Fish Speech.",

#             "Y esta es la tercera generación. "
#             "El modelo debería estar ahora completamente caliente.",
#         ],

#         "de": [
#             text,

#             "Dies ist die zweite Generierung, die "
#             "vom selben Fish Speech Worker durchgeführt wird.",

#             "Und dies ist die dritte Generierung. "
#             "Das Modell sollte jetzt vollständig aufgewärmt sein.",
#         ],
#     }


#     test_texts = test_texts_by_language[language]


#     # --------------------------------------------------------
#     # RUN 3 GENERATIONS
#     # --------------------------------------------------------

#     print("=" * 70)
#     print("RUNNING 3 CONSECUTIVE GENERATIONS")
#     print("=" * 70)

#     print(
#         f"Language: {supported_languages[language]} ({language})"
#     )

#     results = []

#     for i, test_text in enumerate(test_texts, start=1):

#         print()
#         print("=" * 70)
#         print(f"GENERATION {i}/3")
#         print("=" * 70)

#         print(f"Text: {test_text}")

#         result = generate_audio(
#             test_text,
#             language=language,
#         )

#         results.append(result)

#         print(
#             f"Generation {i}/3 finished in "
#             f"{result['total_time']:.2f}s"
#         )


#     # --------------------------------------------------------
#     # SUMMARY
#     # --------------------------------------------------------

#     print()
#     print("=" * 70)
#     print("3-GENERATION TEST SUMMARY")
#     print("=" * 70)

#     for i, result in enumerate(results, start=1):

#         print(
#             f"Generation {i}: "
#             f"{result['num_tokens']} tokens | "
#             f"{result['generation_time']:.2f}s generation | "
#             f"{result['generation_time'] / result['num_tokens']:.2f}s/token | "
#             f"{result['num_tokens'] / result['generation_time']:.2f} tok/s | "
#             f"{result['duration']:.2f}s audio"
#         )

#     print("=" * 70)



#     # --------------------------------------------------------
#     # DELETE TORCHINDUCTOR CACHE
#     # --------------------------------------------------------

#     if del_cache:

#         print()
#         print("=" * 70)
#         print("TORCHINDUCTOR CACHE BEFORE DELETION")
#         print("=" * 70)

#         cache_diagnostics()

#         run_command(
#             "rm -rf /runpod-volume/torchinductor-cache/*"
#         )

#         print()
#         print("TorchInductor cache deleted.")

#         print()
#         print("=" * 70)
#         print("TORCHINDUCTOR CACHE AFTER DELETION")
#         print("=" * 70)

#         cache_diagnostics()


#     # --------------------------------------------------------
#     # RETURN
#     # --------------------------------------------------------

#     # Return the audio from the LAST generation.
#     # The statistics of all 3 generations are returned
#     # so we can compare warm-up vs steady-state performance.

#     return {

#         "audio_base64":
#             results[-1]["audio_base64"],

#         "sample_rate":
#             results[-1]["sample_rate"],

#         "duration":
#             results[-1]["duration"],

#         "num_tokens":
#             results[-1]["num_tokens"],

#         "generation_time":
#             results[-1]["generation_time"],

#         "decode_time":
#             results[-1]["decode_time"],

#         "total_time":
#             results[-1]["total_time"],

#         "language":
#             language,

#         "test_generations": [

#             {
#                 "generation": i + 1,
#                 "text": test_texts[i],
#                 "language": language,
#                 "duration": result["duration"],
#                 "num_tokens": result["num_tokens"],
#                 "generation_time": result["generation_time"],
#                 "decode_time": result["decode_time"],
#                 "total_time": result["total_time"],
#             }

#             for i, result in enumerate(results)
#         ],
#     }

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


    # --------------------------------------------------------
    # VALIDATE LANGUAGE
    # --------------------------------------------------------

    supported_languages = {
        "fr": "French",
        "en": "English",
        "es": "Spanish",
        "de": "German",
    }

    if language not in supported_languages:

        raise ValueError(
            f"Unsupported language: '{language}'. "
            f"Supported languages: {list(supported_languages.keys())}"
        )


    # --------------------------------------------------------
    # RUN  GENERATIONS
    # --------------------------------------------------------

    print("=" * 70)
    print("RUNNING GENERATION")
    print("=" * 70)

    print(
        f"Language: {supported_languages[language]} ({language})"
    )


    result = generate_audio(
        text,
        language=language,
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATION TEST SUMMARY")
    print("=" * 70)

    print(
        f"Generation : "
        f"{result['num_tokens']} tokens | "
        f"{result['generation_time']:.2f}s generation | "
        f"{result['generation_time'] / result['num_tokens']:.2f}s/token | "
        f"{result['num_tokens'] / result['generation_time']:.2f} tok/s | "
        f"{result['duration']:.2f}s audio"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # DELETE TORCHINDUCTOR CACHE
    # --------------------------------------------------------

    if del_cache:

        print()
        print("=" * 70)
        print("TORCHINDUCTOR CACHE BEFORE DELETION")
        print("=" * 70)

        cache_diagnostics()
        # old way before to use  .tar
        # run_command(
        #     "rm -rf /runpod-volume/torchinductor-cache/*"
        # )
        run_command(
                "rm -rf /runpod-volume/torchinductor-cache.tar"
            )

        print()
        print("TorchInductor cache deleted.")

        print()
        print("=" * 70)
        print("TORCHINDUCTOR CACHE AFTER DELETION")
        print("=" * 70)

        cache_diagnostics()


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    # Return the audio from the LAST generation.
    # The statistics of all 3 generations are returned
    # so we can compare warm-up vs steady-state performance.

    return {

        "audio_base64":
            result["audio_base64"],

        "sample_rate":
            result["sample_rate"],

        "duration":
            result["duration"],

        "num_tokens":
            result["num_tokens"],

        "generation_time":
            result["generation_time"],

        "decode_time":
            result["decode_time"],

        "total_time":
            result["total_time"],

        "language":
            language
    }


# ============================================================
# START RUNPOD SERVERLESS
# ============================================================

if __name__ == "__main__":

    runpod.serverless.start(
        {
            "handler": handler
        }
    )



##Code de test dans l'onglet "request" de runpod

# {
#   "input": {
#     "text": "Bonjour, ceci est un test.",
#     "language": "fr",
#     "reload_anything":false,
#     "del_cache":false
#   }
# }