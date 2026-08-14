import torch
import scripts.inference as inf


class MuseTalkRuntime:

    def __init__(
        self,
        unet_model_path="models/musetalkV15/unet.pth",
        unet_config="models/musetalkV15/musetalk.json",
        vae_type="sd-vae",
        whisper_dir="models/whisper",
        version="v15",
        gpu_id=0,
        use_float16=True,
    ):

        print("\n==============================")
        print("Loading MuseTalk runtime...")
        print("==============================")

        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        print("Device:", self.device)

        # --------------------------------------------------
        # MODELS
        # --------------------------------------------------

        self.vae, self.unet, self.pe = inf.load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device,
        )

        if use_float16:
            print("Converting models to float16...")
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.weight_dtype = self.unet.model.dtype

        # --------------------------------------------------
        # AUDIO
        # --------------------------------------------------

        print("Loading AudioProcessor...")

        self.audio_processor = inf.AudioProcessor(
            feature_extractor_path=whisper_dir
        )

        print("Loading Whisper...")

        self.whisper = inf.WhisperModel.from_pretrained(
            whisper_dir
        )

        self.whisper = self.whisper.to(
            device=self.device,
            dtype=self.weight_dtype
        ).eval()

        self.whisper.requires_grad_(False)

        # --------------------------------------------------
        # FACE PARSER
        # --------------------------------------------------

        print("Loading FaceParsing...")

        if version == "v15":
            self.fp = inf.FaceParsing(
                left_cheek_width=0,
                right_cheek_width=0
            )
        else:
            self.fp = inf.FaceParsing()

        # --------------------------------------------------

        self.timesteps = torch.tensor(
            [0],
            device=self.device
        )

        self.version = version

        print("==============================")
        print("MuseTalk runtime READY")
        print("==============================\n")


runtime = MuseTalkRuntime()