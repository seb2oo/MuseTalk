import os
import cv2
import math
import copy
import torch
import glob
import shutil
import pickle
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel
import sys

from musetalk.utils.blending import get_image, get_image_prepare_material

import time

T0 = time.perf_counter()

def log_time(label):
    print(f"[TIMER] {label}: {time.perf_counter() - T0:.3f} s")

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

@torch.no_grad()
def main(args):
    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")
    log_time("after : Configure ffmpeg path")
    
    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")


    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path, 
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    log_time("after : Load model weights")

    # Convert models to half precision if float16 is enabled
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    log_time("after : Convert models to half precision if float16 is enabled")
    
    # Move models to specified device
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    log_time("after : Move models to specified device")
        
    # Initialize audio processor and Whisper model
    t_model_start = time.perf_counter()
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    log_time("after : Initialize audio processor and Whisper model")
    
    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()
    log_time("after : Initialize face parser with configurable parameters based on version")
    
    # Load inference configuration
    inference_config = OmegaConf.load(args.inference_config)
    print("Loaded inference config:", inference_config)
    log_time("after : Load inference configuration")
    
    # Process each task
    for task_nb,task_id in enumerate(inference_config):
        try:
            # Get task configuration
            video_path = inference_config[task_id]["video_path"]
            audio_path = inference_config[task_id]["audio_path"]
            if "result_name" in inference_config[task_id]:
                args.output_vid_name = inference_config[task_id]["result_name"]
            log_time(f"after : Get task configuration, task_nb = {task_nb}")
            
            # Set bbox_shift based on version
            if args.version == "v15":
                bbox_shift = 0  # v15 uses fixed bbox_shift
            else:
                bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)  # v1 uses config or default
            log_time(f"after : Set bbox_shift based on version, task_nb = {task_nb}")
            
            # Set output paths
            input_basename = os.path.basename(video_path).split('.')[0]
            audio_basename = os.path.basename(audio_path).split('.')[0]
            output_basename = f"{input_basename}_{audio_basename}"
            log_time(f"after : Set output paths, task_nb = {task_nb}")
            
            # Create temporary directories
            temp_dir = os.path.join(args.result_dir, f"{args.version}")
            os.makedirs(temp_dir, exist_ok=True)
            log_time(f"after : Create temporary directories, task_nb = {task_nb}")
            
            # Set result save paths
            result_img_save_path = os.path.join(temp_dir, output_basename)
            crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
            os.makedirs(result_img_save_path, exist_ok=True)
            log_time(f"after : Set result save paths, task_nb = {task_nb}")
            
            # Set output video paths
            if args.output_vid_name is None:
                output_vid_name = os.path.join(temp_dir, output_basename + ".mp4")
            else:
                output_vid_name = os.path.join(temp_dir, args.output_vid_name)
            output_vid_name_concat = os.path.join(temp_dir, output_basename + "_concat.mp4")
            log_time(f"after : Set output video paths, task_nb = {task_nb}")
            
            # Extract frames from source video
            save_dir_full = None
            if get_file_type(video_path) == "video":
                save_dir_full = os.path.join(temp_dir, input_basename)
                os.makedirs(save_dir_full, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = args.fps
            elif os.path.isdir(video_path):
                input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                fps = args.fps
            else:
                raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
            log_time(f"after : Extract frames from source video, task_nb = {task_nb}")

            # Extract audio features
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            log_time(f"after : get_audio_feature, task_nb = {task_nb}")
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, 
                device, 
                weight_dtype, 
                whisper, 
                librosa_length,
                fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )
            log_time(f"after : get_whisper_chunk, task_nb = {task_nb}")
            
            # Preprocess input images
            if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                print("Using saved coordinates")
                with open(crop_coord_save_path, 'rb') as f:
                    coord_list = pickle.load(f)
                frame_list = read_imgs(input_img_list)
            else:
                print("Extracting landmarks... time-consuming operation")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
            
            print(f"Number of frames: {len(frame_list)}")        
            log_time(f"after : Preprocess input images, task_nb = {task_nb}") 

            # ----------------------------------------------------------
            # Pre-compute blending mask for static PNG input
            # ----------------------------------------------------------

            mask_array = None
            mask_crop_box = None

            if args.use_png:
                print("Static PNG mode: pre-computing blending mask...")

                x1, y1, x2, y2 = coord_list[0]

                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame_list[0].shape[0])

                mask_array, mask_crop_box = get_image_prepare_material(
                    frame_list[0],
                    [x1, y1, x2, y2],
                    fp=fp,
                    mode=args.parsing_mode
                )

                log_time("after : Pre-compute static PNG blending mask")
            
            # Process each frame
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256,256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)
            log_time(f"after : Process each frame, task_nb = {task_nb}") 
        
            # Smooth first and last frames
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
            log_time(f"after : Smooth first and last frames, task_nb = {task_nb}") 
            
            # Batch inference
            print("Starting inference")
            video_num = len(whisper_chunks)
            batch_size = args.batch_size
            gen = datagen(
                whisper_chunks=whisper_chunks,
                vae_encode_latents=input_latent_list_cycle,
                batch_size=batch_size,
                delay_frame=0,
                device=device,
            )
            
            res_frame_list = []
            total = int(np.ceil(float(video_num) / batch_size))
            log_time(f"after : Batch inference, task_nb = {task_nb}") 
            
            # Execute inference
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
                audio_feature_batch = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                for res_frame in recon:
                    res_frame_list.append(res_frame)
            log_time(f"after : Execute inference, task_nb = {task_nb}")
            
            # Pad generated images to original video size
            print("Padding generated images to original video size")
            blend_time = 0
            write_time = 0
            for i, res_frame in enumerate(tqdm(res_frame_list)):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                if args.version == "v15":
                    y2 = y2 + args.extra_margin
                    y2 = min(y2, frame.shape[0])
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                except:
                    continue
                
                # # Merge results with version-specific parameters
                # if args.version == "v15":
                #     t = time.perf_counter()
                #     combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
                #     blend_time += time.perf_counter() - t
                # else:
                #     combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)

                if args.version == "v15":

                    if args.use_png:
                        combine_frame = get_image(
                            ori_frame,
                            res_frame,
                            [x1, y1, x2, y2],
                            mode=args.parsing_mode,
                            fp=fp,
                            mask_array=mask_array,
                            crop_box=mask_crop_box
                        )
                    else:
                        combine_frame = get_image(
                            ori_frame,
                            res_frame,
                            [x1, y1, x2, y2],
                            mode=args.parsing_mode,
                            fp=fp
                        )

                else:

                    combine_frame = get_image(
                        ori_frame,
                        res_frame,
                        [x1, y1, x2, y2],
                        fp=fp
                    )

                t = time.perf_counter()
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                write_time += time.perf_counter() - t
            log_time(f"after : Pad generated images to original video size, task_nb = {task_nb}")
            print(f"[TIMER] Total blending: {blend_time:.3f} s")
            print(f"[TIMER] Total PNG writing: {write_time:.3f} s")

            # Save prediction results
            temp_vid_path = f"{temp_dir}/temp_{input_basename}_{audio_basename}.mp4"
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid_path}"
            print("Video generation command:", cmd_img2video)
            os.system(cmd_img2video)   
            
            
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid_path} {output_vid_name}"
            print("Audio combination command:", cmd_combine_audio) 
            os.system(cmd_combine_audio)
            log_time(f"after : Save prediction results, task_nb = {task_nb}")
            
            # Clean up temporary files
            shutil.rmtree(result_img_save_path)
            os.remove(temp_vid_path)
            if save_dir_full is not None and os.path.exists(save_dir_full):
                shutil.rmtree(save_dir_full)
            if not args.saved_coord:
                os.remove(crop_coord_save_path)
                    
            print(f"Results saved to {output_vid_name}")
            log_time(f"after : Clean up temporary files, task_nb = {task_nb}")
        except Exception as e:
            print("Error occurred during processing:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    parser.add_argument(
    "--use_png",
    action="store_true",
    help="Use a static PNG image and reuse the blending mask for all generated frames"
    )
    args = parser.parse_args()
    main(args)
