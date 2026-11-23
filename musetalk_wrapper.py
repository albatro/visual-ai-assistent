"""
MuseTalk v1.5 integration wrapper for the Visual AI Assistant.

Provides a simple API:
  1. MuseTalkEngine.load_models()      — load all ML models (once)
  2. MuseTalkEngine.prepare_avatar()   — pre-process a reference photo (once per avatar)
  3. MuseTalkEngine.generate_frames()  — audio → list of BGR frames (real-time)

Designed for RTX 5060 Ti 16GB. Runs UNet + VAE in fp16 (~4-6 GB VRAM).
"""

import os
import sys
import copy
import glob
import json
import pickle
import shutil
import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add MuseTalk to Python path
MUSETALK_DIR = Path(__file__).parent / "MuseTalk"
sys.path.insert(0, str(MUSETALK_DIR))

from transformers import WhisperModel
from musetalk.utils.utils import load_all_model, datagen
from musetalk.utils.preprocessing_mp import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing


@dataclass
class AvatarData:
    """Pre-processed avatar data ready for inference."""
    avatar_id: str
    frame_list_cycle: list = field(repr=False)
    coord_list_cycle: list = field(repr=False)
    input_latent_list_cycle: list = field(repr=False)
    mask_list_cycle: list = field(repr=False)
    mask_coords_list_cycle: list = field(repr=False)


class MuseTalkEngine:
    """
    High-level wrapper around MuseTalk v1.5 for real-time lip-sync generation.

    Usage:
        engine = MuseTalkEngine()
        engine.load_models()
        avatar = engine.prepare_avatar("my_avatar", "photo.jpg")
        frames = engine.generate_frames(avatar, "speech.wav")
        # frames is a list of BGR numpy arrays (H, W, 3)
    """

    def __init__(
        self,
        gpu_id: int = 0,
        batch_size: int = 20,
        fps: int = 25,
        extra_margin: int = 10,
        cache_dir: Optional[str] = None,
    ):
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.fps = fps
        self.extra_margin = extra_margin
        self.cache_dir = Path(cache_dir or (MUSETALK_DIR / "results" / "v15" / "avatars"))

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False

        # Model references (populated by load_models)
        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.whisper = None
        self.audio_processor = None
        self.fp = None
        self.weight_dtype = None

    def load_models(self) -> None:
        """Load all ML models into GPU memory. Call once at startup."""
        if self.models_loaded:
            return

        models_dir = MUSETALK_DIR / "models"

        # Save current dir and change to MuseTalk root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(str(MUSETALK_DIR))

        print("[MuseTalk] Loading VAE + UNet + PE...")
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=str(models_dir / "musetalkV15" / "unet.pth"),
            vae_type="sd-vae-ft-mse",
            unet_config=str(models_dir / "musetalkV15" / "musetalk.json"),
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)

        # Move to fp16
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)

        print("[MuseTalk] Loading Whisper...")
        whisper_dir = str(models_dir / "whisper")
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        print("[MuseTalk] Loading FaceParsing...")
        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

        self.models_loaded = True

        # Restore original working directory
        os.chdir(original_cwd)
        print("[MuseTalk] All models loaded successfully.")

    def prepare_avatar(
        self,
        avatar_id: str,
        image_path: str,
        force_recreate: bool = False,
    ) -> AvatarData:
        """
        Pre-process a reference photo for lip-sync generation.
        Results are cached to disk for fast reload.

        Args:
            avatar_id: Unique identifier for this avatar.
            image_path: Path to the reference photo (JPG/PNG).
            force_recreate: If True, re-process even if cache exists.

        Returns:
            AvatarData with pre-processed materials.
        """
        assert self.models_loaded, "Call load_models() first"

        avatar_path = self.cache_dir / avatar_id
        full_imgs_path = avatar_path / "full_imgs"
        mask_out_path = avatar_path / "mask"
        latents_path = avatar_path / "latents.pt"
        coords_path = avatar_path / "coords.pkl"
        mask_coords_path = avatar_path / "mask_coords.pkl"

        # Check cache
        if not force_recreate and latents_path.exists():
            print(f"[MuseTalk] Loading cached avatar '{avatar_id}'...")
            return self._load_cached_avatar(avatar_id, avatar_path)

        # Create directories
        for d in [avatar_path, full_imgs_path, mask_out_path]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"[MuseTalk] Preparing avatar '{avatar_id}' from {image_path}...")

        # Determine if input is video or image
        ext = Path(image_path).suffix.lower()
        is_video = ext in ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm')

        if is_video:
            # Extract frames from video
            cap = cv2.VideoCapture(image_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {image_path}")
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(full_imgs_path / f"{frame_idx:08d}.png"), frame)
                frame_idx += 1
            cap.release()
            print(f"[MuseTalk] Extracted {frame_idx} frames from video")
        else:
            # Single image
            src_img = cv2.imread(image_path)
            if src_img is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            cv2.imwrite(str(full_imgs_path / "00000000.png"), src_img)

        # Extract landmarks
        input_img_list = sorted(glob.glob(str(full_imgs_path / "*.[jpJP][pnPN]*[gG]")))
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, upperbondrange=0)

        # Encode face crops into VAE latents — keep coords/frames/latents in sync
        # by dropping any frame where face detection failed (placeholder bbox).
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        kept_coords = []
        kept_frames = []
        input_latent_list = []
        dropped = 0
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                dropped += 1
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + self.extra_margin, frame.shape[0])
            if x2 <= x1 or y2 <= y1:
                dropped += 1
                continue
            crop_frame = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized)
            input_latent_list.append(latents)
            kept_coords.append([x1, y1, x2, y2])
            kept_frames.append(frame)
        if dropped:
            print(f"[MuseTalk] Dropped {dropped} frames with no face detected")
        if not kept_frames:
            raise RuntimeError("No usable frames — face detector found nothing")

        # Create cycling lists (forward + reverse for natural looping)
        frame_list_cycle = kept_frames + kept_frames[::-1]
        coord_list_cycle = kept_coords + kept_coords[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # Generate masks for blending
        mask_list_cycle = []
        mask_coords_list_cycle = []
        for i, frame in enumerate(tqdm(frame_list_cycle, desc="Generating masks")):
            cv2.imwrite(str(full_imgs_path / f"{i:08d}.png"), frame)
            x1, y1, x2, y2 = coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], fp=self.fp, mode="jaw"
            )
            cv2.imwrite(str(mask_out_path / f"{i:08d}.png"), mask)
            mask_list_cycle.append(mask)
            mask_coords_list_cycle.append(crop_box)

        # Save to cache
        with open(coords_path, "wb") as f:
            pickle.dump(coord_list_cycle, f)
        with open(mask_coords_path, "wb") as f:
            pickle.dump(mask_coords_list_cycle, f)
        torch.save(input_latent_list_cycle, str(latents_path))

        # Save avatar info
        info = {"avatar_id": avatar_id, "image_path": image_path, "version": "v15"}
        with open(avatar_path / "avatar_info.json", "w") as f:
            json.dump(info, f)

        print(f"[MuseTalk] Avatar '{avatar_id}' prepared and cached.")
        return AvatarData(
            avatar_id=avatar_id,
            frame_list_cycle=frame_list_cycle,
            coord_list_cycle=coord_list_cycle,
            input_latent_list_cycle=input_latent_list_cycle,
            mask_list_cycle=mask_list_cycle,
            mask_coords_list_cycle=mask_coords_list_cycle,
        )

    def _load_cached_avatar(self, avatar_id: str, avatar_path: Path) -> AvatarData:
        """Load pre-processed avatar data from disk cache."""
        input_latent_list_cycle = torch.load(
            str(avatar_path / "latents.pt"), weights_only=False
        )
        with open(avatar_path / "coords.pkl", "rb") as f:
            coord_list_cycle = pickle.load(f)
        with open(avatar_path / "mask_coords.pkl", "rb") as f:
            mask_coords_list_cycle = pickle.load(f)

        full_imgs_path = avatar_path / "full_imgs"
        input_img_list = sorted(
            glob.glob(str(full_imgs_path / "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(Path(x).stem),
        )
        frame_list_cycle = read_imgs(input_img_list)

        mask_path = avatar_path / "mask"
        mask_img_list = sorted(
            glob.glob(str(mask_path / "*.[jpJP][pnPN]*[gG]")),
            key=lambda x: int(Path(x).stem),
        )
        mask_list_cycle = read_imgs(mask_img_list)

        return AvatarData(
            avatar_id=avatar_id,
            frame_list_cycle=frame_list_cycle,
            coord_list_cycle=coord_list_cycle,
            input_latent_list_cycle=input_latent_list_cycle,
            mask_list_cycle=mask_list_cycle,
            mask_coords_list_cycle=mask_coords_list_cycle,
        )

    @torch.no_grad()
    def generate_frames(self, avatar: AvatarData, audio_path: str) -> list[np.ndarray]:
        """
        Generate lip-synced video frames from audio.

        Args:
            avatar: Pre-processed avatar data.
            audio_path: Path to WAV audio file.

        Returns:
            List of BGR numpy arrays (H, W, 3), one per video frame at self.fps.
        """
        assert self.models_loaded, "Call load_models() first"

        t0 = time.time()

        # Extract audio features
        whisper_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        print(f"[MuseTalk] Audio processed in {(time.time() - t0)*1000:.0f}ms")

        # Generate frames batch by batch
        gen = datagen(whisper_chunks, avatar.input_latent_list_cycle, self.batch_size)
        video_num = len(whisper_chunks)

        output_frames = []
        idx = 0
        t1 = time.time()

        for whisper_batch, latent_batch in tqdm(
            gen, total=int(np.ceil(video_num / self.batch_size)), desc="Generating"
        ):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)

            for res_frame in recon:
                n = len(avatar.coord_list_cycle)
                bbox = avatar.coord_list_cycle[idx % n]
                ori_frame = copy.deepcopy(avatar.frame_list_cycle[idx % n])
                x1, y1, x2, y2 = bbox
                res_frame = cv2.resize(
                    res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                )
                mask = avatar.mask_list_cycle[idx % n]
                mask_crop_box = avatar.mask_coords_list_cycle[idx % n]
                combine_frame = get_image_blending(
                    ori_frame, res_frame, bbox, mask, mask_crop_box
                )
                output_frames.append(combine_frame)
                idx += 1

        elapsed = time.time() - t1
        fps_actual = len(output_frames) / elapsed if elapsed > 0 else 0
        print(
            f"[MuseTalk] Generated {len(output_frames)} frames in {elapsed:.2f}s "
            f"({fps_actual:.1f} FPS)"
        )
        return output_frames

    @torch.no_grad()
    def generate_frames_streaming(
        self, avatar: AvatarData, audio_path: str
    ):
        """
        Generator version — yields frames one by one as they are produced.
        Useful for real-time streaming to a WebSocket/WebRTC client.

        Yields:
            BGR numpy arrays (H, W, 3).
        """
        assert self.models_loaded, "Call load_models() first"

        whisper_features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype
        )
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=self.fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        gen = datagen(whisper_chunks, avatar.input_latent_list_cycle, self.batch_size)
        idx = 0

        for whisper_batch, latent_batch in gen:
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)

            for res_frame in recon:
                n = len(avatar.coord_list_cycle)
                bbox = avatar.coord_list_cycle[idx % n]
                ori_frame = copy.deepcopy(avatar.frame_list_cycle[idx % n])
                x1, y1, x2, y2 = bbox
                res_frame = cv2.resize(
                    res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                )
                mask = avatar.mask_list_cycle[idx % n]
                mask_crop_box = avatar.mask_coords_list_cycle[idx % n]
                combine_frame = get_image_blending(
                    ori_frame, res_frame, bbox, mask, mask_crop_box
                )
                yield combine_frame
                idx += 1


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuseTalk wrapper test")
    parser.add_argument("--image", required=True, help="Path to reference photo")
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    engine = MuseTalkEngine(fps=args.fps)
    engine.load_models()

    avatar = engine.prepare_avatar("test_avatar", args.image)
    frames = engine.generate_frames(avatar, args.audio)

    # Save as video
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"Saved {len(frames)} frames to {args.output}")
