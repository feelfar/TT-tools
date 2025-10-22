from __future__ import annotations
import argparse
import io
import os
import struct
import sys
import logging
import pathlib
import zlib
from typing import Optional
from PIL import Image
import numpy as np
import torch
import tempfile
import uuid

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tt_img_dec_v2_node.py
A small, resilient decoder for "v2" image container files.
This script attempts to detect several common layouts:
 - Single PNG/JPEG file (copied/converted)
 - A custom "TTV2" container with a simple header and per-frame sized blobs
 - A simple stream of length-prefixed image frames (4-byte BE size + blob)

Usage:
    python tt_img_dec_v2_node.py input_file_or_dir -o out_dir

This file is intended to be a standalone starting-point. Adjust the
container parsing logic to match the exact project specification if needed.
"""



LOG = logging.getLogger("tt_img_dec_v2_node")
BUFFER_SIZE = 64 * 1024


class TTImgDecV2Node:
    """
    Decoder class that supports both CLI-style usage (providing a source path)
    and a ComfyUI node style where an image/tensor is provided and the
    embedded container is extracted.
    """
    # ComfyUI node metadata (kept similar to tt_img_dec_node.py)
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_filename": ("STRING", {"default": "tt_img_dec_v2_file", "multiline": False}),
            },
            "optional": {
                "usage_notes": ("STRING", {"default": "从图片中提取 v2 容器并保存到 ComfyUI output 目录", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "FLOAT")
    RETURN_NAMES = ("image", "audio", "file_path", "fps")
    FUNCTION = "extract_file_from_image"
    CATEGORY = "TT Tools"
    OUTPUT_NODE = True

    def __init__(self, src: Optional[str] = None, out_dir: Optional[str] = None, overwrite: bool = False):
        self.overwrite = overwrite
        # CLI mode: src provided
        if src:
            self.src = pathlib.Path(src)
            self.out_dir = pathlib.Path(out_dir) if out_dir else (self.src.parent / (self.src.stem + "_dec"))
            self.out_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Node mode: try to obtain ComfyUI output directory via folder_paths, fallback to ./output
            try:
                import folder_paths
                if hasattr(folder_paths, 'get_output_directory'):
                    self.out_dir = pathlib.Path(folder_paths.get_output_directory())
                elif hasattr(folder_paths, 'output_directory'):
                    self.out_dir = pathlib.Path(folder_paths.output_directory)
                else:
                    self.out_dir = pathlib.Path("output")
            except Exception:
                self.out_dir = pathlib.Path("output")
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        if self.src.is_dir():
            for p in sorted(self.src.iterdir()):
                if p.is_file():
                    try:
                        self._process_file(p)
                    except Exception:
                        LOG.exception("Failed to process %s", p)
        elif self.src.is_file():
            self._process_file(self.src)
        else:
            LOG.error("Source %s does not exist", self.src)
            raise FileNotFoundError(self.src)

    def _process_file(self, path: pathlib.Path):
        LOG.info("Processing: %s", path)
        with path.open("rb") as f:
            head = f.read(16)
            f.seek(0)
            # PNG magic
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                # try to open image and extract embedded V2 bytes from pixels first
                f.seek(0)
                raw = f.read()
                try:
                    bio = io.BytesIO(raw)
                    im = Image.open(bio).convert('RGB')
                    arr = np.array(im)
                    payload_bytes = self._extract_bytes_from_image_array(arr)
                    found = self._parse_ttv2_from_bytes(payload_bytes, path.stem)
                    if found > 0:
                        return
                except Exception:
                    # fallback to saving the image as-is
                    pass
                # if no embedded data found, save the image normally
                self._save_single_image_from_stream(io.BytesIO(raw), path.stem, 0)
                return

            # JPEG magic (very common header start)
            if head.startswith(b"\xff\xd8\xff"):
                f.seek(0)
                raw = f.read()
                try:
                    bio = io.BytesIO(raw)
                    im = Image.open(bio).convert('RGB')
                    arr = np.array(im)
                    payload_bytes = self._extract_bytes_from_image_array(arr)
                    found = self._parse_ttv2_from_bytes(payload_bytes, path.stem)
                    if found > 0:
                        return
                except Exception:
                    pass
                self._save_single_image_from_stream(io.BytesIO(raw), path.stem, 0)
                return

            # Custom container magic: try legacy TTV2/TTIMGV2 and newer TTv2
            if head.startswith(b"TTV2") or head.startswith(b"TTIMGV2") or head.startswith(b"TTv2"):
                LOG.debug("Detected TTV2/TTv2 container")
                self._parse_ttv2_container(f, path.stem)
                return

            # Generic: try parsing as sequence of length-prefixed frames
            # read until EOF: 4-byte BE size + blob
            if self._looks_like_length_prefixed_stream(f):
                LOG.debug("Detected length-prefixed frame stream")
                self._parse_length_prefixed_stream(f, path.stem)
                return

            # Unknown: try to interpret whole file as an image
            try:
                f.seek(0)
                img = Image.open(f)
                img.load()
                out_path = self.out_dir / (path.stem + ".png")
                if out_path.exists() and not self.overwrite:
                    LOG.info("Skipping (exists): %s", out_path)
                    return
                img.save(out_path, "PNG")
                LOG.info("Saved single image: %s", out_path)
                return
            except Exception:
                LOG.exception("Unknown file format and cannot decode: %s", path)
                raise ValueError("Unsupported or corrupted file: " + str(path))

    def _save_single_image_from_stream(self, stream, base_name: str, idx: int):
        stream.seek(0)
        try:
            img = Image.open(stream)
            img.load()
        except Exception:
            # Maybe compressed container with one frame
            stream.seek(0)
            raw = stream.read()
            img = self._try_decode_bytes(raw)
            if img is None:
                raise
        out_path = self.out_dir / f"{base_name}_{idx:04d}.png"
        if out_path.exists() and not self.overwrite:
            LOG.info("Skipping (exists): %s", out_path)
            return
        img.save(out_path, "PNG")
        LOG.info("Saved: %s", out_path)

    def _extract_bytes_from_image_array(self, image_array: np.ndarray) -> bytes:
        """
        从 RGB 图像数组按行优先顺序提取字节流（R,G,B,R,G,B,...）。
        这对应 enc_v2 中的直接 RGB 字节流写入方式（bits_per_channel==8）。
        """
        try:
            h, w, c = image_array.shape
            if c < 3:
                # not RGB-like
                return b''
            # Ensure uint8
            arr = image_array.astype(np.uint8)
            # Flatten rows, then cols, then channels (R,G,B)
            return arr.reshape(-1, c)[:, :3].ravel().tobytes()
        except Exception:
            return b''

    def _parse_ttv2_from_bytes(self, data: bytes, base_name: str):
        """
        Parse TTv2 packets from a raw bytes buffer (which may be from a file
        or reconstructed from image pixels). Logic extracted from previous
        _parse_ttv2_container implementation.
        """
        if not data or len(data) < 12:
            LOG.warning("TTv2 data too short for parsing")
            return 0

        offset = 0
        found = 0
        while True:
            idx = data.find(b'TTv2', offset)
            if idx == -1:
                break
            if idx + 10 > len(data):
                LOG.warning("TTv2 header truncated at %d", idx)
                break
            hdr_len = int.from_bytes(data[idx+4:idx+8], 'big')
            crc16 = int.from_bytes(data[idx+8:idx+10], 'big')
            inner_start = idx + 10
            inner_end = inner_start + hdr_len
            if inner_end > len(data):
                LOG.warning("TTv2 inner truncated at %d (need %d bytes)", idx, hdr_len)
                break
            inner = data[inner_start:inner_end]

            try:
                calc = self._crc16_ccitt(inner)
                if calc != crc16:
                    LOG.debug("CRC mismatch for TTv2 at %d: got %04x expected %04x", idx, crc16, calc)
            except Exception:
                LOG.exception("CRC compute failed")

            if len(inner) < 7:
                LOG.warning("TTv2 inner too short at %d", idx)
                offset = inner_end
                continue
            version = inner[0]
            flags = inner[1]
            ext_len = inner[2]
            pos = 3
            if pos + ext_len + 4 > len(inner):
                LOG.warning("TTv2 inner ext/data truncated at %d", idx)
                offset = inner_end
                continue
            ext = inner[pos:pos+ext_len].decode('utf-8', errors='ignore')
            pos += ext_len
            data_len = int.from_bytes(inner[pos:pos+4], 'big')
            pos += 4
            payload = inner[pos:pos+data_len]

            is_multi = bool(flags & 0x02)
            if is_multi:
                # scan for nested TTv2 inside payload
                sub_off = 0
                nested_found = 0
                while True:
                    sub_idx = payload.find(b'TTv2', sub_off)
                    if sub_idx == -1:
                        break
                    nested = payload[sub_idx:]
                    try:
                        tmp = tempfile.NamedTemporaryFile(delete=False)
                        try:
                            tmp.write(nested)
                            tmp.flush()
                            tmp.close()
                            self._process_file(pathlib.Path(tmp.name))
                        finally:
                            try:
                                os.unlink(tmp.name)
                            except Exception:
                                pass
                    except Exception:
                        LOG.exception("Failed processing nested TTv2 payload")
                    nested_found += 1
                    sub_off = sub_idx + 4
                if nested_found == 0:
                    out_raw = self.out_dir / f"{base_name}_ttv2_{found:04d}.{ext or 'bin'}"
                    with out_raw.open('wb') as of:
                        of.write(payload)
                    LOG.info("Saved raw multi payload -> %s", out_raw)
            else:
                img = self._try_decode_bytes(payload)
                if img is not None:
                    out_path = self.out_dir / f"{base_name}_{found:04d}.png"
                    if out_path.exists() and not self.overwrite:
                        LOG.info("Skipping (exists): %s", out_path)
                    else:
                        img.save(out_path, 'PNG')
                        LOG.info("Saved TTv2 image -> %s", out_path)
                else:
                    try:
                        dec = zlib.decompress(payload)
                        img = self._try_decode_bytes(dec)
                        if img is not None:
                            out_path = self.out_dir / f"{base_name}_{found:04d}.png"
                            img.save(out_path, 'PNG')
                            LOG.info("Saved TTv2 zlib image -> %s", out_path)
                        else:
                            raise ValueError('not image')
                    except Exception:
                        out_raw = self.out_dir / f"{base_name}_ttv2_{found:04d}.{ext or 'bin'}"
                        with out_raw.open('wb') as of:
                            of.write(payload)
                        LOG.info("Saved raw TTv2 payload -> %s", out_raw)

            found += 1
            offset = inner_end

        return found

    def _parse_ttv2_container(self, f, base_name: str):
        f.seek(0)
        raw = f.read()
        # If file is an image (PNG/JPEG) raw will be compressed file bytes; try to open as image and
        # reconstruct the embedded byte stream from pixel data first.
        tried_from_image = False
        try:
            bio = io.BytesIO(raw)
            im = Image.open(bio)
            im = im.convert('RGB')
            arr = np.array(im)
            tried_from_image = True
            payload_bytes = self._extract_bytes_from_image_array(arr)
            found = self._parse_ttv2_from_bytes(payload_bytes, base_name)
            if found > 0:
                LOG.info("Parsed %d TTv2 packets from embedded image %s", found, base_name)
                return
        except Exception:
            # not an image or failed to reconstruct; fall back to parsing raw bytes
            pass

        # fallback: try parsing raw file bytes as TTv2 container
        found2 = self._parse_ttv2_from_bytes(raw, base_name)
        if found2 > 0:
            LOG.info("Parsed %d TTv2 packets from raw bytes %s", found2, base_name)
            return

        # If nothing parsed, fall back to length-prefixed stream parser
        LOG.warning("No TTv2 packets parsed (tried_from_image=%s), falling back to length-prefixed stream parser", tried_from_image)
        f.seek(0)
        self._parse_length_prefixed_stream(f, base_name)

    def _looks_like_length_prefixed_stream(self, f) -> bool:
        f.seek(0)
        size_data = f.read(4)
        if len(size_data) < 4:
            return False
        (sz,) = struct.unpack(">I", size_data)
        if 0 < sz < 50 * 1024 * 1024:  # arbitrary 50MB sanity check
            # check next bytes for PNG/JPEG header
            blob = f.read(min(8, sz))
            if blob.startswith(b"\x89PNG\r\n\x1a\n") or blob.startswith(b"\xff\xd8\xff"):
                return True
        return False

    def _parse_length_prefixed_stream(self, f, base_name: str):
        f.seek(0)
        idx = 0
        while True:
            header = f.read(4)
            if not header or len(header) < 4:
                break
            (sz,) = struct.unpack(">I", header)
            if sz == 0:
                LOG.debug("Zero-sized frame encountered, stopping.")
                break
            data = f.read(sz)
            if len(data) < sz:
                LOG.warning("Unexpected EOF reading frame %d", idx)
                break
            img = self._try_decode_bytes(data)
            if img is None:
                try:
                    dec = zlib.decompress(data)
                    img = self._try_decode_bytes(dec)
                except Exception:
                    img = None
            if img is None:
                out_raw = self.out_dir / f"{base_name}_frame_{idx:04d}.bin"
                with out_raw.open("wb") as of:
                    of.write(data)
                LOG.info("Saved raw blob: %s", out_raw)
            else:
                out_path = self.out_dir / f"{base_name}_{idx:04d}.png"
                if out_path.exists() and not self.overwrite:
                    LOG.info("Skipping (exists): %s", out_path)
                else:
                    img.save(out_path, "PNG")
                    LOG.info("Saved frame %d -> %s", idx, out_path)
            idx += 1

    def _try_decode_bytes(self, data: bytes) -> Optional[Image.Image]:
        try:
            bio = io.BytesIO(data)
            img = Image.open(bio)
            img.load()
            return img
        except Exception:
            return None

    # --- ComfyUI node wrapper -------------------------------------------------
    def extract_file_from_image(self, image, output_filename: str = "tt_img_dec_v2_file", usage_notes: Optional[str] = None):
        """
        ComfyUI-facing wrapper. Accepts an IMAGE (torch tensor / numpy / PIL),
        writes it to a temporary file and attempts to decode it using the
        same file parsing logic as the CLI flow. Returns the first detected
        image/audio output (if any) together with the saved file path and fps.
        """
        try:
            # Convert incoming image to numpy uint8 HWC
            if hasattr(image, 'cpu'):
                img_np = image.cpu().numpy()
            else:
                img_np = np.array(image)

            # Handle common tensor shapes: (B,H,W,C) or (H,W,C)
            if img_np.ndim == 4:
                img_np = img_np[0]

            # If in CHW and floats 0..1, convert to HWC
            if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4) and img_np.shape[0] != img_np.shape[2]:
                # assume CHW
                img_np = np.transpose(img_np, (1, 2, 0))

            # normalize to uint8 if float
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)

            # Save to temporary file
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = pathlib.Path(tmp.name)
            tmp.close()
            try:
                pil = Image.fromarray(img_np)
                pil.save(tmp_path, format="PNG")
            except Exception:
                # fallback: try writing raw bytes
                with tmp_path.open('wb') as wf:
                    if isinstance(image, (bytes, bytearray)):
                        wf.write(image)
                    else:
                        wf.write(img_np.tobytes())

            # Record existing files in output dir
            before = set(p.name for p in self.out_dir.iterdir())

            # Process the temporary file using existing logic
            try:
                self._process_file(tmp_path)
            except Exception:
                LOG.exception("Failed to process temporary image %s", tmp_path)

            # Find new files
            after = [p for p in self.out_dir.iterdir() if p.name not in before]
            after_sorted = sorted(after)

            if not after_sorted:
                return (None, None, "", 0.0)

            # Prefer image outputs
            image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
            audio_exts = {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a'}

            # If multiple image frames generated with naming base_0000.., gather as sequence
            imgs = [p for p in after_sorted if p.suffix.lower() in image_exts]
            auds = [p for p in after_sorted if p.suffix.lower() in audio_exts]

            if imgs:
                # If multiple frames, stack them
                frames = []
                for p in imgs:
                    try:
                        im = Image.open(p).convert('RGB')
                        arr = np.array(im).astype(np.float32) / 255.0
                        frames.append(arr)
                    except Exception:
                        LOG.exception("Failed to open extracted image %s", p)
                if not frames:
                    return (None, None, str(imgs[0]), 0.0)
                frames_array = np.stack(frames, axis=0)
                tensor = torch.from_numpy(frames_array)
                return (tensor, None, str(imgs[0]), 0.0)

            if auds:
                try:
                    import soundfile as sf
                    data, sr = sf.read(str(auds[0]))
                    audio_tensor = torch.from_numpy(data.astype(np.float32))
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                    elif audio_tensor.ndim == 2:
                        audio_tensor = audio_tensor.transpose(0,1).unsqueeze(0)
                    return (None, {'waveform': audio_tensor, 'sample_rate': sr}, str(auds[0]), 0.0)
                except Exception:
                    LOG.exception("Failed to process extracted audio %s", auds[0])
                    return (None, None, str(auds[0]), 0.0)

            # Otherwise return path to first new file
            return (None, None, str(after_sorted[0]), 0.0)

        except Exception:
            LOG.exception("extract_file_from_image failed")
            return (None, None, "", 0.0)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Decode TT image v2 containers / streams")
    p.add_argument("src", help="Source file or directory")
    p.add_argument("-o", "--out", help="Output directory (defaults to <src>_dec)", default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args(argv)


def setup_logging(verbose: bool):
    h = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    LOG.addHandler(h)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    setup_logging(args.verbose)
    node = TTImgDecV2Node(args.src, args.out, overwrite=args.force)
    try:
        node.run()
    except Exception:
        LOG.exception("Decoder failed")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())