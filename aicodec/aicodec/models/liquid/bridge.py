"""Liquid VLM bridge (FoundationVision/Liquid + Chameleon VQGAN tokenizer).

Ported from the legacy aiortc-fork codec ``aiortc/codecs/liquid_tokenizer.py``
(commit d505a20) to the plugin bridge API.  Checkpoints are NOT vendored;
download them with ``aicodec.models.download`` into
``aicodec/models/liquid/checkpoints/``:

    checkpoints/
      chameleon/vqgan.yaml
      chameleon/vqgan.ckpt
      model/               (Liquid VQ-LLM weights + tokenizer)

``src/`` is vendored upstream code (Liquid MIT / Chameleon BSD-3-Clause),
see ``src/LICENSE``.
"""

import logging
from pathlib import Path
from threading import Thread
from typing import Iterator

import numpy as np
import PIL
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from ..base import VLMBridge
from .src.conversation import conv_templates
from .src.image_tokenizer import ImageTokenizer

logger = logging.getLogger(__name__)

IMAGE_TOKEN_INDEX = -200


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class LiquidBridge(VLMBridge):
    """Liquid VQ-LLM + Chameleon VQGAN image tokenizer."""

    def setup(self) -> None:
        model_path = self.checkpoints_path
        vqgan_cfg_path = model_path / "chameleon" / "vqgan.yaml"
        vqgan_ckpt_path = model_path / "chameleon" / "vqgan.ckpt"
        llm_model_path = model_path / "model"

        # Check for actual WEIGHTS, not directory existence: the model/
        # dir ships placeholder files (README, Liquid_download.py), so
        # llm_model_path.exists() is always true and would never warn
        # when the LLM weights are missing. The safetensors index (or
        # any *.safetensors shard) is only present after a real download.
        llm_weights_present = (
            llm_model_path / "model.safetensors.index.json").exists() or any(
            llm_model_path.glob("*.safetensors"))

        if not (
            vqgan_cfg_path.exists()
            and vqgan_ckpt_path.exists()
            and llm_weights_present
        ):
            logger.warning(
                "liquid checkpoints not found at %s "
                "(run aicodec/models/download.py first); "
                "tokenize/detokenize/generate will fail",
                model_path,
            )

        if self.enc_dec in ("enc", "dec"):
            self.image_tokenizer = ImageTokenizer(
                cfg_path=str(vqgan_cfg_path),
                ckpt_path=str(vqgan_ckpt_path),
                device=self.device,
            )
        elif self.enc_dec == "llm":
            self.image_tokenizer = ImageTokenizer(
                cfg_path=str(vqgan_cfg_path),
                ckpt_path=str(vqgan_ckpt_path),
                device=self.device,
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
            self.vqllm = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        else:
            raise ValueError(f"invalid enc_dec: {self.enc_dec!r}")

    @torch.no_grad()
    def tokenize(self, image) -> list[int]:
        pad_image = expand2square(image, (122, 116, 104))
        input_image = pad_image.resize((512, 512), PIL.Image.LANCZOS)
        tokens = self.image_tokenizer.img_tokens_from_pil(input_image)
        return tokens.tolist()

    @torch.no_grad()
    def detokenize(self, tokens) -> object:
        rec_img = self.image_tokenizer.pil_from_img_toks(
            torch.tensor(tokens).to(self.device)
        )
        return rec_img

    @torch.no_grad()
    def generate(self, image_tokens, user_prompt: str) -> Iterator[str]:
        image_tokens = torch.tensor(image_tokens)
        qs = user_prompt
        qs = "<boi><image><eoi>" + "\n" + qs
        conv = conv_templates["gemma"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        vqcode = image_tokens + len(self.text_tokenizer)
        text_ids = tokenizer_image_token(
            prompt, self.text_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(
            text_ids == IMAGE_TOKEN_INDEX
        )[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(
                text_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            if i < num_images:
                cur_input_ids.append(vqcode)
        input_ids = torch.cat(cur_input_ids, dim=0)
        streamer = TextIteratorStreamer(
            self.text_tokenizer,
            **{"skip_special_tokens": True, "skip_prompt": True},
        )
        generation_kwargs = dict(
            input_ids=input_ids.unsqueeze(0).to(self.device),
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            bos_token_id=self.text_tokenizer.bos_token_id,
            eos_token_id=self.text_tokenizer.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
        )
        # generate in a thread so the caller can consume the streamer
        # without blocking
        thread = Thread(target=self.vqllm.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    @torch.no_grad()
    def estimate_token_number(self, text: str) -> int:
        tokens = self.text_tokenizer(text, return_tensors="pt")
        return len(tokens["input_ids"][0])
