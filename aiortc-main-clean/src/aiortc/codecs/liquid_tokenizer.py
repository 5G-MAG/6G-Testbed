import logging
import random
from struct import pack, unpack_from
from typing import Optional, Type, TypeVar, cast
from pathlib import Path

import torch
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from aiortc.liquid.evaluation.chameleon.inference.image_tokenizer import ImageTokenizer
from aiortc.liquid.evaluation.VQA_Eval.conversation import  conv_templates
from threading import Thread

import numpy as np

IMAGE_TOKEN_INDEX=-200

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

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def get_model_path():
    return Path(__file__).parent.parent / "liquid" / "checkpoints"

class liquidBridge:
    def __init__(self, enc_dec):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = get_model_path()
        vqgan_cfg_path = model_path / "chameleon" / "vqgan.yaml"
        vqgan_ckpt_path = model_path / "chameleon" / "vqgan.ckpt"
        llm_model_path = model_path / "model"

        if vqgan_cfg_path.exists() == False or vqgan_ckpt_path.exists() == False or llm_model_path.exists() == False:
            logging.warning(f"Model does not found at: {model_path}")

        if enc_dec == "enc" or enc_dec == "dec":
            self.image_tokenizer = ImageTokenizer(cfg_path=str(vqgan_cfg_path), 
                ckpt_path=str(vqgan_ckpt_path), device=self.device)
        if enc_dec == "llm":
            self.image_tokenizer = ImageTokenizer(cfg_path=str(vqgan_cfg_path), 
                ckpt_path=str(vqgan_ckpt_path), device=self.device)
            self.text_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
            self.vqllm = AutoModelForCausalLM.from_pretrained(
                str(llm_model_path),
                # attn_implementation='flash_attention_2',
                torch_dtype=torch.bfloat16,
                ).to('cuda')
    
    @torch.no_grad()
    def tokenize(self, image):
        pad_image = expand2square(image, (122, 116, 104) )
        input_image = pad_image.resize((512,512), PIL.Image.LANCZOS)
        tokens = self.image_tokenizer.img_tokens_from_pil(input_image)
        return tokens.tolist()
    
    @torch.no_grad()
    def detokenize(self, tokens):
        rec_img = self.image_tokenizer.pil_from_img_toks(torch.tensor(tokens).to(self.device))
        return rec_img

    @torch.no_grad()
    def generate(self, image_tokens, user_prompt = "What is in the video? answer directly"):
        image_tokens = torch.tensor(image_tokens)
        prompt = user_prompt
        qs = prompt
        qs = '<boi><image><eoi>' + '\n' + qs
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        vqcode = image_tokens + len(self.text_tokenizer)
        text_ids = tokenizer_image_token(prompt, self.text_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append( vqcode )
        input_ids = torch.cat(cur_input_ids, dim=0)
        inputs =  {
            "input_ids":input_ids.unsqueeze(0).to("cuda:0"),
            "max_new_tokens":1024,
            "bos_token_id":self.text_tokenizer.bos_token_id,  # Begin of sequence token
            "eos_token_id":self.text_tokenizer.eos_token_id,  # End of sequence token
            "pad_token_id":self.text_tokenizer.pad_token_id,  # Pad token
            }
        streamer = TextIteratorStreamer(self.text_tokenizer, **{"skip_special_tokens": True, "skip_prompt": True})

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024, do_sample=False)
        thread = Thread(target=self.vqllm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer
    
    @torch.no_grad()
    def estimate_token_number(self, text):
        tokens = self.text_tokenizer(text, return_tensors="pt")
        return len(tokens["input_ids"][0])