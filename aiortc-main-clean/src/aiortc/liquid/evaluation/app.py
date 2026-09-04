import time
from threading import Thread

import gradio as gr
import torch
import PIL
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from chameleon.inference.image_tokenizer import ImageTokenizer
from T2I_Eval.mjhq_generation import sample
# from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import TextIteratorStreamer
from VQA_Eval.conversation import  conv_templates


import os 
# os.system("pip uninstall -y gradio") 
# os.system("pip install gradio==4.44.1")
# os.system("pip install gradio_client==1.3.0")


IMAGE_TOKEN_INDEX=-200
PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 20px; margin-bottom: 1px; opacity: 0.55;">Liquid-7B</h1>
</div>
"""

CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""


title_html = """
<div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
<h1 style="margin: 0; line-height: 1; text-align: center;"> Liquid: Language Models are Scalable and Unified Multi-modal Generators </h1>
</div>
"""

links_html = f"""
<center><font size=3><a href='https://foundationvision.github.io/Liquid/'>Liquid</a> has been open-sourced on <a href='https://huggingface.co/Junfeng5/Liquid_V1_7B'>😊 Huggingface</a> and <a href='https://github.com/FoundationVision/Liquid'>🌟 GitHub</a>. If you find Liquid useful, a like❤️ or a star🌟 would be appreciated.</font></center>
"""

introduction = f"""
 Liquid explores the potential of a single LLM as a multimodal generator and its scaling laws. It achieves the level of diffusion models in visual generation and discovers the mutual enhancement between understanding and generation. More details can be found on the project <a href='https://foundationvision.github.io/Liquid/'> homepage</a> and in the <a href='https://arxiv.org/abs/2412.04332'> paper</a>. """



model_id = 'Junfeng5/Liquid_V1_7B'
model_id = 'Liquid_models/Liquid_V1_7B'
tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side='left')
vqllm = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    ).to("cuda:0")

stop_flag = False

ori_vocabe_size = len(tokenizer)

vqgan_cfg_path = "chameleon/vqgan.yaml"
vqgan_ckpt_path = "chameleon/vqgan.ckpt"
image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0",)




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
    

def bot_streaming_I2T(message, history):
    print(message)
    global stop_flag
    stop_flag = True
    time.sleep(0.2)
    stop_flag = False
    torch.cuda.empty_cache()
    if message["files"]:
        # message["files"][-1] is a Dict or just a string
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]
    try:
        if image is None:
            # Handle the case where image is None
            gr.Error("You need to upload an image for LLaVA to work.")
    except NameError:
        # Handle the case where 'image' is not defined at all
        gr.Error("You need to upload an image for LLaVA to work.")

    qs = message['text']
    qs = '<boi><image><eoi>' + '\n' + qs
    conv = conv_templates['gemma'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    
    print(prompt)
    image = Image.open(image).convert('RGB')
    pad_image = expand2square(image, (122, 116, 104) )
    input_image = pad_image.resize((512,512), PIL.Image.LANCZOS)
    with torch.no_grad():
        vq_code =  image_tokenizer.img_tokens_from_pil(input_image) 
        vqcode = vq_code.cpu() 
        vqcode = vqcode+ len(tokenizer)

        text_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append( vqcode )
        input_ids = torch.cat(cur_input_ids, dim=0)
        # input_embeddings = vqllm.embed_tokens(input_ids)
        inputs =  {
            "input_ids":input_ids.unsqueeze(0).to("cuda:0"),
            "max_new_tokens":1024,
            "bos_token_id":tokenizer.bos_token_id,  # Begin of sequence token
            "eos_token_id":tokenizer.eos_token_id,  # End of sequence token
            "pad_token_id":tokenizer.pad_token_id,  # Pad token
            }
        streamer = TextIteratorStreamer(tokenizer, **{"skip_special_tokens": False, "skip_prompt": True})

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
        thread = Thread(target=vqllm.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            time.sleep(0.06)
            yield generated_text
        


def show_gallery(images):
    gallery = gr.Gallery(images, label="Gallery", columns=4, height="auto",preview=True,scale=0.05)  # 设置两行两列的布局
    return gallery

def bot_streaming_T2I(message, history,guidance_scale, temperature, top_K, top_P):
 
    global stop_flag
    stop_flag = True
    time.sleep(0.2)
    stop_flag = False
    
    text_inputs = [message]*4  # generate 4 samples once
    uncondition_text_inputs = ['<unconditional><boi>']*len(text_inputs)
    for i in range(len(text_inputs)):
        text_inputs[i] = text_inputs[i]+' Generate an image based on this description.<boi>'

    ori_batchsize = len(text_inputs)

    if guidance_scale>1:
        model_inputs = tokenizer(text_inputs+uncondition_text_inputs, return_tensors="pt",padding=True).to("cuda:0")
    else:
        model_inputs = tokenizer(text_inputs, return_tensors="pt",padding=True).to("cuda:0")
    with torch.no_grad():
        sampling_kwargs={'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        input_ids = model_inputs['input_ids']
        cur_len = input_ids.shape[1]
        model_kwargs = {'attention_mask':model_inputs['attention_mask']  , 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        pred_tokens = []
        for i in tqdm(range(1024)):
            if stop_flag:
                print("generation is stoped")
                del sampling_kwargs
                del model_inputs
                del outputs
                torch.cuda.empty_cache()
                break
            model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if i > 0 and guidance_scale>1:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            else:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            next_token_logits = outputs.logits[:, -1:, :]
            
            if guidance_scale>1:
                cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0) 
                cfg_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
                half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                pred_tokens.append(half_next_token)
                next_token = torch.cat([half_next_token,half_next_token])


            else:
                next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                pred_tokens.append(next_token)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

        del sampling_kwargs
        del model_inputs
        del outputs
        image_vq_id = torch.cat(pred_tokens,dim=1)-ori_vocabe_size
        image_vq_id = torch.clamp(image_vq_id, min=0, max=8191)
        
        generated_image_list = []
        for index, generate_id in enumerate(image_vq_id):
            rec_img = image_tokenizer.pil_from_img_toks(generate_id)
            generated_image_list.append(rec_img)
            # rec_img.save('{}/{}.jpg'.format(image_save_pth,str(idx)))

        torch.cuda.empty_cache()
         # yield gr.Image(value=generated_image_list[0], label="Generated Image", show_download_button=True) 
        yield show_gallery(generated_image_list)

def bot_streaming_T2T(message, history,temperature):
    print(message)
    global stop_flag
    stop_flag = True
    time.sleep(0.2)
    stop_flag = False
    torch.cuda.empty_cache()
    qs = message 
    conv = conv_templates['gemma'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
 
    print(prompt)
    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
        streamer = TextIteratorStreamer(tokenizer, **{"skip_special_tokens": False, "skip_prompt": True})

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
        thread = Thread(target=vqllm.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield generated_text
 

chatbot_T2I=gr.Chatbot(placeholder=PLACEHOLDER,height=600)
chat_input_T2I = gr.Textbox(placeholder="Enter text prompts...", show_label=False)

chatbot_I2T=gr.Chatbot(placeholder=PLACEHOLDER, scale=1)
chat_input_I2T = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

chatbot_T2T=gr.Chatbot(placeholder=PLACEHOLDER, scale=1)
chat_input_T2T = gr.Textbox(placeholder="Enter text prompts...", show_label=False)


with gr.Blocks(fill_height=True) as demo:

    gr.Markdown(title_html)
    gr.Markdown(links_html)  
    gr.Markdown(introduction)

    with gr.Tab("Text To Image"):

        description="Enter a text prompt or simply try one of the examples below to generate 4 images at once. Click to display the full image. You can configure hyperparameters for image generation in the Advanced Settings. "
        gr.Markdown(description)  
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                guidance_scale = gr.Slider(1.0, 20.0, value=7.0, label="Guidance Scale")
                temperature = gr.Slider(0.0, 1.0, value=0.9, label="temperature")
                top_K = gr.Slider(1, 8192, value=4096, label="Top K")
                top_P = gr.Slider(0.0, 1.0, value=0.99, label="Top P")
 
        gr.ChatInterface(
            fn=bot_streaming_T2I,
            examples=[
                ["young blue dragon with horn lightning in the style of dd fantasy full body",5.0, 0.9,4096,0.99],
                ["A majestic Goddes of beauty, charming dressed in a regal, jeweled gown and ornate crown, her golden hair cascading down her back, in the style of Pino Daeni",5.0, 0.9,4096,0.99],
                ["A highly realistic, closeup photograph of a beautiful 35 year old redread woman writing in her journal, sitting on her balcony wearing warm, stylish outfits. Shot on a Canon EOS R5, the image boasts sharp focus and intricate details. The heartwarming scene conveys love, connection, and the crisp winter atmosphere, dramatic lighting.",5.0, 0.9,4096,0.99],
                ["Portrait of an asian woman. She has pink violet hair style with modern complex hairdressing. The background is dark with cyberpunk neon lights. Inspired by Cyberpunk 2077 and Blade Runner. Ultra realistic picture. To capture the image, you will use a fullframe DSLR or mirrorless camera with a highresolution sensor, an aperture of f2.8 or wider, and a shutter speed of 1500 second or faster. You will use natural light and reflectors to create a balanced and welllit image, and will experiment with different angles and compositions to create the most i",5.0, 0.9,4096,0.99],
                ["female character fantasy world, for fantasy story, protagonist, interesting and detailed clothes, beautiful, medieval fantasy  cinematic shot  photo taken by canon, photo taken by fuji, photo taken by kodak  incredibly detailed, sharpen, details  professional lighting , film lighting  350mm  lightroom  cinematography, hyper realism, cinematic, film quality",5.0, 0.9,4096,0.99],
                ["strawberries splashing, swirling liquid, realism, octane render, raytracing",5.0, 0.9,4096,0.99],
                      ["hedgehog face, floating in space, wearing space suit no helmet, cinematic, 50mm f1.8, unreal engine 5",5.0, 0.9,4096,0.99],
                      ["artificial intelligence, revolution, publishing, writer, hyperrealistic",5.0, 0.9,4096,0.99],
                      ["A pig dressed as a mason, by Bill Gekas",5.0, 0.9,4096,0.99],
                      ],
            stop_btn="Stop Generation",
            additional_inputs = [guidance_scale, temperature, top_K, top_P],
            additional_inputs_accordion="⚙️ Advanced Settings",
            multimodal=False,
            cache_examples=False,
            textbox=chat_input_T2I,
            chatbot=chatbot_T2I,
            fill_height=True,
            )

   

 
    with gr.Tab("Image To Text"):
        gr.ChatInterface(
            fn=bot_streaming_I2T,
            examples=[ {"text": "How to make this pastry?", "files": ["samples/baklava.png"]}],
            description="Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
            stop_btn="Stop Generation",
            multimodal=True,
            cache_examples=False,
            textbox=chat_input_I2T,
            chatbot=chatbot_I2T,
            )
        
    with gr.Tab("Text To Text"):
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                texttemperature = gr.Slider(0.0, 1.0, value=0.9, label="texttemperature")
                
        gr.ChatInterface(
            fn=bot_streaming_T2T,
            examples=[["Tell me about Paris.", 0.9]],
            description="Chat with Liquid without images.",
            stop_btn="Stop Generation",
            additional_inputs = [texttemperature],
            additional_inputs_accordion="⚙️ Advanced Settings",
            cache_examples=False,
            multimodal=False,
            textbox=chat_input_T2T,
            chatbot=chatbot_T2T,
            )
# demo.queue(api_open=False)
demo.launch(allowed_paths=["./"], share=False )