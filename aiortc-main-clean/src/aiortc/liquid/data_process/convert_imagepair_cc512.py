import os
import json
import argparse
import cv2
import PIL

from PIL import Image, ImageFile
import argparse
from multiprocessing.pool import ThreadPool
import time
import torch
from vqgan.image_tokenizer import ImageTokenizer
from tqdm import tqdm
import numpy as np

from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)



def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
    Width,Height = ori_image.size
    factor = min(Width,Height)/min(tgt_width,tgt_height)
    input_image = ori_image.resize((int(Width/factor),int(Height/factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size   # Get dimensions

    left = (resize_width - tgt_width)//2
    top = (resize_height - tgt_height)//2
    right = (resize_width + tgt_width)//2
    bottom = (resize_height + tgt_height)//2
    # Crop the center of the image
    input_image = input_image.crop((left, top, right, bottom))
    return input_image

def process_item(data,image_tokenizer):
    image_path = data['image_path']
    try:
        ori_image = Image.open(image_path).convert("RGB")
    except:
        print('load error', image_path)
        return None

    original_ratio = max(ori_image.size) / min(ori_image.size) # remove images with ratio>2
    if original_ratio >2 :
        return None    

    input_image = center_crop_image(ori_image)

    with torch.no_grad():
        vqcode = image_tokenizer.img_tokens_from_pil(input_image) 
        vqcode = vqcode.cpu().tolist()

    new_anno={    
                'data_type': 'image_text',
                'text': data['text'],
                'length': len(data['text'])+len(vqcode)*4,
                'vqcode_512': json.dumps(vqcode),
                'vqcode_multi768': 'no',
                'width': 'no',
                'height': 'no',
                }
    return  new_anno



def main(args):
    
    chunk_idx = args.chunk_idx
    num_chunks=args.num_chunks
    
    
    vqgan_cfg_path = "{}/vqgan.yaml".format(args.vqgan_path)
    vqgan_ckpt_path = "{}/vqgan.ckpt".format(args.vqgan_path)
    image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)

    
    all_datas = []
    with open(args.input_pairs, 'r', encoding="utf-8") as f:
        for line in f:
            all_datas.append(json.loads(line))  
    
    chunked_filenames = np.array_split(all_datas, num_chunks)
    subset = chunked_filenames[chunk_idx].tolist()
    print(len(subset))
    
    pool = ThreadPool(processes=args.num_processes)
    thread_list = []    
    
    total_images_num = len(subset)
    valid_pair_list = []
    pbar = tqdm(total=len(subset))
    pbar.set_description('process')
    def update(result):
        if result is not None:
            valid_pair_list.append(result)
        pbar.update()

    for image_index in range(total_images_num):
        if args.Async:
            async_result = pool.apply_async(func=process_item, args=(subset[image_index],image_tokenizer,), callback=update)  
        else:
            async_result = pool.apply(func=process_item, args=(subset[image_index],image_tokenizer,), callback=update)  
            thread_list.append(async_result)

    pool.close()
    pool.join()

    valid_images = len(valid_pair_list)

    print('keep {} images from {} pairs in rank {}'.format(str(valid_images),str(total_images_num),str(chunk_idx)))

    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)

    with open( os.path.join(args.temp_path, str(chunk_idx).zfill(6)+'.jsonl'), 'w') as f:
        for item in valid_pair_list:
            f.write(json.dumps(item)+'\n')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_pairs', type=str, default='/path/to/save/josnl',help='jsonl file, where image pairs meta saved')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    parser.add_argument('--chunk_idx', type=int, default=0, help='chunk id')
    parser.add_argument('--num_chunks', type=int, default=1, help='num of chunks')
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--Async', type=bool, default=True)
    parser.add_argument('--vqgan_path', type=str, default='/path/to/vqgan_weights',help='where vqgan.yaml and vqgan.ckpt saved')
    
    
    args = parser.parse_args()
    main(args)

