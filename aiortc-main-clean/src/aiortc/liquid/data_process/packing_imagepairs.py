import os
import json
import argparse

from datasets import load_dataset

def main(args):
    
    file_list = os.listdir(args.temp_path)
    data_list = [os.path.join(args.temp_path,filename) for filename in file_list ]
    data_files = {"train":data_list }
    my_dataset = load_dataset("json", data_files=data_files, split='train', streaming=False, num_proc=40)
 
    my_dataset = my_dataset.shuffle(seed=42)
    my_dataset.save_to_disk(args.save_path, num_shards=256)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    args = parser.parse_args()
    main(args)

