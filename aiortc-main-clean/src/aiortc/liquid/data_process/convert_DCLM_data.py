import os
import zstandard as zstd
from tqdm import tqdm
import io
import json
import argparse
from datasets import load_dataset


def read_jsonl_zst(file_path):
    with open(file_path, 'rb') as file:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(file)
        stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
        for line in stream:
            yield json.loads(line)


def main(args):
    input_path = args.input_path
    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)
        
    files_list = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        for file in filenames:
            if file.endswith('zst'):
                full_path = os.path.join(dirpath, file)
                files_list.append(full_path)
    files_list.sort()
    # files_list = files_list[:10] 
    for idx in tqdm(range(len(files_list))):
        filepath = files_list[idx]
        alldata_in_one_path = list(read_jsonl_zst(filepath))
        all_datas = []
        for data in alldata_in_one_path:
            all_datas.append(
                {    
                    'data_type': 'text_pretrain',
                    'text': data['text'],
                    'length': len(data['text']),
                    'vqcode_512': 'no',
                    'vqcode_multi768': 'no',
                    'width': 'no',
                    'height': 'no',
                }
            )
        with open(  os.path.join(args.temp_path, str(idx).zfill(6)+'.jsonl'  ), 'w') as f:
            for item in all_datas:
                f.write(json.dumps(item)+'\n')

    file_list = os.listdir(args.temp_path)
    data_list = [args.temp_path+filename for filename in file_list ]
    data_files = {"train":data_list }
    my_dataset = load_dataset("json", data_files=data_files, split='train', streaming=False, num_proc=40)
 
    my_dataset = my_dataset.shuffle(seed=42)
    my_dataset.save_to_disk(args.save_path, num_shards=256)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_path', type=str, default='/path/to/ori_dclm_files', help='path where jsonl.zst files saved')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf files')
    args = parser.parse_args()
    main(args)

