# Data

The optimization of data preprocessing significantly impacts the training efficiency of Liquid, especially when a large amount of training data is required. Data loading and shuffling can consume a substantial amount of time. In worse cases, if your dataset is large enough, even a super server with 1TB of RAM may not be able to load all the training data at once. Therefore, we first classify and provide recommendations for several data scenarios, along with two data preprocessing methods that can be used based on specific requirements.

## Scenario 1: Map-style Dataset

**Use Case:**  
- My training dataset is small (less than 10M images).  
- I need to frequently replace the image tokenizer and embed the tokenizer into the model (similar to replace CLIP in LLaVA).  

In this scenario, we can use a standard PyTorch **Map-style Dataset** to load the data. All data is stored in a `jsonl` format, where each line is a dictionary containing the following fields:

```python
{
  "data_type": "",  
  "text": "",  
  "vq_code": []  # or "image_path" if the image tokenizer is embedded in the model architecture
}
```



## Scenario 2: IterableDataset

**Use Case:**  

1. I have a very large training dataset (think hundreds of GBs or even more!).  
2. I want to preprocess the images using an image tokenizer to extract discrete codes before training, simplifying the model architecture and speeding up training.  

In this scenario, both the massive dataset and the pre-extracted image VQ codes will occupy significant disk space, leading to unacceptable loading speeds and a high likelihood of memory out-of-memory (OOM) errors. Using an **IterableDataset** is essential. We recommend referring to [TinyLlama](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md) or following our approach using the more integrated Hugging Face [Datasets](https://huggingface.co/docs/datasets/en/quickstart) library.

---



## Data Preparation

Since concatenating different datasets in Hugging Face `Datasets` requires them to have the same features, we first define a unified `jsonl` format and convert all image-text pairs and language data into this format before transforming them into Hugging Face datasets.

---

### 1. Image-Text Pair Preprocessing

**Assumption:**  
You already have image-text pairs from any source, stored in a `jsonl` file. Each line, when read, returns a dictionary containing at least the following fields:  

```python
{"image_path": "/xx/xx/xx/image1.jpg", "text_prompt": "A picture of a small cat."}
```

**Steps:**  
- Use `convert_imagepair_cc512.py` to perform center cropping on the images and extract image VQ codes.  
- Alternatively, use `convert_imagepair_multiratio.py` (WIP) to extract multi-resolution VQ codes. For multi-resolution generation, we find the closest target size from our predefined chunked sizes based on the original image resolution, center crop the image to the target size, and record the cropped size in the metadata.  

```bash
cd data_process

# Single GPU execution
python convert_imagepair_cc512.py --input_pair 'path/to/save/jsonl' --temp_path '/path/to/save/tempdata' --save_path '/path/to/save/hfdata' --vqgan_path '/path/to/vqgan_weights'

python packing_imagepairs.py --temp_path '/path/to/save/tempdata' --save_path '/path/to/save/hfdata'

# Multi-GPU or multi-machine execution
bash extract_vqcodes.sh
```

---

### 2. Language Data Preprocessing

**Example:** Using `mlfoundations/dclm-baseline-1.0` as an example, extract a portion of the `json.zst` files to the `/path/to/ori/DCLM` directory and run `convert_DCLM_data.py` to pre-save the data in a format supported by Hugging Face datasets in temp-dir and then pack them into HF datasets.

```bash
cd data_process
python convert_DCLM_data.py --input_path '/path/to/ori/DCLM' --temp_path '/path/to/save/tempdata' --save_path '/path/to/save/hfdata'
```

---

> **Note**
>
> Training solely on JourneyDB will not reproduce Liquid's T2I (Text-to-Image) performance. We used additional high-quality data and [InternVL2.0](https://huggingface.co/collections/OpenGVLab/internvl20-667d3961ab5eb12c7ed1463e) recaptioning to achieve satisfactory T2I results.
>



### 3. SFT Data for Visual Understanding (WIP)

