from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from torch import nn
import torch
import shutil 
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def main(args):

    LLM_pth = args.model_path
    vq_codebook_size = args.num_add_token

    llm = AutoModelForCausalLM.from_pretrained(LLM_pth)
    tokenizer = AutoTokenizer.from_pretrained(LLM_pth )

    llm.generation_config.do_sample = True
    ori_vocabe_size = llm.config.vocab_size
    
    llm.resize_token_embeddings(new_num_tokens=ori_vocabe_size+vq_codebook_size, mean_resizing=True )
    
    
    # llm.config.vocab_size = ori_vocabe_size+vq_codebook_size
    # NC = llm.config.hidden_size
    # LMhead_weights = llm.lm_head.state_dict()['weight'].clone()
    # llm.lm_head = nn.Linear(NC, ori_vocabe_size+vq_codebook_size, bias=False)
    # torch.nn.init.xavier_uniform_(llm.lm_head.weight)
    # new_weights  = llm.lm_head.state_dict()
    # new_weights['weight'][:ori_vocabe_size] = LMhead_weights[:ori_vocabe_size]
    # llm.lm_head.load_state_dict(new_weights)
    # old_embedding_tokens  = llm.model.embed_tokens.weight.detach().clone()
    # llm.model.embed_tokens = nn.Embedding(ori_vocabe_size+vq_codebook_size, NC, 0)
    # torch.nn.init.xavier_uniform_( llm.model.embed_tokens.weight)
    # llm.model.embed_tokens.weight.requires_grad=False
    # llm.model.embed_tokens.weight[:ori_vocabe_size] = old_embedding_tokens[:ori_vocabe_size]
    # llm.model.embed_tokens.weight.requires_grad=True
    
    
    
    llm.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='/path/to/gemma-7b',help='path to ori model')
    parser.add_argument('--save_path', type=str, default='/path/to/save/gemma-7b-addtoken',help='path to save new model')
    parser.add_argument('--num_add_token', type=int, default=8192,help='num of tokens need to be added')
    args = parser.parse_args()
    main(args)




