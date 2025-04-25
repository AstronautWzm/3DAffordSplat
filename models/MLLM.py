import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Attention import *
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,RobertaTokenizer,RobertaForCausalLM
from transformers import BartTokenizer, BartForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 语言模型
'''
    Input:
        __init__:
             text_encoder_type: str
             freeze_text_encoder: bool
             embedding_dim: int
        forward:
            Questions: [Batch_size, str]
            Answers: [Batch_size, str]
            device: str
    Output:
        forward:
            output_dict={
                'aff_hidden_states':aff_hidden_states：list,[Batch_size,1,embedding_dim]
                'text_loss': text_loss: tensor
                'predicted_text': predicted_text：list,[Batch_size,str]
            }
'''
class MLLM(nn.Module):
    def __init__(self, text_encoder_type="roberta-base",freeze_model = False,embedding_dim=256):
        super(MLLM, self).__init__()
        update_path = "./checkpoints/" + text_encoder_type.split("-")[0]+"-updated/"
        load_path = update_path

        if text_encoder_type == "roberta-base":
            if not os.path.exists(load_path):
                load_path = "./checkpoints/roberta-base/"
            self.tokenizer = RobertaTokenizer.from_pretrained(load_path, local_files_only=True)
            self.model = RobertaForCausalLM.from_pretrained(load_path, local_files_only=True)
            print(f"Language model {text_encoder_type} initialization succeed.")
        elif text_encoder_type == "bart-base":
            if not os.path.exists(load_path):
                load_path = "./checkpoints/bart-base/"
            self.tokenizer = BartTokenizer.from_pretrained(load_path, local_files_only=True)
            self.model = BartForCausalLM.from_pretrained(load_path, local_files_only=True)
            print(f"Language model {text_encoder_type} initialization succeed.")
        elif text_encoder_type == "gpt2":
            if not os.path.exists(load_path):
                load_path = "./checkpoints/gpt2/"
            self.tokenizer = GPT2Tokenizer.from_pretrained(load_path, local_files_only=True)
            self.model = GPT2LMHeadModel.from_pretrained(load_path, local_files_only=True)
            print(f"Language model {text_encoder_type} initialization succeed.")
        else:
            try:
                if not os.path.exists(load_path):
                    load_path = "./checkpoints/"+text_encoder_type
                self.model = AutoModel.from_pretrained(text_encoder_type)
                self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
                print(f"Language model {text_encoder_type} initialization succeed.")
            except Exception as e:
                print(e)

        if self.model is None:
            raise ValueError("Model initialization failed. Please check the model path or model name.")

        self.freeze_model = freeze_model
        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)

        new_token = "<Aff>"
        if new_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([new_token])
            self.model.resize_token_embeddings(len(self.tokenizer))
            # 保存更新后的 tokenizer 和模型配置
            self.tokenizer.save_pretrained(update_path)
            self.model.save_pretrained(update_path)
        else:
            print(f"The token '{new_token}' is already in the vocabulary. No need to add again.")

        self.aff_id = self.tokenizer.convert_tokens_to_ids([new_token])[0]

        self.embedding_dim = embedding_dim
        self.adapter = nn.Sequential(nn.Linear(self.model.config.hidden_size, embedding_dim,bias=True),
                                     nn.LayerNorm(embedding_dim, eps=1e-12))

    def forward(self, Questions,Answers,device):
        batch_size=len(Questions)

        tokenized_Q = self.tokenizer.batch_encode_plus(Questions, padding='max_length', truncation=True,max_length=40,return_tensors='pt').to(device)
        tokenized_A = self.tokenizer.batch_encode_plus(Answers, padding='max_length', truncation=True, max_length=40,return_tensors='pt').to(device)

        outputs = self.model(input_ids=tokenized_Q['input_ids'],attention_mask=tokenized_Q['attention_mask'], labels=tokenized_A['input_ids'],output_hidden_states=True,output_attentions=True) # dict

        text_loss = outputs.loss # tensor
        hidden_states = outputs.hidden_states[-1] # (batch_size, max_length, hidden_size=768)

        logits = outputs.logits # (batch_size, max_length, vocab_size)
        predicted_ids = torch.argmax(logits, dim=-1) # (batch_size, max_length)
        predicted_text = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True) # (batch_size,str)

        aff_hidden_states = []
        for i in range(batch_size):
            aff_indices = (predicted_ids[i] == self.aff_id).nonzero(as_tuple=True)[0].tolist()
            if aff_indices == []:
                aff_hidden_state = hidden_states[i, :, :].mean(0, keepdim=True) # [1,hidden_size=768] # 如果句子没有<Aff>，取所有hidden_states的平均值
            else:
                aff_hidden_state=hidden_states[i,aff_indices,:].mean(0, keepdim=True) # [1,hidden_size=768] # 如果句子有多个<Aff>，取平均值
            aff_hidden_state = self.adapter(aff_hidden_state)
            aff_hidden_states.append(aff_hidden_state)

        aff_hidden_states = torch.cat(aff_hidden_states, dim=0) # [batch_size,1,hidden_size]
        output_dict={
            'aff_hidden_states':aff_hidden_states,
            'text_loss': text_loss,
            'predicted_text': predicted_text
        }
        return output_dict

