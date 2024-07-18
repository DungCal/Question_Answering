import numpy as np
from tqdm.auto import tqdm
import collections
import faiss

import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForQuestionAnswering # class, module rieng của hugging face lam cho 1 task
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

device='cuda' if torch.cuda.is_available() else 'cpu'


dataset_name='squad_v2'
raw_dataset=load_dataset(dataset_name,split='train')

#loc ra cau hoi tim duoc cau tra loi
raw_dataset=raw_dataset.filter(lambda x: len(x['answers']['text'])>0)

#initialize model to get vector embedding 
model_name='distilbert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name).to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]
def get_embeddings(text_list):
    encoded_input=tokenizer(text_list,padding=True,truncation=True,return_tensors='pt')
    encoded_input={k:v.to(device) for k,v in encoded_input.items()}
    model_output=model(**encoded_input)
    
    return cls_pooling(model_output)

#create question embedding column
EMBEDDING_COLUMN = 'question_embedding'
embeddings_dataset = raw_dataset.map(lambda x:{EMBEDDING_COLUMN : get_embeddings(x['question']).detach().cpu().numpy()[0]})

#create faiss vector database which store question embedding
embeddings_dataset.add_faiss_index(column=EMBEDDING_COLUMN)

#examples
input_question = 'When did Beyonce start becoming popular ?'

input_quest_embedding = get_embeddings([ input_question])
input_quest_embedding = input_quest_embedding.cpu().detach().numpy()

TOP_K = 5
scores , samples = embeddings_dataset.get_nearest_examples(EMBEDDING_COLUMN , input_quest_embedding , k= TOP_K)

for idx , score in enumerate(scores):
    print(f'Top {idx + 1}\ tScore: {score}')
    print(f'Question: {samples["question"][idx]}')
    print(f'Context: {samples["context"][idx]}')
    print()
