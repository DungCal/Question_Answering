# Question Answering using Fine-Tuning RoBERTa and FAISS on squad_v2

## Dependencies
- Python 3.10
- [PyTorch](https://github.com/pytorch/pytorch) 2.0 +
  ```
  pip install -r requirements.txt
  ```
## Dataset
- Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

  [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)

## Fine-Tuning roBERTa
  ### Training
  ```
    python Reader/run_fine_tuning_roberta.py \
        --dataset_name rajpurkar/squad_v2 \
        --model_name_or_path deepset/roberta-base-squad2 \
        --do_train True \
        --do_eval True \
        --do_predict True
  ```
## Predict
  Load model from huggingface repository
  ```
    from transformers import pipeline
    
    PIPELINE_NAME = 'question-answering'
    MODEL_NAME = 'thangduong0509/distilbert-finetuned-squadv2'
    pipe = pipeline(PIPELINE_NAME,model=MODEL_NAME)
    
    
    input_question = 'When did Beyonce start becoming popular ?'
    
    input_quest_embedding = get_embeddings([ input_question])
    input_quest_embedding = input_quest_embedding.cpu().detach().numpy()
    
    TOP_K = 5
    scores , samples = embeddings_dataset.get_nearest_examples(EMBEDDING_COLUMN , input_quest_embedding , k= TOP_K)
    
    for idx , score in enumerate(scores):
        question=samples['question'][idx]
        context=samples['context'][idx]
        answer=pipe(question=question,context=context)
        
        print(f'Top {idx + 1}\ tScore: {score}')
        print(f'Context: {context}')
        print(f'Answer: {answer}')
    
    ```
