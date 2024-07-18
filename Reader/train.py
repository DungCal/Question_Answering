import os
import sys
import logging
from functools import partial
import numpy as np
import tqdm
import collections
import evaluate

import datasets
from datasets import DatasetDict

import transformers
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    AutoModelForQuestionAnswering,
    set_seed,
)

from huggingface_hub import login, create_repo, delete_repo

from model.metric import compute_metrics
from model.dataloader import load_dataset_from_path, preprocess_training_examples,preprocess_validation_examples
from model.model import load_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logger = logging.getLogger(__name__)


import sys

def train(model_args, data_args, training_args):

  # Setup logging
  # thiet lap logging co ban: dinh dang va cau hinh
  logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #thiet lap muc do logging
  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
    # thiet lap logging cho cac thu vien  con
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    #thiet lap cac thong tin huan luyen vao log nhu device, gpu,
    #những thông tin này có thể quan trọng và cần được chú ý
  logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
  #ghi lại các tham số huấn luyện/đánh giá (training_args) với mức độ "INFO
  logger.info(f"Training/evaluation parameters {training_args}")


  set_seed(training_args.seed)
    # login hub
  if training_args.push_to_hub:
    login(
            token=training_args.hub_token
        )

  try:
    create_repo(training_args.hub_model_id, private=False)
  except:
    pass

  #load dataset
  raw_dataset=load_dataset_from_path(data_args.save_data_dir,data_args.dataset_name,data_args.train_file,data_args.validation_file,data_args.test_file)
  raw_dataset=DatasetDict(raw_dataset)

  # num_labels,label2id,id2label=get_label_list(raw_dataset['train'].unique('label'))
  #.unique(data_args.label_column_name
  #ghi lai thong tin raw dataset
  logger.info(f'Dataset loaded: {raw_dataset}')

  #load pretrained model and tokenizer
  tokenizer=AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                                          cache_dir=model_args.cache_dir,
                                          use_fast=model_args.use_fast_tokenizer)


  config=AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                    #num_labels=num_labels,
                                    finetuning_task='question-answering',
                                    cache_dir=model_args.cache_dir)
  
  model=load_model(model_args)




  #để đảm bảo rằng một số tác vụ nhất định (như tải hoặc xử lý dữ liệu) chỉ được thực hiện bởi một tiến trình duy nhất trong trường hợp bạn đang sử dụng huấn luyện phân tán
  #desc: Tham số desc trong ví dụ này được đặt là "Running tokenizer on dataset". Khi tiến trình chính thực hiện công việc tiền xử lý dữ liệu, mô tả này sẽ xuất hiện trong
  #các log hoặc các thông báo, giúp bạn biết rằng tiến trình chính đang thực hiện việc tokenization trên tập dữ liệu.

  print(raw_dataset['validation'])

  #raise KeyboardInterrupt

  with training_args.main_process_first(desc='Dataset map preprosessing'):
    processed_train_dataset=raw_dataset['train'].map(partial(preprocess_training_examples,tokenizer=tokenizer,data_args=data_args), #create a new function where data_args, tokenizer, and label2id are already provided
                                      batched=True,
                                      load_from_cache_file=not data_args.overwrite_cache,
                                      remove_columns=raw_dataset['train'].column_names,
                                      desc='Running tokenize on dataset')

    processed_validation_dataset=raw_dataset['validation'].map(partial(preprocess_validation_examples,tokenizer=tokenizer,data_args=data_args), #create a new function where data_args, tokenizer, and label2id are already provided
                                      batched=True,
                                      load_from_cache_file=not data_args.overwrite_cache,
                                      remove_columns=raw_dataset['validation'].column_names,
                                      desc='Running tokenize on dataset')

    processed_test_dataset=raw_dataset['test'].map(partial(preprocess_validation_examples,tokenizer=tokenizer,data_args=data_args),
                                                   batched=True,
                                      load_from_cache_file=not data_args.overwrite_cache,
                                      remove_columns=raw_dataset['test'].column_names,
                                      desc='Running tokenize on dataset')

    processed_dataset=DatasetDict({
        'train':processed_train_dataset,
        'validation':processed_validation_dataset,
        'test':processed_test_dataset
    })



  if data_args.pad_to_max_length:
    data_collator=default_data_collator
  else:
    data_collator=None

  #Trainer
  print('create trainer object')

  metric=evaluate.load('squad_v2')
  n_best=20 # so luong ket qua tot nhat duoc lua chon sau khi du doan
  max_answer_length=30 # do dai toi da cho cau tra loi


  class MyTrainer(Trainer):
    def __init__(self, features,examples , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features=features
        self.examples=examples

    def compute_metrics(self, eval_pred: EvalPrediction):
        start_logits, end_logits = eval_pred.predictions
#         start_positions, end_positions = eval_pred.label_ids

        examples_to_features=collections.defaultdict(list)
        for idx,feature in enumerate(self.features):
            examples_to_features[feature['example_id']].append(idx)#dict chua key la cac id

        predicted_answers=[]
        for example in tqdm(self.examples):
            example_id=example['id']
            context=example['context']
            answers=[]
            # lap qua dac trung lien quan den vi du do
            for feature_index in examples_to_features[example_id]:
                start_logit=start_logits[feature_index]
                end_logit=end_logits[feature_index]
                offsets=self.features[feature_index]['offset_mapping']

                #lay cac chi so co top k gia tri lon nhat cho start logits vaf end logits
                start_indexes=np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
                end_indexes=np.argsort(end_logit)[-1:-n_best-1:-1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # bo qua cac cau tra loi ko nam trong ngu canh
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # bo qua cac cau tra loi co do dai > max_answer_length
                        if end_index-start_index+1>max_answer_length:
                            continue
                        #tao 1 cau tra loi moi
                        answer={
                            # theo chatgpt
                            'text': context[offsets[start_index][0] : offsets[end_index][1]],
                            'logit_score': start_logit[start_index]+end_logit[end_index]
                        }
                        answers.append(answer)

            if len(answers)>0:
                best_answer=max(answers,key=lambda x:x['logit_score'])
                answer_dict={
                    'id':example_id,
                    'prediction_text': best_answer['text'],
                    'no_answer_probability': 1-best_answer['logit_score']
                }
            else:
                answer_dict={
                    'id': example_id,
                    'prediction_text':'',
                    'no_answer_probability':1.0
                }
            predicted_answers.append(answer_dict)

        theoretical_answers=[{'id':ex['id'],'answers':ex['answers']} for ex in self.examples]

        return metric.compute(predictions=predicted_answers,references=theoretical_answers)


  trainer=MyTrainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=processed_dataset['train'],
                  eval_dataset=processed_dataset['validation'],
                  features=processed_dataset['validation'],
                  examples=raw_dataset['validation'],
                  tokenizer=tokenizer)

  print('training...')
  if training_args.do_train:
    #return processed_dataset
    train_result=trainer.train()
    metrics=train_result.metrics
    metrics['train_samples']=len(processed_dataset['train'])
    trainer.log_metrics('train',metrics)
    trainer.save_metrics('train',metrics)
    trainer.save_state()
  print('eval...')
  if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics=trainer.evaluate(eval_dataset=processed_dataset['validation'])
    metrics['eval_samples']=len(processed_dataset['validation'])
    trainer.log_metrics('eval',metrics)
    trainer.save_metrics('eval',metrics)
  print('predict...')
  if training_args.do_predict and processed_dataset['test'] is not None:
    logger.info("*** Predict ***")

    metrics=trainer.evaluate(eval_dataset=processed_dataset['test'])
    metrics['test_samples']=len(processed_dataset['test'])
    trainer.log_metrics('eval',metrics)
    trainer.save_metrics('eval',metrics)

    start_logits,end_logits=trainer.predict(processed_dataset['test'])[0]

    def return_predict_output(start_logits,end_logits,features,examples):# ham danh gia model
        #tao tu dien de anh xa moi vi du voi danh sach cac dac trung tuong ung
        examples_to_features=collections.defaultdict(list)
        for idx,feature in enumerate(features):
            examples_to_features[feature['example_id']].append(idx)#dict chua key la cac id

        predicted_answers=[]
        for example in tqdm(examples):
            example_id=example['id']
            context=example['context']
            answers=[]
            # lap qua dac trung lien quan den vi du do
            for feature_index in examples_to_features[example_id]:
                start_logit=start_logits[feature_index]
                end_logit=end_logits[feature_index]
                offsets=features[feature_index]['offset_mapping']

                #lay cac chi so co top k gia tri lon nhat cho start logits vaf end logits
                start_indexes=np.argsort(start_logit)[-1:-n_best-1:-1].tolist()
                end_indexes=np.argsort(end_logit)[-1:-n_best-1:-1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # bo qua cac cau tra loi ko nam trong ngu canh
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # bo qua cac cau tra loi co do dai > max_answer_length
                        if end_index-start_index+1>max_answer_length:
                            continue
                        #tao 1 cau tra loi moi
                        answer={
                            # theo chatgpt
                            'text': context[offsets[start_index][0] : offsets[end_index][1]],
                            'logit_score': start_logit[start_index]+end_logit[end_index]
                        }
                        answers.append(answer)

            if len(answers)>0:
                best_answer=max(answers,key=lambda x:x['logit_score'])
                answer_dict={
                    'id':example_id,
                    'prediction_text': best_answer['text'],
                    'no_answer_probability': 1-best_answer['logit_score']
                }
            else:
                answer_dict={
                    'id': example_id,
                    'prediction_text':'',
                    'no_answer_probability':1.0
                }
            predicted_answers.append(answer_dict)

        return predicted_answers

    predictions=return_predict_output(start_logits,end_logits,processed_dataset['test'],raw_dataset['test'])

    output_predict_file=os.path.join(training_args.ouput_dir,'predict_results.txt')
    with open(output_predict_file,'w') as writer:
      logger.info("***** Predict results *****")
      writer.write('index\tprediction\n')
      for index,item in enumerate(predictions):
        item=item['prediction_text']
        writer.write(f'{index}\t{item}\n')

    logger.info(f'Predicted results saved at: {output_predict_file}')

    print('het vi')
    tokenizer.save_pretrained(training_args.ouput_dir)
    if training_args.push_to_hub:
      trainer.create_model_card()
      trainer.push_to_hub()

