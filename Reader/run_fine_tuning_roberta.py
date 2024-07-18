import numpy as np
from tqdm.auto import tqdm
import collections
import datasets
import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering # class, module rieng của hugging face lam cho 1 task
from transformers import TrainingArguments
from transformers import Trainer
import evaluate
import datasets
import transformers

import os
import json
import logging
from functools import partial
from typing import Union, List, Optional
from dataclasses import field,dataclass
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, load_dataset

from transformers import HfArgumentParser,TrainingArguments,AutoConfig,DataCollatorWithPadding,EvalPrediction,Trainer, default_data_collator,set_seed

from transformers.trainer_utils import IntervalStrategy, HubStrategy
from huggingface_hub import login,create_repo

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)



@dataclass
class DataTrainingArguments:
  dataset_name: Optional[str] = field(
      default="squad_v2", metadata={"help": "The name of the dataset to use (via the datasets library)."}
  )
  save_data_dir: Optional[str] = field(
      default="data", metadata={"help": "A folder save the dataset."}
  )
  train_file: Optional[str] = field(
      default="train.jsonl", metadata={"help": "A csv or a json file containing the training data."}
  )
  validation_file: Optional[str] = field(
      default="validation.jsonl", metadata={"help": "A csv or a json file containing the validation data."}
  )
  test_file: Optional[str] = field(
      default="test.jsonl", metadata={"help": "A csv or a json file containing the test data."}
  )
  text_column_name: Optional[str] = field(
      default="review_body",
      metadata={
          "help": (
              "The name of the text column in the input dataset or a CSV/JSON file."
              'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
          )
      },
  )
  text_column_delimiter: Optional[str] = field(
      default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
  )
  label_column_name: Optional[str] = field(
      default="feeling",
      metadata={
          "help": (
              "The name of the label column in the input dataset or a CSV/JSON file."
              'If not specified, will use the "label" column for single/multi-label classifcation task'
          )
      },
  )
  max_seq_length: int = field(
      default=384,
      metadata={
          "help": (
              "The maximum total input sequence length after tokenization. Sequences longer "
              "than this will be truncated, sequences shorter will be padded."
          )
      },

  )
  stride: int = field(
      default=128,
      metadata={
          "help": (
              "The authorized overlap between two consecutive"
              " input sequences after tokenization."

  )
      },
  )

  overwrite_cache: bool = field(
      default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
  )
  pad_to_max_length: bool = field(
      default=True,
      metadata={
          "help": (
              "Whether to pad all samples to `max_seq_length`. "
              "If False, will pad the samples dynamically when batching to the maximum length in the batch."
          )
      },
  )
  shuffle_train_dataset: bool = field(
      default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
  )
  shuffle_seed: int = field(
      default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
  )
  max_train_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
          )
      },
  )
  max_eval_samples: Optional[int] = field(
      default=None,
      metadata={
          "help": (
              "For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set."
          )
      },
  )
  metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})

  def __post_init__(self):
      save_data_dir = os.path.join(self.save_data_dir, self.dataset_name)
      if not os.path.exists(save_data_dir):
          #print('ditmemay')
          os.makedirs(save_data_dir, exist_ok=True)
          load_dataset_from_datahub(
              self.dataset_name,
              save_data_dir
          )



@dataclass
class ModelArguments:
  model_name_or_path: str = field(
      default="deepset/roberta-base-squad2", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
      default=None,
      metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
  )
  use_fast_tokenizer: bool = field(
      default=True,
      metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
  )



@dataclass
class TrainingArgumentsCustom(TrainingArguments):
  output_dir: str = field(
      default="outputs",
      metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
  )
  per_device_train_batch_size: int = field(
      default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for training."}
  )
  per_device_eval_batch_size: int = field(
      default=32, metadata={"help": "Batch size per GPU/TPU/MPS/NPU core/CPU for evaluation."}
  )
  do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
  do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
  do_predict: bool = field(default=True, metadata={"help": "Whether to run predictions on the test set."})
  num_train_epochs: float = field(
      default=5.0, metadata={"help": "Total number of training epochs to perform."}
  )
  logging_dir: Optional[str] = field(
      default="logs", metadata={"help": "Tensorboard log dir."}
  )
  logging_strategy: Union[IntervalStrategy, str] = field(
      default="steps",
      metadata={"help": "The logging strategy to use."},
  )
  logging_steps: float = field(
      default=500,
      metadata={
          "help": (
              "Log every X updates steps. Should be an integer or a float in range `[0,1)`."
              "If smaller than 1, will be interpreted as ratio of total training steps."
          )
      },
  )
  evaluation_strategy: Union[IntervalStrategy, str] = field(
      default="epoch",
      metadata={"help": "The evaluation strategy to use."},
  )
  save_strategy: Union[IntervalStrategy, str] = field(
      default="epoch", metadata={"help": "The checkpoint save strategy to use."},
  )
  save_steps: float = field(
      default=100,
      metadata={
          "help": (
              "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
              "If smaller than 1, will be interpreted as ratio of total training steps."
          )
      },
  )
  save_total_limit: Optional[int] = field(
      default=2, metadata={"help": ("If a value is passed, will limit the total amount of checkpoints.")},
  )
  load_best_model_at_end: Optional[bool] = field(
      default=True,
      metadata={
          "help": (
              "Whether or not to load the best model found during training at the end of training. When this option"
              " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
          )
      },
  )
  metric_for_best_model: Optional[str] = field(
      default="squad_v2", metadata={"help": "The metric to use to compare two different models."}
  )
  report_to: Optional[List[str]] = field(
      default="tensorboard", metadata={"help": "The list of integrations to report the results and logs to."}
  )
  optim: str = field(
      default="adamw_torch",
      metadata={"help": "The optimizer to use."},
  )
  bf16: bool = field(
      default=False,
      metadata={
          "help": (
              "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
              " architecture or using CPU (use_cpu). This is an experimental API and it may change."
          )
      },
  )
  fp16: bool = field(
      default=False,
      metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
  )
  push_to_hub: bool = field(
      default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
  )
  hub_strategy: Union[HubStrategy, str] = field(
      default="every_save",
      metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
  )
  hub_model_id: Optional[str] = field(
      default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
  )
  hub_token: Optional[str] = field(
      default=None, metadata={"help": "The token to use to push to the Model Hub."}
  )



from datasets import load_dataset
dataset = load_dataset("squad_v2")


def load_dataset_from_datahub(dataset_name,save_data_dir):
  raw_dataset=load_dataset(dataset_name)

  # Chia tập validation thành hai phần với tỉ lệ 2:1
  split_dataset = dataset['validation'].train_test_split(test_size=1/3)

# Đổi tên các tập hợp
  split_dataset = DatasetDict({
    'validation': split_dataset['train'],
    'test': split_dataset['test']
})

# Cập nhật lại DatasetDict gốc
  final_dataset = DatasetDict({
    'train': dataset['train'],
    'validation': split_dataset['validation'],
    'test': split_dataset['test']
})

  for dataset_type in final_dataset:
    examples=final_dataset[dataset_type]
    sentences=[]

    id=examples['id']
    title=examples['title']
    context=examples['context']
    question=examples['question']
    answer=examples['answers']

    save_data_file=os.path.join(save_data_dir,f'{dataset_type}.jsonl')
    print(f'Write into...{save_data_file}')
    with open(save_data_file,'w') as f:
      for id,title,context,question,answer in zip(id,title,context,question,answer):
        data={'id':id,'title':title,'context':context,'question':question,'answers':answer}
        print(json.dumps(data,ensure_ascii=False),file=f)


def load_dataset_from_path(save_data_dir, dataset_name, train_file, validation_file, test_file):
    save_data_dir = os.path.join(save_data_dir, dataset_name)
    print(f"Load data from: {save_data_dir}")

    train_file_path = os.path.join(save_data_dir, train_file)
    train_dataset = load_dataset('json', data_files=train_file_path)

    validation_file_path = os.path.join(save_data_dir, validation_file)
    validation_dataset = load_dataset('json', data_files=validation_file_path)

    test_file_path = os.path.join(save_data_dir, test_file)
    test_dataset = load_dataset('json', data_files=test_file_path)

    return {
        'train': train_dataset['train'],
        'validation': validation_dataset['train'],
        'test': test_dataset['train']
    }




def preprocess_training_examples(examples,tokenizer,data_args):
    #bo khoang trang du thua
    questions=[q.strip() for q in examples['question']]

    #tokenize
    inputs=tokenizer(questions,examples['context'],max_length=data_args.max_seq_length,truncation='only_second',
                 stride=data_args.stride,
                 return_overflowing_tokens=True,
                 return_offsets_mapping=True,
                 padding='max_length')

    #lay offset_mapping ra khoi input
    offset_mapping=inputs.pop('offset_mapping')

    #lay sample_mapping ra khoi input
    sample_mapping=inputs.pop('overflow_to_sample_mapping')

    #lay ra answer
    answers=examples['answers']

    #khoi tao danh sach vi tri bat dau va ket thuc cau tra loi
    start_positions=[]
    end_positions=[]

    for i,offset in enumerate(offset_mapping):
        #xac dinh index cua sample lien quan den offset hien tai
        sample_idx=sample_mapping[i]

        #input segment embedding
        sequence_ids=inputs.sequence_ids(i)

        # dua ra start va end index cua context trong input moi
        idx=0
        while sequence_ids[idx]!=1:
            idx+=1
        context_start=idx
        while sequence_ids[idx]==1:
            idx+=1
        context_end=idx-1

        #trich xuat thong tin ve cau tra loi
        answer=answers[sample_idx]

        #xet xem dap an co phai cau tra loi rong hay khong
        if len(answer['text'])==0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # xac dinh vi tri bat dau vaf ket thuc cua answer trong context
            start_char=answer['answer_start'][0]
            end_char=answer['answer_start'][0]+len(answer['text'][0])

            #so sanh xem vi tri trong offset dich nguoc ve input context xem co trung voi vi tri cua word do trong context hay ko
            if offset[context_start][0]>start_char or offset[context_end][1]<end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                #neu khong, gan vi tri bat dau va ket thuc dua tren vi tri cua cac ma thong tin
                idx=context_start
                while idx<=context_end and offset[idx][0] <=start_char: #offset[idx][0]: vi tri bat dau cua word trong context
                    # lap cho den khi tim duoc vi tri bat dau cua answer
                    idx+=1
                start_positions.append(idx-1)

                idx=context_end
                while idx>=context_start and offset[idx][1]>=end_char:#offset[idx][1]: vi tri ket thuc cua word trong context
                    idx-=1
                end_positions.append(idx+1)

    inputs['start_positions']=start_positions
    inputs['end_positions']=end_positions

    return inputs


def preprocess_validation_examples(examples,tokenizer,data_args):
    #loai bo khoang trang
    questions=[q.strip() for q in examples['question']]

    inputs=tokenizer(questions,examples['context'],
                     max_length=data_args.max_seq_length,
                     truncation='only_second',
                     stride=data_args.stride,
                     return_offsets_mapping=True,
                     return_overflowing_tokens=True,
                     padding='max_length')
    #lay anh xa de anh xa lai vi du tham chieu cho tung dong trong inputs
    sample_map=inputs.pop('overflow_to_sample_mapping')
    example_ids=[]
    #Xac dinh vi du tham chieu cho moi dong dau vao vaf dieu chinh anh xa offset
    for i in range(len(inputs['input_ids'])):
        sample_idx=sample_map[i]
        example_ids.append(examples['id'][sample_idx])

        sequence_ids=inputs.sequence_ids(i)
        offset=inputs['offset_mapping'][i]
        #loai bo cac offset  ko phu hop voi sequence_ids
        inputs['offset_mapping'][i]=[o if sequence_ids[k]==1 else None for k,o in enumerate(offset)]
    #them thong tin vi du tham chieu dau vao
    inputs['example_id']=example_ids
    return inputs




metric=evaluate.load('squad_v2')
n_best=20 # so luong ket qua tot nhat duoc lua chon sau khi du doan
max_answer_length=30 # do dai toi da cho cau tra loi


def compute_metrics(start_logits,end_logits,features,examples):# ham danh gia model

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

    theoretical_answers=[{'id':ex['id'],'answers':ex['answers']} for ex in examples]

    return metric.compute(predictions=predicted_answers,references=theoretical_answers)



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

  model=AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path,
                                                           from_tf=bool('.ckpt' in model_args.model_name_or_path),
                                                           #num_labels=num_labels,
                                                           config=config,
                                                           cache_dir=model_args.cache_dir)
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

        theoretical_answers=[{'id':ex['id'],'answers':ex['answers']} for ex in examples]

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




def main():
  parser=HfArgumentParser((ModelArguments,DataTrainingArguments,TrainingArgumentsCustom))

  def filter_args(args):
    # Chỉ giữ lại các đối số có tiền tố là '--'
    return [arg for arg in args if arg.startswith("--")]

# Nếu không có đối số dòng lệnh, sử dụng giá trị mặc định
  if len(sys.argv) == 1:
      model_args, data_args,training_args = parser.parse_args_into_dataclasses(args=[])
  else:
      filtered_args = filter_args(sys.argv)
      model_args, data_args ,training_args = parser.parse_args_into_dataclasses(filtered_args)

  #return model_args,data_args,training_args

  os.makedirs(training_args.output_dir,exist_ok=True)

  save_model_path=model_args.model_name_or_path.split('/')[-1]+'-'+data_args.dataset_name.replace('_','-')
  training_args.hub_model_id=model_args.model_name_or_path.split('/')[-1]+'-'+data_args.dataset_name.replace('_','-')
  if training_args.fp16:
    training_args.model_hub_id=training_args.model_hub_id+'-fp16'
    save_model_path=save_model_path+'-fp16'
  elif training_args.bf16:
    training_args.model_hub_id=training_args.model_hub_id+'-bf16'
    save_model_path=save_model_path+'-bf16'

  training_args.output_dir=os.path.join(training_args.output_dir,save_model_path)
  if not os.path.exists(training_args.output_dir):
    os.makedirs(training_args.output_dir)
  training_args.logging_dir=os.path.join(training_args.output_dir,training_args.output_dir)

  train(model_args,data_args,training_args)

if __name__=="__main__":
  main()
