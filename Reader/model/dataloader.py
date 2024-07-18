import os
from transformers import load_dataset


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