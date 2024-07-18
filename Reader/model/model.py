from transformers import AutoConfig,AutoModelForQuestionAnswering

def load_model(model_args):
    config=AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                                        #num_labels=num_labels,
                                        finetuning_task='question-answering'
                                        )

    model=AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path,
                                                            from_tf=bool('.ckpt' in model_args.model_name_or_path),
                                                            #num_labels=num_labels,
                                                            config=config
                                                            )
    
    return model