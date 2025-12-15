import json
import os
import pandas as pd
import time
import transformers
import sys

from datasets import Dataset, DatasetDict
from datetime import datetime
from transformers import (
    AdapterArguments,
    AdapterConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    Trainer,
)
from transformers.adapters import LoRAConfig
from transformers.trainer_callback import ProgressCallback
from transformers.utils import logging

from arguments import DataArguemnts, ModelArguments
from data import DataCollatorForEmotion
from log import Logger
from metric import Metric
from tokenizer import EmotionalTokenizer
from trainer import EmotionalAdapterTrainer, EmotionalTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_dialogsum(data_dir, data_type, test_label_tag="summary1"):
    data = json.load(open(os.path.join(data_dir, f"{data_type}.emotion.json"), "r"))
    result = {
        "fname": [],
        "summary": [],
        "dialogue": [],
        "dialogue_emotion": [],
    }
    for d in data:
        for j in d.keys():
            if d[j] == "":
                print(f"{d} has empty {j}!!")
                continue
            if j in result.keys():
                result[j].append(d[j])
            elif data_type == "test" and j == test_label_tag:
                result["summary"].append(d[j])

    return Dataset.from_dict(result)


def load_samsum(data_dir, data_type, test_label_tag="summary"):
    data = json.load(open(os.path.join(data_dir, f"{data_type}.emotion.json"), "r"))
    result = {
        "fname": [],
        "summary": [],
        "dialogue": [],
        "dialogue_emotion": [],
    }
    for d in data:
        if d["dialogue"] == "":
            print(f"{d} has empty!!")
            continue

        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
            elif j == "id":
                result["fname"].append(d[j])

    return Dataset.from_dict(result)


def load_datasets(data_dir, data_load_function, test_label_tag):
    train = data_load_function(data_dir, "train")
    validation = data_load_function(data_dir, "val")
    test = data_load_function(data_dir, "test", test_label_tag)
    return DatasetDict({"train": train, "validation": validation, "test": test})


def main():
    try:
        parser = HfArgumentParser((ModelArguments, DataArguemnts, Seq2SeqTrainingArguments, AdapterArguments))
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

        logger = Logger(model_args.enable_logging)

        logger.force_log(f"Model Args {model_args}")
        logger.log(f"Data Args {data_args}")
        logger.log(f"Training Args {training_args}")
        logger.force_log(f"Adapter Args {adapter_args}")

        transformers.trainer.logger.setLevel(logging.WARNING)

        check_emotional_consistency = model_args.check_emotional_consistency
        run_name = ""
        run_name_postfix = f" BZ{training_args.per_device_train_batch_size} {'FP16' if training_args.fp16 else 'FP32'}"

        if training_args.deepspeed is not None:
            run_name = "[Deepspeed] "

        if adapter_args.train_adapter:
            run_name = run_name + f"[Adapter_{adapter_args.adapter_config}_to{adapter_args.adapter_non_linearity} "

        if training_args.resume_from_checkpoint is not None:
            run_name = run_name + f"[Resume {training_args.resume_from_checkpoint.replace('/', '_')}] "

        if model_args.emotional_loss_weight > 0.0:
            run_name = run_name + f"[Emotional_{model_args.emotional_loss_type}/{model_args.emotional_loss_beta}/{model_args.emotional_loss_gamma}] "
            run_name_postfix = run_name_postfix + f" weight_{model_args.emotional_loss_weight}"

        # for legacy compatibility using local file
        emotion_classifier_name = "emotion-english-roberta-large"
        # emotion_classifier_name = "/mnt/hlilabshare/HLILab_Public/emotion-english-roberta-large"
        # emotion_classifier_name = "j-hartmann/emotion-english-roberta-large"
        summarizer_name = model_args.model_name

        run_name = f"{run_name}{summarizer_name.upper().split('/')[-1]}{run_name_postfix}"

        training_args.run_name = f"{datetime.today().strftime('%Y-%h-%d %H')}::{run_name}"

        training_args.output_dir = os.path.join(
            training_args.output_dir,
            run_name,
        )

        last_checkpoint = training_args.resume_from_checkpoint
        if last_checkpoint is None:
            last_checkpoint = summarizer_name
        
        if training_args.do_train:
            logger.force_log(f"Training from {last_checkpoint}")

        if "t5" in summarizer_name.lower():
            logger.force_log("T5 unable to training with FP16!!")
            training_args.fp16 = False

        # Set seed before initializing model.
        set_seed(training_args.seed)

        data_load_function = load_samsum if "SAMSum" in data_args.data_dir else load_dialogsum

        raw_datasets = load_datasets(data_args.data_dir, data_load_function, data_args.test_label_tag)

        logger.log(f"Loaded datasets is {raw_datasets}")

        summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name if last_checkpoint is None else last_checkpoint)

        if adapter_args.train_adapter:
            task_name = "summarization_adapter"

            if adapter_args.adapter_config == "lora":
                adapter_config = LoRAConfig(r=8, alpha=32, dropout=0.1, attn_matrices=adapter_args.adapter_non_linearity.split("#"))
            else:
                adapter_config = AdapterConfig.load(
                    adapter_args.adapter_config,
                    non_linearity=adapter_args.adapter_non_linearity,
                    reduction_factor=adapter_args.adapter_reduction_factor,
                )

            logger.force_log(f"Adapter is {adapter_config}")

            if adapter_args.load_adapter:
                summarization_model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            else:
                summarization_model.add_adapter(task_name, config=adapter_config)
                summarization_model.train_adapter(task_name)
            summarization_model.set_active_adapters(task_name)

        summarization_tokenizer = AutoTokenizer.from_pretrained(summarizer_name if last_checkpoint is None else last_checkpoint)

        emotion_classifier = Trainer(model=AutoModelForSequenceClassification.from_pretrained(emotion_classifier_name)) if check_emotional_consistency else None
        emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_classifier_name) if check_emotional_consistency else None

        tokenizer = EmotionalTokenizer(summarization_tokenizer, emotion_tokenizer)
        metric = Metric(tokenizer, emotion_classifier, raw_datasets)

        if check_emotional_consistency:
            emotion_classifier.remove_callback(ProgressCallback)
            emotion_label2id = emotion_classifier.model.config.label2id
            num_emotion_labels = len(emotion_label2id)
            num_of_emotions = pd.Series(raw_datasets["train"]["dialogue_emotion"]).map(emotion_label2id).value_counts()
            logger.force_log(f"Num of Emotions: {num_of_emotions}")

        logger.log(f"Summarization Model information \n{summarization_model}")

        def tokenize_function(examples):
            def make_one_hot(emotion):
                one_hot_list = [0] * num_emotion_labels
                one_hot_list[emotion_label2id[emotion]] = 1
                return one_hot_list

            def filter_special_token(sentence):
                return sentence.replace(":", "") if "t5" in summarizer_name.lower() else sentence

            prefix = "summarize: " if "t5" in summarizer_name.lower() else ""

            inputs = [prefix + filter_special_token(doc) for doc in examples["dialogue"]]
            model_inputs = summarization_tokenizer(inputs, max_length=data_args.max_source_length, truncation=True)

            # Setup the tokenizer for targets
            with summarization_tokenizer.as_target_tokenizer():
                labels = summarization_tokenizer(
                    examples["summary"],
                    max_length=data_args.max_target_length,
                    truncation=True,
                )

            model_inputs["labels"] = labels["input_ids"]

            if check_emotional_consistency and "dialogue_emotion" in examples.keys():
                model_inputs["emotion_labels"] = [make_one_hot(emotion) for emotion in examples["dialogue_emotion"]]

            return model_inputs

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        logger.log(tokenized_datasets)

        data_collator = DataCollatorForEmotion(tokenizer=summarization_tokenizer, model=summarization_model)

        trainer_class = EmotionalAdapterTrainer if adapter_args.train_adapter else EmotionalTrainer

        logger.force_log(f"Trainer is {trainer_class}")

        early_stopping_callback = None

        trainer = trainer_class(
            model=summarization_model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=summarization_tokenizer,
            compute_metrics=metric.compute_metrics,
            emotion_classifier=emotion_classifier,
            emotional_tokenizer=tokenizer,
            emotional_loss_weight=model_args.emotional_loss_weight,
            emotional_loss_params={"loss_type": model_args.emotional_loss_type, "beta": model_args.emotional_loss_beta, "gamma": model_args.emotional_loss_gamma},
            callbacks=early_stopping_callback,
        )

        def save_output(output, prefix="test"):
            predictions, _, result = output
            logger.force_log(result)

            decoded_preds = tokenizer.decode_summary(predictions)
            decoded_preds = tokenizer.decode_sentence(decoded_preds)

            # output summaries save to file
            with open(os.path.join(training_args.output_dir, f"{prefix}_result.txt"), "w") as f:
                f.write(f"Model Args {model_args}\n")
                f.write(f"Data Args {data_args}\n")
                f.write(f"Training Args {training_args.to_json_string()}\n")
                f.write(f"Adapter Args {adapter_args}\n")
                f.write(f"Final Results from epoch #{trainer.state.epoch} and step #{trainer.state.global_step}\n")
                f.write(f"{result}\n")
                for i in decoded_preds:
                    f.write(i.replace("\n", "") + "\n")

        with open(os.path.join(training_args.output_dir, "rerun.sh"), "w") as f:
            f.write("#!/bin/bash\npython ")
            f.write(" ".join([f"\\\n{arg}" if "--" in arg else arg for arg in sys.argv]))

        if training_args.do_train:
            if adapter_args.adapter_config == "lora":
                summarization_model.reset_adapter()
            trainer.train()
        else:
            logger.force_log("Do prediction from last checkpoint")

        start = time.time()

        if adapter_args.adapter_config == "lora":
            summarization_model.merge_adapter(task_name)

        test_output = trainer.predict(
            tokenized_datasets["test"],
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        logger.force_log("Execution time : {:.4f} sec".format(time.time() - start))
        save_output(test_output)
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
