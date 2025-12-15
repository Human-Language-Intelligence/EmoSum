import json
import os
import telegram
import time
import transformers

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
from transformers.trainer_callback import ProgressCallback
from transformers.utils import logging

from arguments import DataArguemnts, ModelArguments
from data import DataCollatorForEmotion
from log import Logger
from metric import Metric
from tokenizer import EmotionalTokenizer
from trainer import EmotionalAdapterTrainer, EmotionalTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
TOKEN = "5780435181:AAHyonV2dvAYVfk2hUwQDD_EnAO4_q8ETGc"
CHAT_ID = "1190137386"


def load_dataset(data_dir, data_type, summary_tag="summary"):
    data = json.load(open(os.path.join(data_dir, f"{data_type}.speaker.json"), "r"))
    result = {
        "fname": [],
        "summary_labels": [],
        "dialogue": [],
        "dialogue_emotion": [],
    }
    for d in data:
        if summary_tag not in d.keys():
            continue

        for j in d.keys():
            if j in result.keys():
                result[j].append(d[j])
            elif j == summary_tag:
                result["summary_labels"].append(d[j])
            elif j == "dialog_erc_large":
                result["dialogue_emotion"].append(d[j])

    return Dataset.from_dict(result)


def load_datasets(data_dir, target_speaker=None):
    summary_tag = "summary" if target_speaker is None else f"{target_speaker}_summary"

    train = load_dataset(data_dir, "train", summary_tag)
    validation = load_dataset(data_dir, "val", summary_tag)

    summary_tag = "summary1" if target_speaker is None else f"{target_speaker}_summary"
    test = load_dataset(data_dir, "test", summary_tag)
    return DatasetDict({"train": train, "validation": validation, "test": test})


def main():
    bot = telegram.Bot(TOKEN)
    parser = HfArgumentParser(
        (ModelArguments, DataArguemnts, Seq2SeqTrainingArguments, AdapterArguments)
    )
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
    run_name = (
        f"[{model_args.target_speaker} Aware] "
        if model_args.speaker_aware_summarization
        else ""
    )
    run_name_postfix = f" ({training_args.num_train_epochs})"

    if training_args.deepspeed is not None:
        run_name = "[Deepspeed] "

    if adapter_args.train_adapter:
        run_name = run_name + "[Adapter] "

    if training_args.resume_from_checkpoint is not None:
        run_name = (
            run_name
            + f"[Resume {training_args.resume_from_checkpoint.replace('/', '_')}] "
        )

    if model_args.emotional_loss_weight > 0.0:
        run_name = run_name + "[Emotional] "
        run_name_postfix = (
            run_name_postfix + f" weight_{model_args.emotional_loss_weight}"
        )

    emotion_classifier_name = "j-hartmann/emotion-english-roberta-large"
    summarizer_name = model_args.model_name

    run_name = f"{run_name}{summarizer_name.upper().split('/')[-1]}{run_name_postfix}"

    training_args.run_name = f"{datetime.today().strftime('%Y-%h-%d %H')}::{run_name}"

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        run_name,
    )

    last_checkpoint = training_args.resume_from_checkpoint
    logger.force_log(f"Training from {last_checkpoint}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_datasets(data_args.data_dir, model_args.target_speaker)

    logger.log(f"Loaded datasets is {raw_datasets}")

    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(
        summarizer_name if last_checkpoint is None else last_checkpoint
    )

    if adapter_args.train_adapter:
        task_name = "emotional_summarization"

        adapter_config = AdapterConfig.load(
            adapter_args.adapter_config,
            non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=adapter_args.adapter_reduction_factor,
        )
        logger.force_log(f"Adapter is {adapter_config}")
        summarization_model.add_adapter(task_name, config=adapter_config)
        summarization_model.train_adapter(task_name)
        summarization_model.set_active_adapters(task_name)

    # summarization_model.set_input_embeddings(summarization_model.config, summarization_model.get_input_embeddings())
    summarization_tokenizer = AutoTokenizer.from_pretrained(
        summarizer_name if last_checkpoint is None else last_checkpoint
    )

    emotion_classifier = (
        Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(
                emotion_classifier_name
            )
        )
        if check_emotional_consistency
        else None
    )
    emotion_tokenizer = (
        AutoTokenizer.from_pretrained(emotion_classifier_name)
        if check_emotional_consistency
        else None
    )

    tokenizer = EmotionalTokenizer(summarization_tokenizer, emotion_tokenizer)
    metric = Metric(tokenizer, emotion_classifier, raw_datasets, "summary_labels")

    if check_emotional_consistency:
        emotion_classifier.remove_callback(ProgressCallback)
        emotion_label2id = emotion_classifier.model.config.label2id
        num_emotion_labels = len(emotion_label2id)

    logger.log(f"Summarization Model information \n{summarization_model}")

    def tokenize_function(examples):
        def make_one_hot(emotion):
            one_hot_list = [0] * num_emotion_labels
            one_hot_list[emotion_label2id[emotion]] = 1
            return one_hot_list

        prefix = "summarize: " if "t5" in summarizer_name.lower() else ""

        inputs = [prefix + doc for doc in examples["dialogue"]]
        model_inputs = summarization_tokenizer(
            inputs, max_length=data_args.max_source_length, truncation=True
        )

        # Setup the tokenizer for targets
        with summarization_tokenizer.as_target_tokenizer():
            labels = summarization_tokenizer(
                examples["summary_labels"],
                max_length=data_args.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]

        if check_emotional_consistency:
            model_inputs["emotion_labels"] = [
                make_one_hot(emotion) for emotion in examples["dialogue_emotion"]
            ]

        return model_inputs

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    logger.log(tokenized_datasets)

    data_collator = DataCollatorForEmotion(
        tokenizer=summarization_tokenizer, model=summarization_model
    )

    trainer_class = (
        EmotionalAdapterTrainer if adapter_args.train_adapter else EmotionalTrainer
    )

    logger.force_log(f"Trainer is {trainer_class}")

    early_stopping_callback = None
    # (
    #     [EarlyStoppingCallback(early_stopping_patience=5)]
    #     if training_args.eval_steps is not None
    #     else None
    # )

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
        callbacks=early_stopping_callback,
    )

    def save_output(output, prefix="test"):
        predictions, _, result = output
        logger.force_log(result)

        decoded_preds = tokenizer.decode_summary(predictions)
        decoded_preds = tokenizer.decode_sentence(decoded_preds, " ")

        # output summaries save to file
        with open(
            os.path.join(training_args.output_dir, f"{prefix}_result.txt"), "w"
        ) as f:
            f.write(f"Model Args {model_args}\n")
            f.write(f"Data Args {data_args}\n")
            f.write(f"Training Args {training_args.to_json_string()}\n")
            f.write(f"Adapter Args {adapter_args}\n")
            f.write(f"Final Results from {trainer.state.best_model_checkpoint}\n")
            f.write(f"{result}\n")
            for i in decoded_preds:
                f.write(i.replace("\n", "") + "\n")

    bot.sendMessage(
        CHAT_ID,
        f"Start Training on GPU #{model_args.gpu_index}\n{training_args.run_name}",
    )

    if training_args.do_train or adapter_args.train_adapter:
        trainer.train()
    else:
        logger.force_log("Do prediction from last checkpoint")
        # train_output = trainer.predict(
        #     tokenized_datasets["train"],
        #     metric_key_prefix="train",
        #     max_length=training_args.generation_max_length,
        #     num_beams=training_args.generation_num_beams,
        # )
        # save_output(train_output, "train")
        # val_output = trainer.predict(
        #     tokenized_datasets["validation"],
        #     metric_key_prefix="val",
        #     max_length=training_args.generation_max_length,
        #     num_beams=training_args.generation_num_beams,
        # )
        # save_output(val_output, "val")

    start = time.time()
    test_output = trainer.predict(
        tokenized_datasets["test"],
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    logger.force_log("Execution time : {:.4f} sec".format(time.time() - start))
    save_output(test_output)
    bot.sendMessage(
        CHAT_ID,
        f"End of Training from GPU #{model_args.gpu_index}\n{training_args.run_name}\nResult::\n{test_output.metrics}",
    )


if __name__ == "__main__":
    main()
