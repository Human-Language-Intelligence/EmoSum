import numpy as np
import pandas as pd

from bert_score import score
from data import SimpleDataset
from datasets import load_metric
from sklearn.metrics import top_k_accuracy_score


class Metric:
    def __init__(self, tokenizer, emotion_classifier, datasets, summary_tag="summary"):
        self.tokenizer = tokenizer
        self.emotion_classifier = emotion_classifier

        self.train_dataset = datasets["train"]
        self.validation_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]
        self.summary_tag = summary_tag

        self.threshold = 0.5

        self.metric = load_metric("rouge")

    def normalize(self, value):
        return np.maximum(0, value)

    def calculate_emotional_consistency(self, generated_summary, labels):
        assert len(generated_summary) == len(labels)

        dialogue_emotions = pd.Series(labels).map(self.emotion_classifier.model.config.label2id)

        tokenized_output = self.tokenizer.encode_emotion(generated_summary)
        emotion_predict_result = self.emotion_classifier.predict(SimpleDataset(tokenized_output))
        emotion_score = np.exp(emotion_predict_result[0]) / np.exp(emotion_predict_result[0]).sum(-1, keepdims=True)

        emotional_consistency_score_0_5 = np.mean([self.normalize(emotion_score[i][emotion] - self.threshold) for i, emotion in enumerate(dialogue_emotions)])
        emotional_consistency_score_0_2 = np.mean([self.normalize(emotion_score[i][emotion] - 0.2) for i, emotion in enumerate(dialogue_emotions)])

        emotional_consistency_score_0_5_wo_n = np.mean(
            [self.normalize(emotion_score[i][emotion] - self.threshold) for i, emotion in enumerate(dialogue_emotions) if self.emotion_classifier.model.config.id2label[emotion] != "neutral"]
        )
        emotional_consistency_score_0_2_wo_n = np.mean(
            [self.normalize(emotion_score[i][emotion] - 0.2) for i, emotion in enumerate(dialogue_emotions) if self.emotion_classifier.model.config.id2label[emotion] != "neutral"]
        )

        generated_summary_emotions = emotion_predict_result.predictions.argmax(-1)
        generated_summary_labels = pd.Series(generated_summary_emotions).map(self.emotion_classifier.model.config.id2label)

        def get_key(emotional_consistency):
            return "emotional_" + ("consistency" if emotional_consistency else "inconsistency")

        result = (generated_summary_labels == labels).value_counts() / len(labels)
        result = {get_key(k): v * 100 for k, v in result.items()}
        result.update(generated_summary_labels.value_counts().to_dict())

        non_neutral_match_count = 0
        non_neutral_mismatch_count = 0
        non_emotional_count = 0

        for generated_emotion, label in zip(generated_summary_labels, labels):
            if label != "neutral":
                if generated_emotion == "neutral":
                    non_emotional_count += 1

                if generated_emotion == label:
                    non_neutral_match_count += 1
                else:
                    non_neutral_mismatch_count += 1

        result["non_emotional_count"] = non_emotional_count
        result["non_neutral_match_count"] = non_neutral_match_count
        result["non_neutral_mismatch_count"] = non_neutral_mismatch_count
        result["emotional_consistency_wo_n"] = (non_neutral_match_count / (non_neutral_match_count + non_neutral_mismatch_count)) * 100
        result["emotional_consistency_score_0_5"] = emotional_consistency_score_0_5 * 100
        result["emotional_consistency_score_0_2"] = emotional_consistency_score_0_2 * 100
        result["emotional_consistency_score_0_5_wo_n"] = emotional_consistency_score_0_5_wo_n * 100
        result["emotional_consistency_score_0_2_wo_n"] = emotional_consistency_score_0_2_wo_n * 100

        try:
            top_1_acc = top_k_accuracy_score(dialogue_emotions, emotion_predict_result.predictions, k=1)
            top_3_acc = top_k_accuracy_score(dialogue_emotions, emotion_predict_result.predictions, k=3)
            result["top_1_acc"] = top_1_acc * 100
            result["top_3_acc"] = top_3_acc * 100
        except ValueError as ve:
            print(f"Calculate top-k Accuracy failed due to {ve}")

        return result

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        decoded_preds = self.tokenizer.decode_summary(predictions)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id())
        decoded_labels = self.tokenizer.decode_summary(labels)

        # For emotion classification
        decode_preds_for_emotion = self.tokenizer.decode_sentence(decoded_preds, " ")
        decoded_lables_for_dataset = self.tokenizer.decode_sentence(decoded_labels, " ")

        # Rouge expects a newline after each sentence
        decoded_preds = self.tokenizer.decode_sentence(decoded_preds)
        decoded_labels = self.tokenizer.decode_sentence(decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        (P, R, F) = score(decoded_preds, decoded_labels, lang="en")

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result["bertscore_P"] = P.mean().item() * 100
        result["bertscore_R"] = R.mean().item() * 100
        result["bertscore_F"] = F.mean().item() * 100

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id()) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        dataset = None

        labels_set = set(decoded_lables_for_dataset)
        train_prob = len(labels_set - set(self.train_dataset[self.summary_tag]))
        validation_prob = len(labels_set - set(self.validation_dataset[self.summary_tag]))
        test_prob = len(labels_set - set(self.test_dataset[self.summary_tag]))

        prob = min(train_prob, validation_prob, test_prob)

        if prob == train_prob:
            dataset = self.train_dataset
        elif prob == validation_prob:
            dataset = self.validation_dataset
        elif prob == test_prob:
            dataset = self.test_dataset

        if dataset is not None and "dialogue_emotion" in dataset.features.keys():
            result.update(self.calculate_emotional_consistency(decode_preds_for_emotion, dataset["dialogue_emotion"]))

        return {k: round(v, 3) for k, v in result.items()}
