import numpy as np
import torch

from data import SimpleDataset

from loss import CB_loss
from focal_loss import FocalLoss
from transformers import Seq2SeqTrainer, LogitsProcessorList
from transformers.adapters import Seq2SeqAdapterTrainer
from transformers.deepspeed import is_deepspeed_zero3_enabled


class EmotionalGenerator(LogitsProcessorList):
    def __init__(self) -> None:
        super().__init__()


class EmotionalTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        emotional_loss_weight=0.0,
        emotional_loss_params=None,
        emotion_classifier=None,
        emotional_tokenizer=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_emotional_consistency_loss = emotional_loss_weight > 0.0

        if self.compute_emotional_consistency_loss:
            self.num_of_emotions = [742, 32, 34, 3277, 7906, 321, 148]  # Anger  # Disgust  # Fear  # Joy  # Neutral  # Sadness  # Surprise
            self.weights = torch.FloatTensor([1 - (x / sum(self.num_of_emotions)) for x in self.num_of_emotions]).to(self.model.device, (torch.float16 if self.args.fp16 else torch.float32))
            self.loss_type = emotional_loss_params["loss_type"]
            self.beta = emotional_loss_params["beta"]
            self.gamma = emotional_loss_params["gamma"]
            self.emotion_classifier = emotion_classifier
            self.emotional_tokenizer = emotional_tokenizer
            self.emotional_loss_weight = emotional_loss_weight
            self.emotion_loss_fn = FocalLoss(self.weights, self.gamma) if "focal" in self.loss_type else torch.nn.CrossEntropyLoss(weight=self.weights)
            print(f"Emotion Loss Fn is {self.emotion_loss_fn}")

    def extact_emotion_labels(self, inputs):
        _ = inputs.pop("dialogue_emotion", None)
        return inputs.pop("emotion_labels", None)

    def compute_loss(self, model, inputs, return_outputs=False):
        emotion_lables = self.extact_emotion_labels(inputs)

        summary_loss, summary_output = super().compute_loss(model, inputs, return_outputs=True)

        if not self.compute_emotional_consistency_loss:
            return (summary_loss, summary_output) if return_outputs else summary_loss

        generated_summary = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.args.generation_max_length,
            num_beams=self.args.generation_num_beams,
            synced_gpus=True if is_deepspeed_zero3_enabled() else False,
        )

        decoded_output = self.emotional_tokenizer.decode_summary(generated_summary)
        decoded_output = self.emotional_tokenizer.decode_sentence(decoded_output, " ")

        tokenized_output = self.emotional_tokenizer.encode_emotion(decoded_output)
        emotion_predict_result = self.emotion_classifier.predict(SimpleDataset(tokenized_output))
        emotion_scores = (np.exp(emotion_predict_result[0])) / np.exp(emotion_predict_result[0]).sum(-1, keepdims=True)

        emotion_loss = self.emotion_loss_fn(torch.FloatTensor(emotion_scores).to(summary_loss.device), torch.FloatTensor(emotion_lables).to(summary_loss.device))

        T = 0.5

        if self.emotional_loss_weight < 1.0:
            summary_loss = summary_loss * self.emotional_loss_weight

        total_loss = summary_loss + emotion_loss * self.emotional_loss_weight / T

        if self.state.global_step % 100 == 0:
            self.log({"Emotional Loss": emotion_loss.item(), "Summary Loss": summary_loss.item(), "Total Loss": total_loss.item()})

        return (total_loss, summary_output) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        _ = self.extact_emotion_labels(inputs)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )


class EmotionalAdapterTrainer(Seq2SeqAdapterTrainer):
    def __init__(
        self,
        emotional_loss_weight=0.0,
        emotional_loss_params=None,
        emotion_classifier=None,
        emotional_tokenizer=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_emotional_consistency_loss = emotional_loss_weight > 0.0

        if self.compute_emotional_consistency_loss:
            self.num_of_emotions = [742, 32, 34, 3277, 7906, 321, 148]  # Anger  # Disgust  # Fear  # Joy  # Neutral  # Sadness  # Surprise
            self.weights = torch.FloatTensor([1 - (x / sum(self.num_of_emotions)) for x in self.num_of_emotions]).to(self.model.device, (torch.float16 if self.args.fp16 else torch.float32))
            self.loss_type = emotional_loss_params["loss_type"]
            self.beta = emotional_loss_params["beta"]
            self.gamma = emotional_loss_params["gamma"]
            self.emotion_classifier = emotion_classifier
            self.emotional_tokenizer = emotional_tokenizer
            self.emotional_loss_weight = emotional_loss_weight
            self.emotion_loss_fn = FocalLoss(self.weights, self.gamma) if "focal" in self.loss_type else torch.nn.CrossEntropyLoss(weight=self.weights)
            print(f"Emotion Loss Fn is {self.emotion_loss_fn}")

    def extact_emotion_labels(self, inputs):
        _ = inputs.pop("dialogue_emotion", None)
        return inputs.pop("emotion_labels", None)

    def compute_loss(self, model, inputs, return_outputs=False):
        emotion_lables = self.extact_emotion_labels(inputs)

        summary_loss, summary_output = super().compute_loss(model, inputs, return_outputs=True)

        if not self.compute_emotional_consistency_loss:
            return (summary_loss, summary_output) if return_outputs else summary_loss

        generated_summary = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.args.generation_max_length,
            num_beams=self.args.generation_num_beams,
            synced_gpus=True if is_deepspeed_zero3_enabled() else False,
        )

        decoded_output = self.emotional_tokenizer.decode_summary(generated_summary)
        decoded_output = self.emotional_tokenizer.decode_sentence(decoded_output, " ")

        tokenized_output = self.emotional_tokenizer.encode_emotion(decoded_output)
        emotion_predict_result = self.emotion_classifier.predict(SimpleDataset(tokenized_output))
        emotion_scores = (np.exp(emotion_predict_result[0])) / np.exp(emotion_predict_result[0]).sum(-1, keepdims=True)

        emotion_loss = self.emotion_loss_fn(torch.FloatTensor(emotion_scores).to(summary_loss.device), torch.FloatTensor(emotion_lables).to(summary_loss.device))

        T = 0.5

        if self.emotional_loss_weight < 1.0:
            summary_loss = summary_loss * self.emotional_loss_weight

        total_loss = summary_loss + emotion_loss * self.emotional_loss_weight / T

        if self.state.global_step % 100 == 0:
            self.log({"Emotional Loss": emotion_loss.item(), "Summary Loss": summary_loss.item(), "Total Loss": total_loss.item()})

        return (total_loss, summary_output) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        _ = self.extact_emotion_labels(inputs)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )
