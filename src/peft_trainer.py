import numpy as np
import torch

from data import SimpleDataset

from loss import CB_loss
from focal_loss import FocalLoss
from transformers import Seq2SeqTrainer, LogitsProcessorList
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
            # self.emotions_consistency_loss = nn.CrossEntropyLoss()
            self.samples_per_cls = [742, 32, 34, 3277, 7906, 321, 148]  # Anger  # Disgust  # Fear  # Joy  # Neutral  # Sadness  # Surprise
            self.weights = torch.FloatTensor([1 - (x / sum(self.samples_per_cls)) for x in self.samples_per_cls]).to(self.model.device, (torch.float16 if self.args.fp16 else torch.float32))
            self.loss_type = emotional_loss_params["loss_type"]
            self.beta = emotional_loss_params["beta"]
            self.gamma = emotional_loss_params["gamma"]
            self.emotion_classifier = emotion_classifier
            self.emotional_tokenizer = emotional_tokenizer
            self.emotional_loss_weight = emotional_loss_weight
            self.emotion_loss_fn = FocalLoss(self.weights, self.gamma) if "focal" in self.loss_type else torch.nn.CrossEntropyLoss()
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
            input_ids=inputs["input_ids"],
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

        # emotional_consistency_loss = self.emotions_consistency_loss(torch.FloatTensor(emotion_scores), torch.FloatTensor(emotion_lables))

        emotional_cb_loss = CB_loss(
            torch.FloatTensor(emotion_lables).to(summary_loss.device),
            torch.FloatTensor(emotion_scores).to(summary_loss.device),
            self.samples_per_cls,
            len(self.samples_per_cls),
            self.loss_type[3:],
            self.beta,
            self.gamma,
        )

        emotion_focal_loss = self.emotion_loss_fn(torch.FloatTensor(emotion_scores).to(summary_loss.device), torch.FloatTensor(emotion_lables).to(summary_loss.device))

        emotion_loss = emotional_cb_loss if "cb" in self.loss_type else emotion_focal_loss

        T = 1.0

        if self.emotional_loss_weight < 1.0:
            T = 1 - self.emotional_loss_weight
            summary_loss = summary_loss * T

        total_loss = summary_loss + emotion_loss * self.emotional_loss_weight / T

        if self.state.global_step % 100 == 0:
            self.log({"Emotional(CB) Loss": emotional_cb_loss.item(), "Emotional(focal) Loss": emotion_focal_loss.item(), "Summary Loss": summary_loss.item(), "Total Loss": total_loss.item()})

        return (total_loss, summary_output) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None):
        _ = self.extact_emotion_labels(inputs)

        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            input_ids=generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if gen_kwargs.get("max_length") is not None and generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[-1] < (gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_new_tokens"] + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if gen_kwargs.get("max_length") is not None and labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (gen_kwargs["max_new_tokens"] + 1):
                labels = self._pad_tensors_to_max_len(labels, (gen_kwargs["max_new_tokens"] + 1))
        else:
            labels = None

        return (loss, generated_tokens, labels)
