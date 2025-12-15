from dataclasses import dataclass
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
)
from typing import Any


@dataclass
class DataCollatorForEmotion:
    tokenizer: PreTrainedTokenizerBase
    model: Any

    def __post_init__(self):
        self.data_collator_seq2seq = DataCollatorForSeq2Seq(self.tokenizer, self.model)

    def __call__(self, features):
        summary_features = [
            {k: feat[k] for k in ["input_ids", "attention_mask", "labels"]}
            for feat in features
        ]
        out_dict = self.data_collator_seq2seq(summary_features)
        for key in features[0].keys():
            if key in [
                "dialogue_emotion",
                "emotion_labels",
                "speaker1_summary",
                "speaker2_summary",
            ]:
                out_dict[key] = [feature[key] for feature in features]

        return out_dict


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
