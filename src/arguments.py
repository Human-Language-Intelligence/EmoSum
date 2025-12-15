from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(default="facebook/bart-large", metadata={"help": "model"})
    gpu_index: int = field(default=0)
    check_emotional_consistency: bool = field(default=False)
    emotional_loss_weight: float = field(default=0.0)
    emotional_loss_type: str = field(default="cross_entropy", metadata={"help": "emotional loss type"})
    emotional_loss_beta: float = field(default=0.9999)
    emotional_loss_gamma: float = field(default=2.0)
    enable_logging: bool = field(default=False)
    speaker_aware_summarization: bool = field(default=False)
    target_speaker: str = field(default=None)


@dataclass
class DataArguemnts:
    data_dir: str

    max_source_length: int = field(default=512)
    max_target_length: int = field(default=256)
    test_label_tag: str = field(default="summary1")
