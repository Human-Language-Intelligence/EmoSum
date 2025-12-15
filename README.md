# EmoSum

## Prepare
```bash
conda create -n emosum python=3.8
conda activate emosum
pip install -r requirements.txt
ln -s /mnt/hlilabshare/HLILab_Public/yjspecial/yjspecial/everest/EmotionalBART/checkpoints checkpoints
ln -s /mnt/hlilabshare/HLILab_Public/yjspecial/yjspecial/everest/EmotionalBART/data data
````

**Be careful** not to overwrite the original files in `yjspecial`.

```python
import nltk
nltk.download("punkt")
```

**Execute the code above in a Python environment.**

**Note:** Since the version of the `transformers` library used in this codebase is outdated, it cannot automatically download Hugging Face models from the internet.

**Option 1: Use Local Models (For Lab Server Users)**
Since `git-lfs` is not supported on the lab server, please create symbolic links to the pre-downloaded models using the commands below:

```bash
ln -s /mnt/hlilabshare/HLILab_Public/bart-large bart-large
ln -s /mnt/hlilabshare/HLILab_Public/emotion-english-roberta-large emotion-english-roberta-large
```

**Option 2: Download Manually via Git LFS (For External Users)**
If you are working outside the lab server, you can download the models directly using `git lfs`:

```bash
# Ensure git-lfs is installed
git lfs install

# Download models
git clone https://huggingface.co/facebook/bart-large
git clone https://huggingface.co/j-hartmann/emotion-english-roberta-large
```

## Train

### Phase 1: Full Fine-tuning
```bash
GPU_DEVICE=0  # Specify the GPU index you wish to use
FFT_OUTPUT=output/fft

./train_finetuning.sh $GPU_DEVICE $FFT_OUTPUT
```

### Phase 2: LoRA Fine-tuning
```bash
GPU_DEVICE=0  # Specify the GPU index you wish to use
FFT_OUTPUT=output/fft/checkpoint-N # Replace 'checkpoint-N' with the correct checkpoint directory name
#FFT_OUTPUT=checkpoints/ours/BART16 # If you want to test the script, use it
LORA_OUTPUT=output/lora

./train_adapter.sh $GPU_DEVICE $LORA_OUTPUT $FFT_OUTPUT
```

## Evaluate with the Trained Model
```bash
GPU_DEVICE=0 # Specify the GPU index you wish to use
BASE_MODEL=checkpoints/ours/BART16
LORA=checkpoints/ours/lora/summarization_adapter
OUTPUT=eval_result

./peft_eval.sh $GPU_DEVICE $BASE_MODEL $LORA $OUTPUT
```

Expected Result
```json
{'test_loss': 2.8752377033233643, 'test_rouge1': 49.163, 'test_rouge2': 23.381, 'test_rougeL': 40.613, 'test_rougeLsum': 43.877, 'test_bertscore_P': 91.046, 'test_bertscore_R': 92.187, 'test_bertscore_F': 91.603, 'test_gen_len': 34.342, 'test_emotional_consistency': 61.2, 'test_emotional_inconsistency': 38.8, 'test_neutral': 352, 'test_joy': 60, 'test_sadness': 40, 'test_fear': 20, 'test_anger': 13, 'test_surprise': 10, 'test_disgust': 5, 'test_non_emotional_count': 97, 'test_non_neutral_match_count': 51, 'test_non_neutral_mismatch_count': 132, 'test_emotional_consistency_wo_n': 27.869, 'test_emotional_consistency_score_0_5': 16.998, 'test_emotional_consistency_score_0_2': 35.858, 'test_emotional_consistency_score_0_5_wo_n': 5.566, 'test_emotional_consistency_score_0_2_wo_n': 14.614, 'test_top_1_acc': 61.2, 'test_top_3_acc': 88.2, 'test_runtime': 127.0155, 'test_samples_per_second': 3.937, 'test_steps_per_second': 0.984}
```
**'test_top_3_acc': 88.2**

## Citation
```
@inproceedings{10.1145/3605098.3635900,
author = {Jo, Youngjin and Bak, Jinyeong},
title = {EmoSum: Conversation Summarization with Emotional Consistency},
year = {2024},
isbn = {9798400702433},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605098.3635900},
doi = {10.1145/3605098.3635900},
booktitle = {Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing},
pages = {723–730},
numpages = {8},
keywords = {emotional consistency, conversations, summarization, catastrophic forgetting},
location = {Avila, Spain},
series = {SAC '24}
}
```

## Acknowledgement

We thank Taemin Yeom (taemin.yeom@g.skku.edu), Eunbeen Son (nabin111@g.skku.edu), and Taekhyun Kim (treecko.kth@g.skku.edu) for refactoring the code and Dockerfile.
