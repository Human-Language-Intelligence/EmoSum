import nltk

class EmotionalTokenizer:
    def __init__(self, summary_tokenizer, emotion_tokenizer):
        self.summary_tokenizer = summary_tokenizer
        self.emotion_tokenizer = emotion_tokenizer
    
    def pad_token_id(self):
        return self.summary_tokenizer.pad_token_id
    
    def encode_emotion(self, text):
        return self.emotion_tokenizer(text, truncation=True, padding=True)
    
    def decode_summary(self, summary):
        return self.summary_tokenizer.batch_decode(summary, skip_special_tokens=True)

    def decode_sentence(self, sentences, delimiter="\n"):
        return [delimiter.join(nltk.sent_tokenize(pred.strip())) for pred in sentences]
    