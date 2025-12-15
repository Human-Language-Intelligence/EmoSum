import torch.nn as nn

class EmotionalEmbedding(nn.Module):
    def __init__(self, config, word_embedding):
        super().__init__()
        
        assert config is not None
        assert word_embedding is not None
        
        self.word_embedding = word_embedding
        self.emotional_embedding = nn.Embedding(config.vocab_size, config.d_model)
    
    def forward(self, inputs):
        input_embeds = self.word_embedding(inputs)
        
        return None