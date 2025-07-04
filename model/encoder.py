import torch
from torch import nn
from transformers import AutoModel
#from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder):
    print(f"Getting get_audio_encoder")
    if name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
        return TransformerAudioEncoder(model_name=name, finetune=finetune_encoder)
    else:
        print(f"encoder {name} not in approved list")
        raise NotImplementedError
    
class TransformerAudioEncoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        print(f"Getting pretrained model from {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        # for param in self.encoder.encoder.layers[-15:].parameters():
        #     param.requires_grad = finetune

    def forward(self, x):
        return self.encoder(x).last_hidden_state

