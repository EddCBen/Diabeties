import torch
import torch.nn as nn

class Diabeties_classifier(nn.Module):
    r"""
    Diabieties Classifier Class
    """
    def __init__(self):
        super().__init__()
        self.Dense_expand = nn.Linear(5,20)
        self.TransEnc_layer = nn.TransformerEncoderLayer(d_model=20, nhead=1)
        self.Transformer_Encoder = nn.TransformerEncoder(self.TransEnc_layer, num_layers=1)
        self.dropout = nn.Dropout(p=0.005)
        self.Dense_reduce = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        
    def forward(self, input_sequence : torch.Tensor) -> torch.Tensor:
        expanded_sequence = self.Dense_expand(input_sequence)
        expanded_sequence = self.tanh(expanded_sequence)
        expanded_sequence = self.dropout(expanded_sequence)
        encoded_sequence = self.Transformer_Encoder(expanded_sequence)
        encoded_sequence = self.gelu(encoded_sequence)
        encoded_sequence = self.dropout(encoded_sequence)
        reduced_sequence = self.Dense_reduce(encoded_sequence)
        reduced_sequence = self.tanh(reduced_sequence)
        class_ = self.sigmoid(reduced_sequence)
        return class_