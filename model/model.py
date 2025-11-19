import torch
import torch.nn as nn
import math
import torchinfo

class PositionalEncoding(nn.Module):
    """Standard Transformer Positional Encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BasecallerTransformer(nn.Module):
    """
    A simple Transformer model for basecalling.
    """
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=3, num_classes=5):
        super().__init__()
        # Vocabulary: 4 bases (A,C,G,T) + 1 blank token for CTC
        # Basecaller output classes: 0=A, 1=C, 2=G, 3=T, 4=BLANK
        self.num_classes = num_classes
        
        # 1. CNN Frontend to extract features and downsample
        self.cnn_frontend = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=19, stride=3, padding=9),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=11, stride=3, padding=5),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 4. Classifier Head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor):
        # src shape: [batch_size, channels=1, signal_len=4096]
        
        # Pass through CNN frontend
        x = self.cnn_frontend(src)
        # x shape: [batch_size, d_model, new_seq_len]
        
        # Prepare for Transformer (needs [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model])
        x = x.permute(0, 2, 1) # -> [batch_size, new_seq_len, d_model]
        
        # Add positional encoding
        # Note: Our PE class expects [seq_len, batch_size, d_model], but we'll adapt
        x = x.permute(1, 0, 2) # -> [new_seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2) # -> [batch_size, new_seq_len, d_model]
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        # x shape: [batch_size, new_seq_len, d_model]
        
        # Classifier
        logits = self.classifier(x)
        
        # For CTC loss, we need log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        return log_probs

# --- NEW: Main block to run and summarize the model ---
if __name__ == "__main__":
    # Define model parameters
    D_MODEL = 256
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_CLASSES = 5 # A, C, G, T, Blank

    # Instantiate the model
    model = BasecallerTransformer(
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_classes=NUM_CLASSES
    )

    # Define an example input size (batch_size, channels, sequence_length)
    batch_size = 32
    input_size = (batch_size, 1, 4096)

    # Use torchinfo to print the model summary
    print(f"--- BasecallerTransformer Model Summary (d_model={D_MODEL}, nhead={N_HEAD}, layers={NUM_ENCODER_LAYERS}) ---")
    torchinfo.summary(
        model, 
        input_size=input_size, 
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        verbose=1
    )