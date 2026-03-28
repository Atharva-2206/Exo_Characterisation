import torch
import torch.nn as nn

class Oblateness1DCNN(nn.Module):
    def __init__(self, sequence_length=1000):
        super().__init__()
        
        # Feature Extraction Block (Time-Series to Feature Maps)
        self.features = nn.Sequential(
            # Layer 1: Catch the broad transit shape
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2: Look for the specific ingress/egress anomalies
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 3: Deep feature compression
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate the flattened size: 
        # 1000 sequence length / 2 / 2 / 2 = 125
        # 125 * 64 channels = 8000
        self.flattened_size = 64 * 125
        
        # Regression Head (Mapping features to a continuous J2 value)
        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevent overfitting to the synthetic red noise
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Single output node for J2
        )
        
    def forward(self, x):
        # PyTorch Conv1d expects shape: (Batch_Size, Channels, Sequence_Length)
        # Our raw data is (Batch_Size, Sequence_Length). This adds the Channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1) 
            
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        
        # Squeeze removes the extra dimension so output matches our y_target arrays
        return x.squeeze()