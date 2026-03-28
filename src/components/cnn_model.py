import torch
import torch.nn as nn

class Oblateness1DCNN(nn.Module):
    def __init__(self, sequence_length=250):
        super().__init__()
        
        # ------------------------------------------------------------------
        # BRANCH 1: The Light Curve Feature Extractor (1D-CNN)
        # ------------------------------------------------------------------
        self.feature_branch = nn.Sequential(
            # Layer 1: Detect the broad U-shape
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), # Batch Normalization stabilizes training on messy TESS data
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 2: Detect the specific ingress/egress anomalies
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Layer 3: Deep compression
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate flattened size: 250 / 2 / 2 / 2 = 31
        # 31 * 64 channels = 1984
        self.flattened_size = 64 * 31
        
        # ------------------------------------------------------------------
        # BRANCH 2: The Physical Context (Dense)
        # ------------------------------------------------------------------
        # This branch takes a single scalar: The maximum transit depth.
        # This breaks the degeneracy between Radius and Oblateness.
        self.physical_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        
        # ------------------------------------------------------------------
        # FUSION & REGRESSION HEAD
        # ------------------------------------------------------------------
        # We concatenate the 1984 CNN features with the 16 Physical features
        fusion_size = self.flattened_size + 16
        
        self.regressor = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            # Crucial: Monte Carlo Dropout. We leave this ON during inference 
            # to generate error bars for our predictions.
            nn.Dropout(0.3), 
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Single output node for J2
        )
        
    def forward(self, x_flux, x_depth):
        # 1. Process the Flux array through the CNN
        if x_flux.dim() == 2:
            x_flux = x_flux.unsqueeze(1) # Add channel dimension (Batch, 1, 250)
            
        cnn_features = self.feature_branch(x_flux)
        cnn_features = torch.flatten(cnn_features, 1) # Shape: (Batch, 1984)
        
        # 2. Process the scalar Depth through the Dense layer
        if x_depth.dim() == 1:
            x_depth = x_depth.unsqueeze(1) # Shape: (Batch, 1)
            
        phys_features = self.physical_branch(x_depth) # Shape: (Batch, 16)
        
        # 3. Concatenate (Fuse) the branches
        fused = torch.cat((cnn_features, phys_features), dim=1) # Shape: (Batch, 2000)
        
        # 4. Final Regression
        output = self.regressor(fused)
        return output.squeeze()