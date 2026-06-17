"""Simple action prediction model for GEAR-SONIC training."""

import torch
import torch.nn as nn


class SonicActionPredictor(nn.Module):
    """Transformer-based action prediction model.
    
    Takes observation context and predicts future actions.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_length: int = 4,
        action_horizon: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize the model.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            context_length: Number of context frames
            action_horizon: Number of actions to predict
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_pos_encoding(context_length, hidden_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Output projection for action horizon
        self.output_projection = nn.Linear(hidden_dim, action_horizon * action_dim)
    
    @staticmethod
    def _create_pos_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, obs_context: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs_context: (batch_size, context_length, obs_dim)
        
        Returns:
            actions: (batch_size, action_horizon, action_dim)
        """
        batch_size = obs_context.size(0)
        
        # Encode observations
        obs_encoded = self.obs_encoder(obs_context)  # (batch, context_length, hidden_dim)
        
        # Add positional encoding
        seq_len = obs_context.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :].to(obs_encoded.device)
        obs_encoded = obs_encoded + pos_enc
        
        # Transformer
        x = self.transformer(obs_encoded)  # (batch, context_length, hidden_dim)
        
        # Take last token and project to action horizon
        x = x[:, -1, :]  # (batch, hidden_dim)
        actions = self.output_projection(x)  # (batch, action_horizon * action_dim)
        
        # Reshape to (batch, action_horizon, action_dim)
        actions = actions.view(batch_size, self.action_horizon, self.action_dim)
        
        return actions


class SonicMLP(nn.Module):
    """Simple MLP baseline for action prediction."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_length: int = 4,
        action_horizon: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize MLP model.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            context_length: Number of context frames
            action_horizon: Number of actions to predict
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        
        # Flatten context to single vector
        input_dim = obs_dim * context_length
        output_dim = action_dim * action_horizon
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, obs_context: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs_context: (batch_size, context_length, obs_dim)
        
        Returns:
            actions: (batch_size, action_horizon, action_dim)
        """
        batch_size = obs_context.size(0)
        
        # Flatten context
        x = obs_context.view(batch_size, -1)
        
        # MLP forward
        x = self.model(x)
        
        # Reshape to (batch, action_horizon, action_dim)
        x = x.view(batch_size, self.action_horizon, self.action_dim)
        
        return x
