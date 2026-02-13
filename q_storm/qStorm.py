import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class QuantileAttention(nn.Module):
    """Self-attention mechanism for quantile inputs"""

    def __init__(self, quantile_dim: int, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(quantile_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, varpi_q: torch.Tensor) -> torch.Tensor:
        # varpi_q: (batch, num_quantiles)
        # Treat quantiles as sequence
        x = self.embed(varpi_q).unsqueeze(1)  # (batch, 1, embed_dim)
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm(attn_out + x)  # Residual connection
        return self.dropout_layer(attn_out).squeeze(1)  # (batch, embed_dim)


class ResidualBlock(nn.Module):
    """Residual block with pre-norm architecture"""

    def __init__(self, dim: int, activation: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return x + residual  # Skip connection


class EnhancedDGMLayer(nn.Module):
    """Enhanced DGM layer with normalization and dropout"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Combine U, W, B matrices for each gate into single matrices
        # This allows for single matrix multiplication instead of three
        self.z_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.g_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.r_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.h_matrix = nn.Linear(hidden_dim * 3, hidden_dim)

        # Layer normalization for stability
        self.norm_z = nn.LayerNorm(hidden_dim)
        self.norm_g = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # Use better initialization with gain scaling
        for m in [self.z_matrix, self.g_matrix, self.r_matrix, self.h_matrix]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor, S1: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs once for all gates
        combined_input = torch.cat([x, S, S1], dim=-1)

        # Compute all gates with normalization
        Z = torch.sigmoid(self.norm_z(self.z_matrix(combined_input)))
        G = torch.sigmoid(self.norm_g(self.g_matrix(combined_input)))
        R = torch.sigmoid(self.norm_r(self.r_matrix(combined_input)))

        # Compute H gate with modified input
        SR_combined = torch.cat([x, S * R, S1], dim=-1)
        H = torch.tanh(self.norm_h(self.h_matrix(SR_combined)))
        H = self.dropout(H)

        # Fuse final computations
        return (1 - G) * H + Z * S


# Keep OptimizedDGMLayer for backward compatibility
OptimizedDGMLayer = EnhancedDGMLayer


class QStorm(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: List[int],
        dgm_dims: int,
        n_dgm_layers: int,
        hidden_activation: str,
        use_attention: bool = False,
        use_residual: bool = False,
        dropout: float = 0.0,
        quantile_embed_dim: Optional[int] = None
    ):
        super().__init__()

        # Store configuration
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.dropout_rate = dropout

        # Cache activation function
        self.activation = self._get_activation(hidden_activation)

        # Determine quantile dimension (assuming varpi_q is last part of input)
        # Input is: [S', t', rf, varpi_q] = 3 + quantile_dim = input_dims
        # So quantile_dim = input_dims - 3
        quantile_dim = input_dims - 3

        # Quantile attention if enabled
        self.quantile_attention = None
        if use_attention and quantile_dim > 0:
            if quantile_embed_dim is None:
                # Auto-set to first hidden dim or 64
                quantile_embed_dim = hidden_dims[0] if hidden_dims else 64
            self.quantile_attention = QuantileAttention(
                quantile_dim, quantile_embed_dim, dropout=dropout
            )
            # Adjust input dims to account for quantile embedding
            adjusted_input_dims = 3 + quantile_embed_dim
        else:
            adjusted_input_dims = input_dims

        # Build model components
        self.input_net, self.dgm_net, self.output_net = self._build_network(
            adjusted_input_dims, hidden_dims, dgm_dims, n_dgm_layers
        )

        # Pre-allocate concatenation indices
        self.register_buffer(
            'concat_indices',
            torch.arange(4, dtype=torch.long)
        )

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Enhanced activation function lookup"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),  # Swish activation
            'gelu': nn.GELU(),   # GELU activation
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(activation.lower(), nn.Identity())

    def _build_network(
        self,
        input_dims: int,
        hidden_dims: List[int],
        dgm_dims: int,
        n_dgm_layers: int
    ) -> Tuple[nn.Module, Optional[nn.ModuleList], nn.Module]:
        # Build input network
        layers = []
        current_dim = input_dims

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Add residual block if enabled and dimensions match
            if self.use_residual and i > 0 and current_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim, self.activation, self.dropout_rate))
            else:
                # Standard layer with normalization and dropout
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(self.activation)
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))

            current_dim = hidden_dim

        if dgm_dims:
            layers.append(nn.Linear(current_dim, dgm_dims))
            layers.append(nn.LayerNorm(dgm_dims))
            current_dim = dgm_dims

        input_net = nn.Sequential(*layers)

        # Build DGM network if needed
        dgm_net = None
        if dgm_dims and n_dgm_layers:
            dgm_net = nn.ModuleList([
                EnhancedDGMLayer(dgm_dims, self.dropout_rate)
                for _ in range(n_dgm_layers)
            ])

        # Output network with optional normalization
        if self.dropout_rate > 0 or self.use_residual:
            output_layers = [
                nn.Linear(current_dim, current_dim // 2),
                nn.LayerNorm(current_dim // 2),
                nn.ReLU(),
            ]
            if self.dropout_rate > 0:
                output_layers.append(nn.Dropout(self.dropout_rate))
            output_layers.append(nn.Linear(current_dim // 2, 1))
            output_net = nn.Sequential(*output_layers)
        else:
            # Simple linear layer for backward compatibility
            output_net = nn.Linear(current_dim, 1)

        return input_net, dgm_net, output_net

    def forward(
        self,
        S_prime: torch.Tensor,
        t_prime: torch.Tensor,
        rf: torch.Tensor,
        varpi_q: torch.Tensor
    ) -> torch.Tensor:
        # Process quantiles with attention if enabled
        if self.quantile_attention is not None:
            quantile_features = self.quantile_attention(varpi_q)
            # Replace quantile part of input with attention features
            scalar_inputs = torch.cat([S_prime, t_prime, rf], dim=1)
            model_inps = torch.cat([scalar_inputs, quantile_features], dim=1)
        else:
            # Standard concatenation
            model_inps = torch.cat([S_prime, t_prime, rf, varpi_q], dim=1)

        # Forward through input network
        X = self.input_net(model_inps)

        # DGM processing if present
        if self.dgm_net is not None:
            dgm_s = dgm_s1 = X
            for layer in self.dgm_net:
                dgm_s = layer(X, dgm_s, dgm_s1)
            X = dgm_s

        # Output projection
        return F.softplus(self.output_net(X))


if __name__ == "__main__":
    # Test backward compatibility
    model = QStorm(35, [256, 64, 64, 32], 10, 2, 'relu')
    dummy_s_prime = torch.randn(100, 1)
    dummy_t_prime = torch.rand(100, 1)
    dummy_rf = torch.rand(100, 1)
    dummy_varpi_q = torch.rand(100, 32)
    model_V = model(dummy_s_prime, dummy_t_prime, dummy_rf, dummy_varpi_q)
    print(
        f"Backward compatibility test: V max={model_V.max():.4f}, min={model_V.min():.4f}, mean={model_V.mean():.4f}")

    # Test enhanced model
    model_enhanced = QStorm(35, [256, 64, 64, 32], 10, 2, 'relu',
                            use_attention=True, use_residual=True, dropout=0.1)
    model_V_enhanced = model_enhanced(dummy_s_prime, dummy_t_prime, dummy_rf, dummy_varpi_q)
    print(
        f"Enhanced model test: V max={model_V_enhanced.max():.4f}, min={model_V_enhanced.min():.4f}, mean={model_V_enhanced.mean():.4f}")
