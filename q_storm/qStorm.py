import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class OptimizedDGMLayer(nn.Module):
    """Optimized DGM layer with fused operations"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Combine U, W, B matrices for each gate into single matrices
        # This allows for single matrix multiplication instead of three
        self.z_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.g_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.r_matrix = nn.Linear(hidden_dim * 3, hidden_dim)
        self.h_matrix = nn.Linear(hidden_dim * 3, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        # Use faster initialization
        for m in [self.z_matrix, self.g_matrix, self.r_matrix, self.h_matrix]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, S: torch.Tensor, S1: torch.Tensor) -> torch.Tensor:
        # Concatenate inputs once for all gates
        combined_input = torch.cat([x, S, S1], dim=-1)

        # Compute all gates with single matrix multiplications
        Z = torch.sigmoid(self.z_matrix(combined_input))
        G = torch.sigmoid(self.g_matrix(combined_input))
        R = torch.sigmoid(self.r_matrix(combined_input))

        # Compute H gate with modified input
        SR_combined = torch.cat([x, S * R, S1], dim=-1)
        H = torch.tanh(self.h_matrix(SR_combined))

        # Fuse final computations
        return (1 - G) * H + Z * S


class QStorm(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: List[int],
        dgm_dims: int,
        n_dgm_layers: int,
        hidden_activation: str
    ):
        super().__init__()

        # Cache activation function
        self.activation = self._get_activation(hidden_activation)

        # Build model components
        self.input_net, self.dgm_net, self.output_net = self._build_network(
            input_dims, hidden_dims, dgm_dims, n_dgm_layers
        )

        # Pre-allocate concatenation indices
        self.register_buffer(
            'concat_indices',
            torch.arange(4, dtype=torch.long)
        )

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Fast activation function lookup"""
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }.get(activation, nn.Identity())

    def _build_network(
        self,
        input_dims: int,
        hidden_dims: List[int],
        dgm_dims: int,
        n_dgm_layers: int
    ) -> Tuple[nn.Sequential, Optional[nn.ModuleList], nn.Linear]:
        # Build input network
        layers = []
        current_dim = input_dims

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                self.activation
            ])
            current_dim = hidden_dim

        if dgm_dims:
            layers.append(nn.Linear(current_dim, dgm_dims))
            current_dim = dgm_dims

        input_net = nn.Sequential(*layers)

        # Build DGM network if needed
        dgm_net = None
        if dgm_dims and n_dgm_layers:
            dgm_net = nn.ModuleList([
                OptimizedDGMLayer(dgm_dims)
                for _ in range(n_dgm_layers)
            ])

        # Output network is a simple linear layer
        output_net = nn.Linear(current_dim, 1)

        return input_net, dgm_net, output_net

    def forward(
        self,
        S_prime: torch.Tensor,
        t_prime: torch.Tensor,
        rf: torch.Tensor,
        varpi_q: torch.Tensor
    ) -> torch.Tensor:
        # Fast input concatenation
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
        return self.output_net(X)


if __name__ == "__main__":
    model = QStorm(35, [10, 20], 10, 2, 'relu')
    dummy_s_prime = torch.randn(100, 1)
    dummy_k_prime = torch.randn(100, 1)
    dummy_t_prime = torch.rand(100, 1)
    dummy_rf = torch.rand(100, 1)
    dummy_varpi_q = torch.rand(100, 32)
    dummy_s_mu = torch.rand(100, 1) * 10 + 100
    dummy_s_std = torch.rand(100, 1)
    dummy_T = 15
    model_V, model_S, model_K, model_t, model_S_max = model(dummy_s_prime,
                                                            dummy_k_prime,
                                                            dummy_t_prime,
                                                            dummy_rf,
                                                            dummy_varpi_q,
                                                            dummy_s_mu,
                                                            dummy_s_std,
                                                            dummy_T)
    print(model_V.max(), model_V.min(), model_V.mean())
