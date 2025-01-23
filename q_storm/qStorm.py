import torch
from torch import nn


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return lambda x: x


def create_dgm_layers(dgm_dims: int, n_dgm_layers: int):
    if not dgm_dims:
        return

    layers = nn.ModuleList()
    for _ in range(n_dgm_layers):
        layers.append(DGMLayer(dgm_dims))
    return layers


def create_fc_layers(input_dim: int,
                     hidden_dims: list[int],
                     dense_activation: str,
                     dgm_dims: int,
                     n_dgm_layers: int,
                     output_activation: str,
                     output_dim: int = 1):
    dense_layers_list = nn.ModuleList()
    output_layers = nn.ModuleList()
    dgm_layers_list = None
    output_layer_in_params = 1
    input_layer = nn.Linear(input_dim, output_dim)
    if hidden_dims:
        input_layer = nn.Linear(input_dim, hidden_dims[0])
        if dense_activation:
            dense_layers_list.append(get_activation(dense_activation))
        for i in range(1, len(hidden_dims)):
            dense_layers_list.append(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if dense_activation:
                dense_layers_list.append(get_activation(dense_activation))
        output_layer_in_params = hidden_dims[-1]
        if dgm_dims and n_dgm_layers:
            dense_layers_list.append(nn.Linear(hidden_dims[-1], dgm_dims))

    if dgm_dims and n_dgm_layers:
        if not dense_layers_list:
            input_layer = nn.Linear(input_dim, dgm_dims)
        dgm_layers_list = create_dgm_layers(
            dgm_dims, n_dgm_layers)
        output_layer_in_params = dgm_dims

    output_layers.append(nn.Linear(output_layer_in_params, output_dim))
    if output_activation:
        output_layers.append(get_activation(output_activation))
    output_layer = nn.Sequential(*output_layers)

    if not dense_layers_list and not dgm_layers_list:
        input_layer = nn.Sequential(
            input_layer, get_activation(dense_activation))

    return input_layer, dense_layers_list, dgm_layers_list, output_layer


class DGMLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(DGMLayer, self).__init__()

        # Initialize all linear layers with Xavier/Glorot initialization
        self.U_z = nn.Linear(hidden_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)
        self.B_z = nn.Linear(hidden_dim, hidden_dim)

        self.U_g = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)
        self.B_g = nn.Linear(hidden_dim, hidden_dim)

        self.U_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.B_r = nn.Linear(hidden_dim, hidden_dim)

        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.B_h = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, S, S1):
        Z = torch.sigmoid(self.U_z(x) + self.W_z(S) + self.B_z(S1))
        G = torch.sigmoid(self.U_g(x) + self.W_g(S1) + self.B_g(S))
        R = torch.sigmoid(self.U_r(x) + self.W_r(S) + self.B_r(S1))
        H = torch.tanh(self.U_h(x) + self.W_h(S * R) + self.B_h(S1))
        S_new = (1 - G) * H + Z * S
        return S_new


class QStorm(nn.Module):
    def __init__(self,
                 input_dims: int,
                 hidden_dims: list,
                 dgm_dims: int,
                 n_dgm_layers: int,
                 hidden_activation: str):
        super(QStorm, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dgm_dims = dgm_dims
        self.n_dgm_layers = n_dgm_layers
        self.hidden_activation = hidden_activation
        self.output_activation = "sigmoid"

        layers = create_fc_layers(
            input_dims, hidden_dims, hidden_activation, dgm_dims,
            n_dgm_layers, "sigmoid")
        self.input_layer, self.hidden_layers, self.dgm_layers, self.output_layer = layers

    def forward(self, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):  # -> Any:
        model_inps = torch.cat([S_prime, K_prime, t_prime, rf, varpi_q], dim=1)
        X = self.input_layer(model_inps)
        if self.hidden_layers:
            for layer in self.hidden_layers:
                X = layer(X)
        if self.dgm_layers:
            S1 = X
            S = X
            for layer in self.dgm_layers:
                S = layer(X, S, S1)
            X = S

        V_prime = self.output_layer(X)
        t = t_prime * T
        S_price_std = S_std * S_mu
        S = S_prime * S_price_std + S_mu
        S_max = 10 * S_price_std + S_mu
        K = K_prime * S_price_std + S_mu
        V = V_prime * S_max
        return V, S, K, t


if __name__ == "__main__":
    model = QStorm(36, [10, 20], 10, 2, 'relu')
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
