import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp(
    X_np,
    y_np,
    iyengar_w=None,
    hidden_layers=None,
    epochs=100,
    lr=0.001,
    seed=42,
):

    if hidden_layers is None:
        hidden_layers = [64]

    if not isinstance(hidden_layers, (list, tuple)) or len(hidden_layers) == 0:
        raise ValueError("hidden_layers must be a non-empty list or tuple of positive integers.")

    hidden_layers = [int(n) for n in hidden_layers]
    if any(n <= 0 for n in hidden_layers):
        raise ValueError("Every hidden-layer size must be a positive integer.")

    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)

    if X_np.ndim != 2:
        raise ValueError("X_np must be a 2-dimensional feature matrix.")
    if len(X_np) != len(y_np):
        raise ValueError("X_np and y_np must contain the same number of rows.")
    if len(X_np) == 0:
        return np.array([], dtype=np.float32)

    # Reproducible model initialization.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_sizes, iyengar_weights=None):
            super().__init__()

            layers = []
            previous_dim = input_dim

            for layer_index, hidden_dim in enumerate(hidden_sizes):
                linear = nn.Linear(previous_dim, hidden_dim)

                if layer_index == 0 and iyengar_weights is not None:
                    iyengar_weights = np.asarray(
                        iyengar_weights, dtype=np.float32
                    ).reshape(-1)

                    if len(iyengar_weights) != input_dim:
                        raise ValueError(
                            "Length of iyengar_w must match the number of input features."
                        )

                    with torch.no_grad():
                        weight_matrix = torch.tensor(
                            iyengar_weights,
                            dtype=torch.float32,
                        ).repeat(hidden_dim, 1)
                        linear.weight.copy_(weight_matrix)
                        linear.bias.fill_(0.0)
                else:
                    # Additional hidden layers use standard Xavier initialization.
                    nn.init.xavier_uniform_(linear.weight)
                    nn.init.zeros_(linear.bias)

                layers.append(linear)
                layers.append(nn.ReLU())
                previous_dim = hidden_dim

            output_layer = nn.Linear(previous_dim, 1)

            # Preserve the original output-layer initialization.
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            layers.append(output_layer)

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(device)

    model = MLP(
        input_dim=X_np.shape[1],
        hidden_sizes=hidden_layers,
        iyengar_weights=iyengar_w,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_t).cpu().numpy().flatten()

    return predictions
