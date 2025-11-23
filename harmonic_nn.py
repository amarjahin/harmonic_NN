import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from trajectories import ho_trajectory, generate_dataset


class HO_Net(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim+16),
            nn.Tanh(),
            nn.Linear(hidden_dim+16, hidden_dim+16),
            nn.Tanh(),
            nn.Linear(hidden_dim+16, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x: (..., 3) -> (..., 2)
        return self.net(x)

def model_train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    n_epochs=2000,
    batch_size=1024,
    lr=1e-3,
    device="cpu"
    ):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, y_val.to(device)).item()

        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{n_epochs}  "
                f"Train MSE: {train_loss:.3e}  Val MSE: {val_loss:.3e}")
    return None



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)

    omega = 2*np.pi
    # Generate data
    X, y = generate_dataset(
        n_samples=50_000,
        t_min=0.0,
        t_max=3.5,
        x0_range=(-1.0, 1.0),
        v0_range=(-1.0, 1.0),
        omega=omega,
        device=device,
    )

    # X_2, y_2 = generate_dataset(
    #     n_samples=50_000,
    #     t_min=5.0,
    #     t_max=10.0,
    #     x0_range=(-1.0, 1.0),
    #     v0_range=(-1.0, 1.0),
    #     omega=omega,
    #     device=device,
    # )

    # X,y = torch.cat([X_1, X_2], dim=0), torch.cat([y_1, y_2], dim=0)

    # Train/validation split
    n_train = int(0.8 * X.size(0))
    perm = torch.randperm(X.size(0), device=device)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_val,   y_val   = X[perm[n_train:]], y[perm[n_train:]]

    model = HO_Net(hidden_dim=16)

    print("Training on", device)
    model_train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        n_epochs=4000,
        batch_size=128,
        lr=5e-4,
        device=device,
    )

    torch.save(model.state_dict(), "ho_net_weights.pth")
    print("Model saved to ho_net_weights.pth")
    # return model


if __name__ == "__main__":
    main()
