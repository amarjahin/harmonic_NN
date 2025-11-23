import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from trajectories import ho_trajectory
from harmonic_nn import HO_Net

device = "cuda" if torch.cuda.is_available() else "cpu"
omega = 2*np.pi

model = HO_Net(hidden_dim=16).to(device)
model.load_state_dict(torch.load("ho_net_weights.pth", map_location=device))

activations = {}

def get_activation(name):
    def hook(model, input, output):
        # detach to avoid tracking gradients
        activations[name] = output.detach()
    return hook

# Register hooks on the two hidden linear layers
# model.net[0] = first Linear, model.net[2] = second Linear
model.net[0].register_forward_hook(get_activation("First Linear"))
model.net[6].register_forward_hook(get_activation("Last Linear"))


model.eval()
print("Loaded saved model.")
with torch.no_grad():
    # Choose an initial condition
    x0_test = torch.tensor([0.5], device=device)
    v0_test = torch.tensor([0.5], device=device)
    t_grid = torch.linspace(0, 3.5, 1000, device=device)  # more points for smoother curves
    # Prepare inputs for the NN
    x0_rep = x0_test.repeat(t_grid.shape[0])
    v0_rep = v0_test.repeat(t_grid.shape[0])

    X_test = torch.stack([x0_rep, v0_rep, t_grid], dim=1)  # (N, 3)

    # NN predictions
    pred = model(X_test)
    x_pred, v_pred = pred[:, 0], pred[:, 1]

    # True solution from analytic formula
    x_true, v_true = ho_trajectory(x0_rep, v0_rep, t_grid, omega=omega)

# Move everything to CPU + numpy for plotting
t_np      = t_grid.detach().cpu().numpy()
x_true_np = x_true.detach().cpu().numpy()
x_pred_np = x_pred.detach().cpu().numpy()
v_true_np = v_true.detach().cpu().numpy()
v_pred_np = v_pred.detach().cpu().numpy()


plt.figure(figsize=(6, 6))
# True trajectory
plt.plot(x_true_np, v_true_np, label="True trajectory", linewidth=2)
# NN trajectory
plt.plot(x_pred_np, v_pred_np, "--", label="NN trajectory", linewidth=2)
plt.xlabel("x  (position)")
plt.ylabel("p  (momentum)")
plt.title("Phase-Space Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Move to CPU / numpy
# t_np = t_grid.detach().cpu().numpy()
layer0_act = activations["First Linear"].cpu().numpy()  # shape (N, hidden_dim)
layer6_act = activations["Last Linear"].cpu().numpy()



plt.figure(figsize=(8, 4))
num_neurons_to_plot = 16
for i in range(num_neurons_to_plot):
    plt.plot(t_np, layer0_act[:, i], label=f"neuron {i}")
plt.xlabel("t")
plt.ylabel("activation")
plt.title("First hidden layer neuron activations vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(8, 4))
for i in range(16):
    plt.plot(t_np, layer6_act[:, i], label=f"neuron {i}")
plt.xlabel("t")
plt.ylabel("activation")
plt.title("Last hidden layer neuron activations vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()