import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
import torch

# -------------------------------
# Parameters
# -------------------------------
torch.manual_seed(42)  # Set seed for reproducibility
np.random.seed(42)
device = torch.device("cuda:1")
print(f"Using device: {device}")

alpha, beta_param = 2, 5  # Beta distribution parameters
theta = 50  # Exponential tilting parameter
n_samples = 500_000_000  # Number of samples
BINS = 200

# -------------------------------
# 1. Sample from original Beta
# -------------------------------
# Use torch.distributions for faster sampling
beta_dist = torch.distributions.Beta(alpha, beta_param)
X = beta_dist.sample((n_samples,)).to(device)

# -------------------------------
# 2. Compute importance weights
# -------------------------------
weights = torch.exp(theta * X)
M_theta = torch.mean(weights)  # Normalization factor
weights_normalized = weights / M_theta

# Convert to numpy for plotting
X_np = X.cpu().numpy()
weights_normalized_np = weights_normalized.cpu().numpy()
M_theta_np = M_theta.cpu().item()  # Convert to Python scalar


# -------------------------------
# 3. Tilted PDF (normalized)
# -------------------------------
def tilted_pdf(x):
    return beta.pdf(x, alpha, beta_param) * np.exp(theta * x) / M_theta_np


x_vals = np.linspace(0, 1, 500)
pdf_vals = tilted_pdf(x_vals)

# -------------------------------
# 4. Plot
# -------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Tilted Beta distribution
ax1 = axes[0]
ax1.hist(
    X_np,
    bins=BINS,
    weights=weights_normalized_np,
    density=True,
    alpha=0.6,
    color="#2E86AB",
    label="Weighted samples",
)
ax1.plot(x_vals, pdf_vals, lw=2, color="#A23B72", label="Tilted PDF")
ax1.set_xlabel("x", fontsize=16, fontweight="bold")
ax1.set_ylabel("Density", fontsize=16, fontweight="bold")
ax1.set_xlim(0.5, 1)
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3, linestyle="--")

# Right subplot: Gamma distribution
ax2 = axes[1]

transformed_samples = theta * (1 - X_np)
gamma_shape = beta_param  # shape parameter (α)
gamma_scale = 1  # scale parameter (θ)

max_y = min(max(transformed_samples), 50) / 2
x_gamma = np.linspace(0, max_y, 500)
gamma_pdf_vals = gamma.pdf(x_gamma, a=gamma_shape, scale=gamma_scale)

ax2.hist(
    transformed_samples,
    bins=BINS,
    weights=weights_normalized_np,
    density=True,
    alpha=0.6,
    color="#2E86AB",
    label="Transformed weighted samples",
)

ax2.plot(
    x_gamma,
    gamma_pdf_vals,
    lw=2,
    color="#A23B72",
    label=f"Gamma({gamma_shape}, {gamma_scale}) PDF",
)
ax2.set_xlabel("$\\theta (1 - x)$", fontsize=16, fontweight="bold")
ax2.set_ylabel("Density", fontsize=16, fontweight="bold")
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xlim(0, max_y)

plt.suptitle(
    rf"$\mathrm{{Beta}}({alpha},{beta_param})$ tilted with $\theta={theta}$",
    fontsize=20,
    fontweight="bold",
)

plt.tight_layout()
plt.savefig("graph_c3.png", dpi=200, bbox_inches="tight", pad_inches=0.3)
