import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# -------------------------


SAMPLE_SIZES = [1_000, 5_000, 20_000, 250_000]
SEED = 2025

THETAS = [2, 2.25, 2.4, 2.49]
OUTFILE = "graph_b.png"


rng = np.random.default_rng(SEED)

LAMBDA = 5


def M(theta):
    return LAMBDA / (LAMBDA - theta)


# Prepare plotting grid with better styling
plt.style.use("seaborn-v0_8-darkgrid")

fig, axs = plt.subplots(1, len(THETAS), figsize=(5 * len(THETAS), 5), sharey=True)
axs = axs.flatten()
# Add proper spacing around plots
plt.subplots_adjust(
    left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.15, wspace=0.15
)

xgrid = np.linspace(-10, 10, 1000)

for i, (t, n) in enumerate(zip(THETAS, SAMPLE_SIZES)):
    # compute n so that ratio = target_r
    achieved_ratio = M(2 * t) / (n * (M(t) ** 2))

    # sample and weights
    x = rng.exponential(scale=1 / LAMBDA, size=n)
    w_unnorm = np.exp(t * x)  # importance weights
    w = w_unnorm / np.sum(w_unnorm)

    ax = axs[i]
    ax.hist(
        x,
        bins=80,
        weights=w,
        density=True,
        alpha=0.6,
        color="#2E86AB",
        label="Reweighted samples",
    )
    ax.plot(
        xgrid,
        expon.pdf(xgrid, scale=1 / (LAMBDA - t)),
        lw=2,
        color="#A23B72",
        label=f"Exp({LAMBDA - t}) pdf",
    )

    xmin = 0
    xmax = 3.0
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("x", fontsize=18, fontweight="bold", labelpad=10)

    if i == 0:
        ax.set_ylabel("Density", fontsize=18, fontweight="bold", labelpad=10)

    # Format ratio in scientific notation
    ratio_exp = int(np.floor(np.log10(achieved_ratio)))
    ratio_mantissa = achieved_ratio / (10**ratio_exp)

    ax.set_title(
        f"$\\theta = {t}$, $n = {n:,}$\n$\\mathrm{{Ratio}} = {ratio_mantissa:.1f} \\times 10^{{{ratio_exp}}}$",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=14, pad=5)

plt.savefig(OUTFILE, dpi=200, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {OUTFILE}")
