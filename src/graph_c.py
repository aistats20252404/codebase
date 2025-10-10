import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import hyp1f1

# -------------------------
USE_BETA = False  # Set to False for Uniform distribution
CONVERGE = True

SEED = 42

if CONVERGE:
    SAMPLE_SIZES = [1_000, 5_000, 50_000, 1_000_000]
    THETAS = [1, 2.5, 7.5, 12.5]
else:
    THETAS = [1, 2.5, 7.5, 12.5]
    SAMPLE_SIZES = [750, 1_000, 1_500, 2_000]


if USE_BETA:
    OUTFILE = f"graph_c_beta{'_converge' if CONVERGE else ''}.png"
    ALPHA = 2  # Beta distribution parameter
    BETA_PARAM = 5  # Beta distribution parameter
else:
    OUTFILE = f"graph_c_unif{'_converge' if CONVERGE else ''}.png"

rng = np.random.default_rng(SEED)


def M(theta):
    if USE_BETA:
        # MGF of Beta(α, β) using confluent hypergeometric function
        # M(t) = 1F1(α; α+β; t)
        # where 1F1 is Kummer's confluent hypergeometric function
        return hyp1f1(ALPHA, BETA_PARAM + ALPHA, theta)
    else:
        # For Uniform[0,1]: M(theta) = (e^theta - 1) / theta
        if abs(theta) < 1e-10:
            return 1.0
        return (np.exp(theta) - 1) / theta


# Prepare plotting grid with better styling
plt.style.use("seaborn-v0_8-darkgrid")

fig, axs = plt.subplots(1, len(THETAS), figsize=(5 * len(THETAS), 5), sharey=True)
axs = axs.flatten()
# Add proper spacing around plots
plt.subplots_adjust(
    left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.15, wspace=0.15
)

xgrid = np.linspace(0, 1, 1000)

for i, (t, n) in enumerate(zip(THETAS, SAMPLE_SIZES)):
    # compute n so that ratio = target_r
    achieved_ratio = M(2 * t) / (n * (M(t) ** 2))

    # sample and weights
    if USE_BETA:
        x = rng.beta(ALPHA, BETA_PARAM, size=n)
    else:
        x = rng.uniform(0, 1, size=n)

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

    if USE_BETA:
        # Target distribution after reweighting
        # This would be Beta(alpha, beta) * exp(theta*x) (unnormalized)
        # For visualization, we show the reweighted empirical distribution
        target_pdf = beta.pdf(xgrid, ALPHA, BETA_PARAM) * np.exp(t * xgrid)
        target_pdf = target_pdf / np.trapezoid(target_pdf, xgrid)  # normalize
        ax.plot(
            xgrid,
            target_pdf,
            lw=2,
            color="#A23B72",
            label=f"Beta({ALPHA},{BETA_PARAM}) × exp({t}x) (normalized)",
        )
    else:
        # For uniform: target is exp(theta*x) / integral(exp(theta*x))
        target_pdf = np.exp(t * xgrid)
        target_pdf = target_pdf / np.trapezoid(target_pdf, xgrid)  # normalize
        ax.plot(
            xgrid,
            target_pdf,
            lw=2,
            color="#A23B72",
            label=f"exp({t}x) (normalized)",
        )

    xmin = 0
    xmax = 1.0
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
