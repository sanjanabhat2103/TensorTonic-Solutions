import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("x must not be empty")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    rng = np.random.default_rng(rng)
    samples = rng.choice(x, size = (n_bootstrap, x.size), replace = True)
    boot_means = samples.mean(axis=1)
    alpha = 1 - ci
    lower, upper = np.quantile(
        boot_means,
        [alpha / 2, 1 - alpha / 2]
    )
    return boot_means, lower, upper