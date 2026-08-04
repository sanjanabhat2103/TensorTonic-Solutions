import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores = np.asarray(real_scores, dtype = float)
    fake_scores = np.asarray(fake_scores, dtype = float)
    real_mean = np.mean(real_scores, axis = 0)
    fake_mean = np.mean(fake_scores, axis = 0)
    return fake_mean - real_mean