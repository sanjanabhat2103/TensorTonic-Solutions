import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    X = np.asarray(X, dtype = float)
    h = np.asarray(h_0, dtype = float)
    batch_size, seq_len, _ = X.shape
    hidden_dim = h.shape[1]
    h_states = np.zeros((batch_size, seq_len, hidden_dim))
    for t in range(seq_len):
        x_t = X[:, t, :]
        h = np.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)
        h_states[:, t, :] = h
    return h_states, h