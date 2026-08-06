import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        """
        Forward pass through entire sequence.
        Returns (y_seq, h_final).
        """
        X = np.asarray(X, dtype = float)
        batch_size, seq_len, _ = X.shape
        if h_0 is None:
            h = np.zeros((batch_size, self.hidden_dim))
        else:
            h = np.asarray(h_0, dtype = float)
        y_seq = []
        h_seq = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            y_t = h @ self.W_hy.T + self.b_y
            h_seq.append(h.copy())
            y_seq.append(y_t)
        y_seq = np.stack(y_seq, axis = 1)
        return y_seq, h