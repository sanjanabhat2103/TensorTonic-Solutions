import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    a, b, c, f, h = map(np.float64, (a, b, c, f, h))
    e = a * b + c
    L = e * f
    analytic = [f * b, f * a, f, e]
    def loss(pa, pb, pc, pf):
        return (pa * pb + pc) * pf
    numerical = [
        (loss(a + h, b, c, f) - L) / h,
        (loss(a, b + h, c, f) - L) / h,
        (loss(a, b, c + h, f) - L) / h,
        (loss(a, b, c, f + h) - L) / h
    ]
    max_diff = np.max(np.abs(np.array(analytic) - np.array(numerical)))
    return float(L), [float(x) for x in analytic], [float(x) for x in numerical], float(max_diff)
