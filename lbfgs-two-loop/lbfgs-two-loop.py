def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    q = grad[:]
    m = len(s_list)
    alpha = [0] * m
    rho = []
    for s, y in zip(s_list, y_list):
        sy = _dot(s, y)
        rho.append(1 / sy if sy != 0 else 0)
    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [qi - alpha[i] * yi for qi, yi in zip(q, y_list[i])]
    if m > 0:
        sy = _dot(s_list[-1], y_list[-1])
        yy = _dot(y_list[-1], y_list[-1])
        gamma = sy / yy if yy != 0 else 1.0
    else:
        gamma = 1.0
    r = [gamma * qi for qi in q]
    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [ri + s_i * (alpha[i] - beta) 
             for ri, s_i in zip(r, s_list[i])]
    return [-ri for ri in r]