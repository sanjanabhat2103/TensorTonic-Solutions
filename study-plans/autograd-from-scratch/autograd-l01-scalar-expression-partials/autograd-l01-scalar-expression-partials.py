import numpy as np

def scalar_expression_partials(a, b, c, h):
    def f(x, y, z):
        return x * y + z    
    val = f(a, b, c)
    da = (f(a + h, b, c) - val) / h
    db = (f(a, b + h, c) - val) / h
    dc = (f(a, b, c + h) - val) / h
    return val, da, db, dc