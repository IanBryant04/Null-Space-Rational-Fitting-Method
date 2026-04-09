import numpy as np
from numpy.polynomial.polynomial import polydiv, polyval
def poly_deriv(poly):
    return np.array([i * x for i, x in enumerate(poly)][1:])

def is_poly_zero(poly):
    return np.allclose(poly, 0)

def sturm_seq(poly):
    degree = len(poly) - 1
    p0 = np.array(poly, dtype=float)
    p1 = poly_deriv(p0)
    seq = [p0, p1]
    p, d = p0, p1
    while len(seq) <= degree and not is_poly_zero(d):
        _, r = polydiv(p, d)
        p, d = d, -r
        seq.append(d)
    return seq

def sturm_variation(seq, x):
    signs = [np.sign(polyval(x, p)) for p in seq]
    non_zero_signs = [s for s in signs if s != 0]
    return len([1 for s, t in zip(non_zero_signs, non_zero_signs[1:]) if s != t])

def sturm_count_zeros(poly, a, b):
   #Count distinct real zeros of polynomial in (a, b]
    seq = sturm_seq(poly)
    va = sturm_variation(seq, a)
    vb = sturm_variation(seq, b)
    return va - vb
