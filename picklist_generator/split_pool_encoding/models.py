"""Vectorized physical transitions and exact reverse-mode derivatives."""
import numpy as np
from scipy.special import softmax

MODELS = ('pca', 'additive', 'on_only', 'on_off')


def addresses(bits):
    if not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 12:
        raise ValueError('Address bits must be an integer from 1 to 12.')
    return ((np.arange(2**bits)[:, None] >> np.arange(bits-1, -1, -1)) & 1)


def validate_data(data, bits=None, allow_partial=False):
    d = np.asarray(data, dtype=float)
    if d.ndim != 2 or min(d.shape) < 1 or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
        raise ValueError('Targets must be a finite matrix with values in [0, 1].')
    if allow_partial:
        inferred = max(1, (len(d)-1).bit_length()) if bits is None else bits
        addresses(inferred)
        if len(d) > 2**inferred:
            raise ValueError('More target designs than available addresses.')
        return d, inferred
    inferred = int(np.log2(len(d)))
    if len(d) != 2**inferred or inferred < 1 or (bits is not None and inferred != bits):
        raise ValueError('Target rows must equal 2**address_bits (no padding or truncation).')
    addresses(inferred)
    return d, inferred


def transitions(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != v.shape or u.ndim != 3 or u.shape[1] != 2 or not np.isfinite([u, v]).all() or np.any(u < 0) or np.any(v < 0) or np.any(u+v > 1+1e-12):
        raise ValueError('u and v must have shape (rounds, 2, positions), be nonnegative, and sum to at most one.')
    a = addresses(len(u))
    U = np.ones((len(a), u.shape[-1])); O = np.zeros_like(U); F = np.zeros_like(U)
    for r in range(len(u)):
        O += U*u[r, a[:, r]]
        F += U*v[r, a[:, r]]
        U = U*(1-u[r, a[:, r]]-v[r, a[:, r]])
    return U, O, F


def shape(model, bits, positions):
    return (2*bits+2, positions) if model == 'additive' else (bits, 2, positions, 2 if model == 'on_off' else 1)


def predict_and_pullback(x, model, bits, positions):
    """Return predictions and a vector-Jacobian product; q=occupancy, t=ON share."""
    a = addresses(bits); x = np.asarray(x).reshape(shape(model, bits, positions))
    if model == 'additive':
        # Convex combination of 1, a_r, 1-a_r, 0 covers every valid affine cube.
        features = np.column_stack((np.ones(len(a)), a, 1-a, np.zeros(len(a))))
        weights = softmax(x, axis=0); p = features @ weights
        def backward(g):
            dw = features.T @ g
            return (weights*(dw-(dw*weights).sum(axis=0))).ravel()
        return p, backward
    q = x[..., 0]; t = x[..., 1] if model == 'on_off' else np.ones_like(q)
    U = np.ones((len(a), positions)); p = np.zeros_like(U); history = []
    for r in range(bits):
        qr, tr = q[r, a[:, r]], t[r, a[:, r]]
        history.append(U)
        p += U*qr*tr; U = U*(1-qr)
    def backward(g):
        grad = np.zeros_like(x); adj_u = np.zeros_like(g)
        for r in reversed(range(bits)):
            qr, tr = q[r, a[:, r]], t[r, a[:, r]]
            dq = history[r]*(g*tr-adj_u)
            dt = history[r]*g*qr
            for b in (0, 1):
                mask = a[:, r] == b
                grad[r, b, :, 0] = dq[mask].sum(axis=0)
                if model == 'on_off':
                    grad[r, b, :, 1] = dt[mask].sum(axis=0)
            adj_u = g*qr*tr + adj_u*(1-qr)
        return grad.ravel()
    return p, backward


def physical_parameters(x, model):
    if model == 'additive':
        w = softmax(x, axis=0); bits = (len(w)-2)//2
        return dict(baseline=w[0]+w[1+bits:-1].sum(axis=0), basis=w[1:1+bits]-w[1+bits:-1])
    q = x[..., 0]; t = x[..., 1] if model == 'on_off' else np.ones_like(q)
    return dict(u=q*t, v=q*(1-t))
