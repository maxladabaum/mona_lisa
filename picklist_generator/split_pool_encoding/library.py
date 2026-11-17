"""Learned shared basis libraries, with branch coefficients shared across sites."""
import numpy as np
from scipy.special import expit, softmax
from .models import addresses, predict_and_pullback


class BasisLibrary:
    def __init__(self, model, bits, positions, k):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= 256:
            raise ValueError('Shared basis size K must be an integer from 1 to 256.')
        self.model, self.bits, self.m, self.k = model, bits, positions, k
        if model == 'additive':
            self.shapes = [(positions,), (bits, k), (k, positions)]
        else:
            self.shapes = [(2*bits, k+1), (k, positions, 3)] if model == 'on_off' else [(2*bits, k+1), (k, positions)]
        self.size = sum(int(np.prod(s)) for s in self.shapes)

    def unpack(self, x):
        parts = []; offset = 0
        for shape in self.shapes:
            n = int(np.prod(shape)); parts.append(np.asarray(x)[offset:offset+n].reshape(shape)); offset += n
        return parts

    def initial(self, rng):
        return rng.normal(0, .5, self.size)

    def pca_initial(self, data, rng):
        """Translate PCA patterns into a valid starting library, not a recipe.

        Signed principal directions are used directly for the additive reference.
        Physical libraries use clipped mean +/- principal variations as nonnegative
        profiles. All branch coefficients and profiles are subsequently optimized.
        """
        from scipy.special import logit
        data = np.asarray(data, float)
        mean = data.mean(axis=0)
        _, singular, components = np.linalg.svd(data-mean, full_matrices=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1.)*1e-12))
        parts = self.unpack(self.initial(rng))
        if self.model == 'additive':
            parts[0][:] = logit(np.clip(mean, 1e-5, 1-1e-5))
            directions = components[:max(rank, 1)]
            parts[2][:] = directions[np.arange(self.k) % len(directions)]
            parts[1][:] *= .1
        else:
            candidates = [mean]
            for direction, value in zip(components[:rank], singular[:rank]):
                variation = direction*value/np.sqrt(max(len(data)-1, 1))
                candidates.extend((mean+variation, mean-variation))
            # Extra vectors start from PCA reconstructions of actual targets.
            reconstructed = mean+(data-mean)@components[:rank].T@components[:rank]
            candidates.extend(reconstructed)
            candidates = np.asarray(candidates)
            profiles = np.clip(candidates[np.arange(self.k) % len(candidates)], 1e-5, 1-1e-5)
            if self.model == 'on_only':
                steps = 1 if getattr(self, 'wells', None) is not None else self.bits
                profiles = -np.expm1(np.log1p(-profiles)/steps)
                parts[1][:] = logit(np.clip(profiles, 1e-5, 1-1e-5))
            else:
                probabilities = np.stack((.5*profiles, .5*(1-profiles), np.full_like(profiles, .5)), axis=-1)
                parts[1][:] = np.log(probabilities)
        return np.concatenate([part.ravel() for part in parts])

    def forward(self, x):
        parts = self.unpack(x)
        if self.model == 'additive':
            center_logits, coefficients, basis = parts
            center = expit(center_logits); raw = coefficients @ basis
            span = 1+np.abs(raw).sum(axis=0); radius = np.minimum(center, 1-center)
            scale = 2*radius/span; a = addresses(self.bits)-.5; z = a@raw
            p = center+z*scale
            def backward(g):
                gz = np.sum(g*z, axis=0)
                dw = a.T@(g*scale)-np.sign(raw)*(gz*2*radius/span**2)
                dc = (g.sum(axis=0)+gz*2*np.where(center < .5, 1., -1.)/span)*center*(1-center)
                return np.concatenate((dc.ravel(), (dw@basis.T).ravel(), (coefficients.T@dw).ravel()))
            return p, backward
        coefficient_logits, basis_logits = parts
        weights = softmax(coefficient_logits, axis=1)
        basis = softmax(basis_logits, axis=-1) if self.model == 'on_off' else expit(basis_logits)
        if self.model == 'on_off':
            mixture = (weights[:, :self.k]@basis[..., :2].reshape(self.k, -1)).reshape(self.bits, 2, self.m, 2)
            u, v = mixture[..., 0], mixture[..., 1]; q = u+v
            t = np.divide(u, q, out=np.zeros_like(q), where=q>0)
            physical = np.stack((q, t), axis=-1)
        else:
            physical = (weights[:, :self.k]@basis).reshape(self.bits, 2, self.m, 1)
        p, pullback = predict_and_pullback(physical, self.model, self.bits, self.m)
        def backward(g):
            grad = pullback(g).reshape(physical.shape)
            if self.model == 'on_off':
                qsafe = np.maximum(q, 1e-100)
                du = grad[..., 0]+grad[..., 1]*v/qsafe**2
                dv = grad[..., 0]-grad[..., 1]*u/qsafe**2
                dm = np.stack((du, dv), axis=-1).reshape(2*self.bits, -1)
                db = np.zeros_like(basis)
                db[..., :2] = (weights[:, :self.k].T@dm).reshape(self.k, self.m, 2)
                db = basis*(db-(db*basis).sum(axis=-1, keepdims=True))
                dw = dm@basis[..., :2].reshape(self.k, -1).T
            else:
                dm = grad.reshape(2*self.bits, self.m)
                db = (weights[:, :self.k].T@dm)*basis*(1-basis)
                dw = dm@basis.T
            dw = np.column_stack((dw, np.zeros(2*self.bits)))
            dw = weights*(dw-(dw*weights).sum(axis=1, keepdims=True))
            return np.concatenate((dw.ravel(), db.ravel()))
        return p, backward

    def describe(self, x):
        parts = self.unpack(x)
        if self.model == 'additive':
            center, coefficients, basis = parts
            c = expit(center); raw = coefficients@basis
            effective_basis = basis*(2*np.minimum(c, 1-c)/(1+np.abs(raw).sum(axis=0)))
            return dict(basis_vectors=effective_basis, branch_coefficients=coefficients,
                        baseline=c-.5*(coefficients@effective_basis).sum(axis=0))
        weights = softmax(parts[0], axis=1)
        basis = softmax(parts[1], axis=-1) if self.model == 'on_off' else expit(parts[1])
        return dict(basis_vectors=basis, branch_coefficients=weights[:, :self.k], vacant_weight=weights[:, -1])
