"""Two-stage 25 nL recipes with an explicit, uncalibrated capture model.

C50 is the total staple concentration giving 50% capture in one round.
ON/OFF are assumed equally effective: q=1-exp(-ln(2)*Ctotal/C50).
This is a user-specified kinetic assumption, not a volume/probability identity.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit, softmax
from .library import BasisLibrary
from .models import predict_and_pullback, transitions


@dataclass(frozen=True)
class RecipeSettings:
    on_stock_nm: float
    off_stock_nm: float
    vector_volume_nl: float
    vector_max_nm: float
    branch_volume_nl: float
    branch_budget_nl: float
    capture_c50_nm: float
    increment_nl: float = 25.

    def validate(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('All stock, volume, and capture settings must be finite and positive.')
        if self.increment_nl != 25:
            raise ValueError('Split–pool recipes use 25 nL increments.')
        for key in ('vector_volume_nl', 'branch_volume_nl', 'branch_budget_nl'):
            steps = values[key]/self.increment_nl
            if not math.isclose(steps, round(steps), abs_tol=1e-8, rel_tol=0):
                raise ValueError(f'{key} must be a multiple of 25 nL.')
        if self.branch_budget_nl > self.branch_volume_nl:
            raise ValueError('Basis-transfer budget cannot exceed final branch volume.')
        return self


def quantize_volumes(desired, budget_nl, increment=25.):
    """Round a reagent group to whole droplets without exceeding its volume budget."""
    desired = np.asarray(desired, float)
    if not np.isfinite(desired).all() or np.any(desired < 0):
        raise ValueError('Desired transfer volumes must be finite and nonnegative.')
    capacity = int(round(budget_nl/increment))
    units = desired.ravel()/increment
    if units.sum() > capacity:
        units = units*(capacity/units.sum())
    base = np.floor(units).astype(np.int64)
    remaining = max(0, capacity-int(base.sum()))
    fractions = units-base
    eligible = np.flatnonzero(fractions >= .5)
    order = eligible[np.argsort(-fractions[eligible], kind='stable')]
    base[order[:remaining]] += 1
    return (base.reshape(desired.shape)*increment).astype(float)


class ConcentrationLibrary(BasisLibrary):
    def __init__(self, model, bits, positions, k, settings, wells=None, response_model='capture'):
        if model not in ('on_only', 'on_off'):
            raise ValueError('Concentration recipes require ON-only or ON/OFF; signed additive/PCA are reference models.')
        super().__init__(model, bits, positions, k)
        self.settings = settings.validate()
        self.wells = wells
        if response_model not in ('capture', 'rgb_ratio'):
            raise ValueError('Unknown capture response model.')
        self.response_model = response_model
        self.destinations = 2*bits if wells is None else wells
        if wells is not None:
            self.shapes[0] = (wells, k+1)
            self.size = sum(int(np.prod(s)) for s in self.shapes)

    def target_initial(self, targets, rng):
        """One vector per target when enough stocks exist; physical limits still apply."""
        from scipy.special import logit
        parts = self.unpack(self.initial(rng))
        weights = np.full((self.wells, self.k+1), 1e-8)
        weights[:, -1] = .05
        weights[np.arange(self.wells), np.arange(self.wells)] = .95
        weights /= weights.sum(axis=1, keepdims=True)
        parts[0][:] = np.log(weights)
        if self.response_model == 'rgb_ratio' and self.model == 'on_off':
            profile = np.clip(targets, 1e-6, 1-1e-6)
            parts[1][:self.wells] = np.log(np.stack((.999*profile, .999*(1-profile), np.full_like(profile, .001)), axis=-1))
            return np.concatenate([part.ravel() for part in parts])
        scale = self.settings.vector_max_nm*self.settings.branch_budget_nl/self.settings.branch_volume_nl
        concentration = -np.log1p(-np.clip(targets, 0, 1-1e-6))*self.settings.capture_c50_nm/np.log(2)
        profile = np.clip(concentration/(scale*.95), 1e-6, 1-2e-6)
        if self.model == 'on_only':
            parts[1][:self.wells] = logit(profile)
        else:
            parts[1][:self.wells] = np.log(np.stack((profile, np.full_like(profile, 1e-6), 1-profile-1e-6), axis=-1))
        return np.concatenate([part.ravel() for part in parts])

    def forward(self, x):
        logits, basis_logits = self.unpack(x)
        weights = softmax(logits, axis=1)
        basis = softmax(basis_logits, axis=-1) if self.model == 'on_off' else expit(basis_logits)
        profiles = basis[..., :2] if self.model == 'on_off' else basis[..., None]
        scale = self.settings.vector_max_nm*self.settings.branch_budget_nl/self.settings.branch_volume_nl
        geometry = (self.bits, 2, self.m) if self.wells is None else (self.wells, self.m)
        concentrations = (weights[:, :self.k]@profiles.reshape(self.k, -1)*scale).reshape(*geometry, -1)
        total = concentrations.sum(axis=-1); beta = np.log(2)/self.settings.capture_c50_nm
        q = (total > 0).astype(float) if self.response_model == 'rgb_ratio' else -np.expm1(-beta*total)
        if self.model == 'on_off':
            t = np.divide(concentrations[..., 0], total, out=np.zeros_like(total), where=total>0)
            physical = np.stack((q, t), axis=-1)
        else:
            physical = q[..., None]
        if self.wells is None:
            p, pullback = predict_and_pullback(physical, self.model, self.bits, self.m)
        else:
            p = q*t if self.model == 'on_off' else q
            def pullback(g):
                return np.stack((g*t, g*q), axis=-1) if self.model == 'on_off' else g[..., None]
        def backward(g):
            grad = pullback(g).reshape(physical.shape)
            capture_grad = np.zeros_like(total) if self.response_model == 'rgb_ratio' else grad[..., 0]*beta*np.exp(-beta*total)
            if self.model == 'on_off':
                safe = np.maximum(total, 1e-100)
                dc = np.stack((capture_grad+grad[..., 1]*concentrations[..., 1]/safe**2,
                               capture_grad-grad[..., 1]*concentrations[..., 0]/safe**2), axis=-1)
            else:
                dc = capture_grad[..., None]
            dm = dc.reshape(self.destinations, -1)*scale
            dw = np.column_stack((dm@profiles.reshape(self.k, -1).T, np.zeros(self.destinations)))
            dw = weights*(dw-(dw*weights).sum(axis=1, keepdims=True))
            db = (weights[:, :self.k].T@dm).reshape(profiles.shape)
            if self.model == 'on_off':
                db = np.concatenate((db, np.zeros((*db.shape[:-1], 1))), axis=-1)
                db = basis*(db-(db*basis).sum(axis=-1, keepdims=True))
            else:
                db = db[..., 0]*basis*(1-basis)
            return np.concatenate((dw.ravel(), db.ravel()))
        return p, backward

    def recipe(self, x):
        library = self.describe(x); b = library['basis_vectors']; s = self.settings
        profiles = b[..., :2] if self.model == 'on_off' else np.stack((b, np.zeros_like(b)), axis=-1)
        desired = profiles*s.vector_max_nm*s.vector_volume_nl/np.array([s.on_stock_nm, s.off_stock_nm])
        source = np.array([quantize_volumes(v, s.vector_volume_nl) for v in desired])
        branch = np.array([quantize_volumes(w*s.branch_budget_nl, s.branch_budget_nl)
                           for w in library['branch_coefficients']])
        # A vector is prepared in one well: all branch withdrawals share its stock.
        for k in range(self.k):
            if branch[:, k].sum() > s.vector_volume_nl:
                branch[:, k] = quantize_volumes(branch[:, k], s.vector_volume_nl)
        return self.complete_recipe(source, branch)

    def complete_recipe(self, source, branch):
        s = self.settings
        source, branch = np.asarray(source, float), np.asarray(branch, float)
        if source.shape != (self.k, self.m, 2) or branch.shape != (self.destinations, self.k):
            raise ValueError('Recipe arrays do not match the library and rounds.')
        for volumes in (source, branch):
            if not np.isfinite(volumes).all() or np.any(volumes < 0) or np.any(volumes % 25):
                raise ValueError('Recipe transfers must be nonnegative multiples of 25 nL.')
        if (np.any(source.sum(axis=(1, 2)) > s.vector_volume_nl)
                or np.any(branch.sum(axis=1) > s.branch_budget_nl)
                or np.any(branch.sum(axis=0) > s.vector_volume_nl)):
            raise ValueError('Recipe exceeds a preparation, branch, or source-well volume budget.')
        if self.model == 'on_only' and np.any(source[..., 1]):
            raise ValueError('ON-only recipes cannot include OFF staples.')
        stocks = source*np.array([s.on_stock_nm, s.off_stock_nm])/s.vector_volume_nl
        geometry = (self.bits, 2, self.m, 2) if self.wells is None else (self.wells, self.m, 2)
        concentrations = (branch@stocks.reshape(self.k, -1)/s.branch_volume_nl).reshape(geometry)
        total = concentrations.sum(axis=-1)
        q = (total > 0).astype(float) if self.response_model == 'rgb_ratio' else -np.expm1(-np.log(2)*total/s.capture_c50_nm)
        ratio = np.divide(concentrations[..., 0], total, out=np.zeros_like(total), where=total>0)
        u, v = q*ratio, q*(1-ratio)
        return dict(settings=asdict(s), response_model=self.response_model, synthesis_mode='split_pool' if self.wells is None else 'independent_wells', source_transfers_nl=source, branch_transfers_nl=branch,
                    vector_buffer_nl=s.vector_volume_nl-source.sum(axis=(1, 2)),
                    branch_remaining_nl=s.branch_volume_nl-branch.sum(axis=1),
                    vector_concentrations_nm=stocks, branch_concentrations_nm=concentrations, u=u, v=v,
                    vector_remaining_nl=s.vector_volume_nl-branch.sum(axis=0))

    def refine_recipe(self, x, targets, flexible, cancel=None, probes=64, progress=None, allowed_addresses=None):
        if progress is not None:
            progress(0.)
        recipe = self.recipe(x)
        def score(candidate):
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Analysis cancelled.')
            p = transitions(candidate['u'], candidate['v'])[1] if self.wells is None else candidate['u']
            eligible = np.arange(len(p)) if allowed_addresses is None else np.asarray(allowed_addresses)
            used, pi = (linear_sum_assignment(cdist(p[eligible], targets, metric='sqeuclidean')) if flexible
                        else (np.arange(len(targets)), np.arange(len(targets))))
            used = eligible[used]
            return float(np.mean((p[used]-targets[pi])**2)), p, used, pi
        best = score(recipe); projected_mse = best[0]
        # Deterministic local one-droplet refinement, with the global assignment
        # updated for each candidate. This is bounded search, not an exact MILP.
        rng = np.random.default_rng(0)
        for step in range(probes):
            if progress is not None:
                progress(step/max(probes, 1))
            source = recipe['source_transfers_nl'].copy(); branch = recipe['branch_transfers_nl'].copy()
            grid = source if step % 2 == 0 else branch
            flat_index = int(rng.integers(grid.size)); index = np.unravel_index(flat_index, grid.shape)
            if grid is source and self.model == 'on_only' and index[-1] == 1:
                continue
            grid[index] += 25 if rng.integers(2) else -25
            limit = self.settings.vector_volume_nl if grid is source else self.settings.branch_budget_nl
            if grid[index] < 0 or grid[index[0]].sum() > limit or (grid is branch and branch[:, index[1]].sum() > self.settings.vector_volume_nl):
                continue
            candidate = self.complete_recipe(source, branch); scored = score(candidate)
            if scored[0] < best[0]:
                recipe, best = candidate, scored
        if progress is not None:
            progress(1.)
        return recipe, best, projected_mse
