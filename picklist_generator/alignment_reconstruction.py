"""Residual rigid alignment errors, applied before decoded shifts.

The sampled mean and within-design covariance give expected squared image error
for N independent origami per design without treating sites as independent jitter.
"""
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
import math
import numpy as np
from PIL import Image
from optimization_regularization import regularization_weight
from alignment_chunks import ChunkedMean, MeanBlock, raster_patches, CHUNK_EDGE, KERNEL_CUTOFF_SIGMA, MAX_CACHED_ELEMENTS
from color_light import grayscale_signal, display_rgb, abundance_scale, canvas_area_gain, quadrant_weight
from gaussian_reconstruction import GaussianReconstruction, REFERENCE_AREA_NM2
from image_encoding import geometry, transfer_fraction_steps, set_site_fraction
from template_generator import EXTENSION_COLUMN_SPACING_NM, EXTENSION_ROW_SPACING_NM, EXTENSION_EXTRA_COLUMN_GAP_NM

DEFAULT_CENTER_X = -EXTENSION_COLUMN_SPACING_NM / 2 - EXTENSION_EXTRA_COLUMN_GAP_NM / 2
DEFAULT_CENTER_Y = 7 * EXTENSION_ROW_SPACING_NM / 2
TRAIN_SEED = 1729
VALIDATION_SEED = 2718
ERROR_SAMPLES = 32


class AlignmentCapacityError(ValueError):
    """The requested output or local kernel cache exceeds the memory budget."""


@dataclass(frozen=True)
class AlignmentSettings:
    translation_x_nm: float = 3.0
    translation_y_nm: float = 3.0
    rotation_deg: float = 3.0
    origami_per_design: int = 20
    center_x_nm: float = DEFAULT_CENTER_X
    center_y_nm: float = DEFAULT_CENTER_Y
    variability_penalty: float = 0.0

    def validate(self):
        for name in ('translation_x_nm', 'translation_y_nm', 'rotation_deg', 'variability_penalty'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Alignment errors and variability penalty must be finite and nonnegative.')
        if not all(math.isfinite(v) for v in (self.center_x_nm, self.center_y_nm)):
            raise ValueError('Alignment center coordinates must be finite.')
        if isinstance(self.origami_per_design, bool) or int(self.origami_per_design) != self.origami_per_design or not 1 <= self.origami_per_design <= 10000:
            raise ValueError('Origami count per design must be an integer from 1 to 10000.')


def transformed_sites(design, settings, error):
    """Error = [dx nm, dy nm, clockwise degrees] in aligned origami coordinates."""
    shift = np.array([design['x_shift_nm'], design['y_shift_nm']])
    points = np.array([[s['x_nm'], s['y_nm']] for s in design['image_sites']]) - shift
    center = np.array([settings.center_x_nm, settings.center_y_nm]) - np.asarray(design.get("active_canvas_offset_nm", [0, 0]))
    angle = math.radians(error[2])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return (points-center) @ rotation.T + center + np.asarray(error[:2]) + shift


def alignment_raster_shape(info, fwhm_nm):
    if not math.isfinite(fwhm_nm) or fwhm_nm <= 0:
        raise ValueError('Gaussian FWHM must be finite and positive.')
    area = info['field_width_nm'] * info['field_height_nm']
    step = min(max(.75, math.sqrt(area / 18000)), fwhm_nm / 4)
    return (math.ceil(info['field_width_nm']/step-1e-9),
            math.ceil(info['field_height_nm']/step-1e-9))


def needs_chunked_alignment(designs, fwhm_nm):
    info = geometry(designs[0]['encoding_scheme'],
                    designs[0].get('origami_rotation_degrees_clockwise', 0),
                    designs[0].get('active_image_columns'))
    width, height = alignment_raster_shape(info, fwhm_nm)
    variables = sum(len(d['image_sites']) for d in designs if d['channel'] == 0)
    return width*height*variables > 16_000_000 or width*height > 80000


class AlignmentModel:
    """Fixed sampled forward model shared by preview, optimization and evaluation."""
    def __init__(self, designs, settings, fwhm_nm=7.5, exposure=1.0, seed=TRAIN_SEED, linear_light=False, normalize_designs=False, compensate_area=False, checkpoint=None, chunked=None, progress=None):
        settings.validate()
        if not designs or not math.isfinite(fwhm_nm) or fwhm_nm <= 0 or not math.isfinite(exposure) or exposure <= 0:
            raise ValueError('Valid designs, positive Gaussian width and exposure are required.')
        self.compensate_area = compensate_area
        self.linear_light, self.normalize_designs = linear_light, normalize_designs
        self.settings, self.fwhm, self.exposure, self.seed = settings, fwhm_nm, exposure, seed
        self.rotation = designs[0].get('origami_rotation_degrees_clockwise', 0)
        self.scheme = designs[0]['encoding_scheme']
        if any(d['encoding_scheme'] != self.scheme or d.get('origami_rotation_degrees_clockwise',0) != self.rotation for d in designs):
            raise ValueError('Alignment models require a single scheme and orientation.')
        self.info = geometry(self.scheme, self.rotation, designs[0].get('active_image_columns'))
        # Larger virtual mosaics need not use the single-origami 0.75 nm raster.
        # Still resolve FWHM with at least four samples and record the true pitch.
        self.width, self.height = alignment_raster_shape(self.info, fwhm_nm)
        if self.width*self.height > 2_000_000:
            raise AlignmentCapacityError('Alignment output raster exceeds two million pixels. Increase Gaussian FWHM to reduce the required resolution.')
        self.dx = self.info['field_width_nm']/self.width
        self.dy = self.info['field_height_nm']/self.height
        self.xs = -self.info['pixel_pitch_x_nm']/2 + (np.arange(self.width)+.5)*self.dx
        self.ys = -self.info['pixel_pitch_y_nm']/2 + (np.arange(self.height)+.5)*self.dy
        self.sigma = fwhm_nm / math.sqrt(8*math.log(2))
        self.scale = abundance_scale(self.scheme, normalize_designs)
        self.area_gain = canvas_area_gain(self.scheme, compensate_area)
        self.gain = REFERENCE_AREA_NM2*exposure*self.scale*self.area_gain
        self.channel_count = self.info['channel_count']
        self.groups = [[d for d in designs if d['channel']==c] for c in range(self.channel_count)]
        keys = [[(d['x_shift'],d['y_shift']) for d in group] for group in self.groups]
        if not keys[0] or any(key != keys[0] for key in keys[1:]):
            raise ValueError('Alignment simulation requires matching color/shift designs.')
        self.signature = [[(d['name'], [(s['name'],s['x_nm'],s['y_nm']) for s in d['image_sites']]) for d in group] for group in self.groups]
        rng = np.random.default_rng(seed)
        errors = rng.normal(size=(ERROR_SAMPLES//2,3))
        errors = np.concatenate((errors,-errors))  # antithetic zero-mean quadrature
        errors *= [settings.translation_x_nm, settings.translation_y_nm, settings.rotation_deg]
        variables = sum(len(d['image_sites']) for d in self.groups[0])
        dense_elements = self.width * self.height * variables
        self.chunked = (dense_elements > 16_000_000 or self.width*self.height > 80000) if chunked is None else chunked
        if not self.chunked and dense_elements > MAX_CACHED_ELEMENTS:
            raise AlignmentCapacityError('Dense alignment model exceeds the memory budget; use chunked alignment.')
        self.mean_blocks, self.covariances = [], []
        self.slices = []
        start, cached_elements = 0, 0
        blocks = []
        for index, design in enumerate(self.groups[0]):
            if progress:
                progress(index / len(self.groups[0]))
            if checkpoint:
                checkpoint()
            count = len(design['image_sites'])
            columns = slice(start, start + count)
            transformed = np.array([transformed_sites(design, settings, error) for error in errors])
            patches = (raster_patches(self.xs, self.ys, transformed, self.sigma) if self.chunked else
                       [(slice(0, self.height), slice(0, self.width))])
            covariance = np.zeros((count, count))
            for y, x in patches:
                if checkpoint:
                    checkpoint()
                pixels = (y.stop-y.start) * (x.stop-x.start)
                cached_elements += pixels * count
                if cached_elements > MAX_CACHED_ELEMENTS:
                    raise AlignmentCapacityError('Alignment chunks exceed the cache budget for these error settings. Reduce alignment errors or the layout size.')
                average = np.zeros((pixels, count))
                second = np.zeros((count, count))
                for points in transformed:
                    if checkpoint:
                        checkpoint()
                    kx, ky = self.kernel_axes(points, x, y)
                    scale = quadrant_weight(design, normalize_designs) / (2*math.pi*self.sigma**2)
                    kernel = (ky[:, None, :] * kx[None, :, :]).reshape(pixels, count) * scale
                    average += kernel/len(errors)
                    second += ((kx.T @ kx) * (ky.T @ ky)) * scale**2 / len(errors)
                covariance += second - average.T @ average
                if self.chunked:
                    blocks.append(MeanBlock(y, x, columns, average))
                else:
                    self.mean_blocks.append(average)
            self.covariances.append((covariance+covariance.T)/2)
            self.slices.append(columns)
            start += count
        self.mean = (ChunkedMean(self.height, self.width, variables, blocks) if self.chunked else
                     np.concatenate(self.mean_blocks, axis=1))
        self.mean_blocks = []
        if progress:
            progress(1.0)
        self._operators = {}  # Shared by count/penalty variants of this geometry.

    @property
    def size(self):
        return (self.height,self.width) if self.rotation in (90,270) else (self.width,self.height)

    def with_settings(self, settings):
        """Reuse immutable geometry when only sample count or penalty changes."""
        settings.validate()
        if replace(settings, origami_per_design=self.settings.origami_per_design,
                   variability_penalty=self.settings.variability_penalty) != self.settings:
            raise ValueError('Alignment geometry settings differ.')
        result = copy(self)
        result.settings = settings
        return result

    def kernel_axes(self, points, x=slice(None), y=slice(None)):
        kx = np.exp(-.5*((self.xs[x, None]-points[None,:,0])/self.sigma)**2)
        ky = np.exp(-.5*((self.ys[y, None]-points[None,:,1])/self.sigma)**2)
        return kx, ky

    def kernel(self, points):
        kx, ky = self.kernel_axes(points)
        return (ky[:,None,:]*kx[None,:,:]).reshape(self.width*self.height,-1)/(2*math.pi*self.sigma**2)

    def weights(self, designs):
        groups = [[d for d in designs if d['channel']==c] for c in range(self.channel_count)]
        signature = [[(d['name'], [(s['name'],s['x_nm'],s['y_nm']) for s in d['image_sites']]) for d in group] for group in groups]
        if signature != self.signature:
            raise ValueError('Design geometry differs from the alignment model.')
        weights = np.array([[s.get('effective_brightness', s['replacement_fraction']) for d in group for s in d['image_sites']] for group in groups]).T
        if not np.all(np.isfinite(weights)) or np.any(weights<0) or any(s["replacement_fraction"] < 0 or s["replacement_fraction"] > 1 for d in designs for s in d["image_sites"]):
            raise ValueError('Replacement fractions must be in [0,1]; effective brightness must be finite and nonnegative.')
        return weights

    def result(self, local, kind, **extra):
        if self.channel_count == 1:
            local = np.repeat(local, 3, axis=2)
        density = np.rot90(local,-self.rotation//90).copy()
        linear = density*self.gain
        portrait = self.rotation in (90,270)
        metadata = dict(model='sampled_rigid_alignment_then_decoded_shift', preview_kind=kind,
            image_mode=self.info['image_mode'], channel_count=self.channel_count,
            alignment_included=True,
            alignment_backend='chunked' if self.chunked else 'dense',
            alignment_chunk_edge=CHUNK_EDGE if self.chunked else None,
            gaussian_tail_cutoff_sigma=KERNEL_CUTOFF_SIGMA if self.chunked else None,
            alignment_mean_bytes=self.mean.nbytes,
            alignment_dense_equivalent_bytes=self.mean.shape[0]*self.mean.shape[1]*8,
            fwhm_nm=self.fwhm, exposure=self.exposure, alignment=asdict(self.settings),
            linear_light=self.linear_light, normalize_designs=self.normalize_designs,
            compensate_area=self.compensate_area, canvas_area_gain=self.area_gain,
            design_abundance_scale=self.scale, linear_rgb_gain=self.gain,
            display_encoding='sRGB' if self.linear_light else 'legacy direct RGB',
            alignment_error_frame='origami-aligned axes, before decoded shift and display rotation',
            alignment_center_frame='unshifted image-site coordinates; C6 H01 is (0,0)',
            alignment_distribution='independent zero-mean Gaussian dx, dy, angle; shared rigid transform within each origami',
            alignment_center_default='center of full C12–C1 / H01–H15 staple grid, including middle column gap',
            error_samples=ERROR_SAMPLES, model_seed=self.seed, encoding_scheme=self.scheme,
            origami_rotation_degrees_clockwise=self.rotation, coordinate_frame='upright image',
            raster_columns=density.shape[1], raster_rows=density.shape[0],
            raster_pixel_x_nm=self.dy if portrait else self.dx, raster_pixel_y_nm=self.dx if portrait else self.dy,
            field_width_nm=self.info['display_field_width_nm'], field_height_nm=self.info['display_field_height_nm'],
            first_sample_x_nm=-self.info['display_pixel_pitch_x_nm']/2+(self.dy if portrait else self.dx)/2,
            first_sample_y_nm=-self.info['display_pixel_pitch_y_nm']/2+(self.dx if portrait else self.dy)/2,
            reference_area_nm2=REFERENCE_AREA_NM2, barcode_sites_included=False,
            normalization='mean per design; normalization balances unequal quadrant sampling and fixes total design weight; shared exposure, no image-dependent scaling',
            finite_count_scope='alignment uncertainty only; fractional site intensities, no occupancy or photon noise',
            clipped_pixel_fraction=float(np.mean(np.any(linear>1+1e-6,axis=2))),
            edge_handling='Gaussian tails outside the field are cropped without renormalization', **extra)
        return GaussianReconstruction(density,display_rgb(linear,self.linear_light),metadata)

    def expected(self, designs):
        local = (self.mean@self.weights(designs)).reshape(self.height,self.width,self.channel_count)
        return self.result(local,'sampled ensemble mean')

    def example(self, designs, seed=42, progress=None):
        self.weights(designs)  # geometry and intensity validation
        rng = np.random.default_rng(seed)
        local = np.zeros((self.height,self.width,self.channel_count))
        n = self.settings.origami_per_design
        for index, design in enumerate(designs):
            if progress:
                progress(index / len(designs))
            weights = np.array([s.get('effective_brightness', s['replacement_fraction']) for s in design['image_sites']]) * quadrant_weight(design,self.normalize_designs)
            errors = rng.normal(size=(n,3))*[self.settings.translation_x_nm,self.settings.translation_y_nm,self.settings.rotation_deg]
            if not np.any(weights):
                continue
            # Batch independent origami into one matrix product, preserving the
            # random draws and shared rigid transform of every individual origami.
            shift = np.array([design['x_shift_nm'], design['y_shift_nm']])
            center = np.array([self.settings.center_x_nm, self.settings.center_y_nm]) - np.asarray(design.get('active_canvas_offset_nm', [0, 0]))
            relative = np.array([[s['x_nm'], s['y_nm']] for s in design['image_sites']]) - shift - center
            for start in range(0, n, 32):
                if progress:
                    progress((index + start / n) / len(designs))
                batch = errors[start:start+32]
                angles = np.deg2rad(batch[:, 2])[:, None]
                cosine, sine = np.cos(angles), np.sin(angles)
                x = cosine*relative[:, 0] - sine*relative[:, 1] + center[0] + shift[0] + batch[:, 0, None]
                y = sine*relative[:, 0] + cosine*relative[:, 1] + center[1] + shift[1] + batch[:, 1, None]
                points = np.stack((x, y), axis=-1).reshape(-1, 2)
                batch_weights = np.tile(weights, len(batch))
                patches = (raster_patches(self.xs, self.ys, points, self.sigma) if self.chunked else
                           [(slice(0, self.height), slice(0, self.width))])
                for y_patch, x_patch in patches:
                    if progress:
                        progress((index + start / n) / len(designs))
                    kx, ky = self.kernel_axes(points, x_patch, y_patch)
                    local[y_patch, x_patch, design['channel']] += (ky*batch_weights) @ kx.T / (n*2*math.pi*self.sigma**2)
        if progress:
            progress(1.0)
        return self.result(local,'finite-count alignment realization',example_seed=seed)

    def evaluate(self, designs, target):
        weights = self.weights(designs)
        target = np.asarray(target)
        if self.channel_count == 1:
            target = grayscale_signal(target)[..., :1]
        local_target = np.rot90(target,self.rotation//90).reshape(-1,self.channel_count)
        bias = float(np.mean((self.gain*(self.mean@weights)-local_target)**2))
        variance = sum(float(np.sum(weights[sl]*(cov@weights[sl]))) for sl,cov in zip(self.slices,self.covariances))
        variance = max(0.,variance*self.gain**2/(self.settings.origami_per_design*local_target.size))
        return dict(mean_image_mse=bias, alignment_variance=variance,
                    expected_mse=bias+variance,
                    objective_mse=bias+(1+self.settings.variability_penalty)*variance)

    def optimize(self, designs, target, regularization_strength=0.0, checkpoint=None):
        target = np.asarray(target,dtype=float)
        if target.shape != (self.size[1],self.size[0],3) or not np.all(np.isfinite(target)) or np.any((target<0)|(target>1)):
            raise ValueError('Target must be finite RGB in [0,1] on the alignment raster.')
        if self.channel_count == 1:
            target = grayscale_signal(target)[..., :1]
        weights = self.weights(designs)
        result = deepcopy(designs)
        alpha = regularization_weight(regularization_strength, self.width*self.height, weights.shape[0])
        # Expected squared error = error of mean + variance / number of origami.
        factor = self.gain**2*(1+self.settings.variability_penalty)/self.settings.origami_per_design
        def apply_hessian(v):
            product = self.gain**2*(self.mean.T@(self.mean@v)) + alpha*v
            for sl,cov in zip(self.slices,self.covariances):
                product[sl] += factor*(cov@v[sl])
            return product
        # Nine-bit grids have thousands of variables: avoid a quadratic-size Hessian.
        hessian = None
        if weights.shape[0] <= 1500:
            if 'gram' not in self._operators:
                self._operators['gram'] = self.gain**2*(self.mean.T@self.mean)
            hessian = self._operators['gram'].copy()
            hessian.flat[::hessian.shape[0]+1] += alpha
            for sl,cov in zip(self.slices,self.covariances):
                hessian[sl,sl] += factor*cov
            lipschitz = float(np.max(np.sum(np.abs(hessian),axis=1)))
            diagonal = np.diag(hessian)
        else:
            rows = self.gain**2*(self.mean.T@(self.mean@np.ones(weights.shape[0]))) + alpha
            diagonal = self.gain**2*(self.mean.column_squared_norms() if self.chunked else
                                     np.einsum('ij,ij->j',self.mean,self.mean)) + alpha
            for sl,cov in zip(self.slices,self.covariances):
                rows[sl] += factor*np.sum(np.abs(cov),axis=1)
                diagonal[sl] += factor*np.diag(cov)
            lipschitz = float(rows.max())
        local_target = np.rot90(target,self.rotation//90).reshape(-1,self.channel_count)
        b = self.gain*(self.mean.T@local_target) + alpha*weights
        # The cached model describes geometry, which can be reused after a
        # volume or stock change. Quantization must use the current designs.
        current_groups = [[d for d in designs if d['channel'] == c] for c in range(self.channel_count)]
        steps = np.array([[transfer_fraction_steps(d, s) for d in group for s in d['image_sites']] for group in current_groups]).T
        if np.any(steps<1) or np.any(steps!=np.rint(steps)):
            raise ValueError('Alignment optimization requires 25 nL transfer increments.')
        w = weights.copy()
        accelerated,t = w.copy(),1.
        for _ in range(250 if hessian is not None else 100):
            if checkpoint:
                checkpoint()
            gradient = hessian@accelerated if hessian is not None else apply_hessian(accelerated)
            new = np.clip(accelerated-(gradient-b)/lipschitz,0,1)
            next_t = (1+math.sqrt(1+4*t*t))/2
            accelerated = new+(t-1)/next_t*(new-w)
            w,t = new,next_t
        best = weights.copy()
        for channel in range(self.channel_count):
            def score(v):
                return float(v@(hessian@v if hessian is not None else apply_hessian(v))-2*v@b[:,channel])
            candidates = [weights[:,channel].copy(),np.rint(w[:,channel]*steps[:,channel])/steps[:,channel]]
            for candidate in candidates:
                if hessian is None:
                    residual = self.gain*(self.mean@candidate)-local_target[:,channel]
                    var_gradient = np.zeros_like(candidate)
                    groups = {}
                    for sl,cov in zip(self.slices,self.covariances):
                        var_gradient[sl] = factor*(cov@candidate[sl])
                        for i in range(sl.start,sl.stop):
                            groups[i] = (sl,cov,i-sl.start)
                else:
                    gradient = hessian@candidate-b[:,channel]
                for _ in range(10 if hessian is not None else 4):
                    if checkpoint:
                        checkpoint()
                    changed = False
                    for i in range(len(candidate)):
                        if diagonal[i] < 1e-20:
                            continue
                        derivative = gradient[i] if hessian is not None else self.gain*(self.mean.column_dot(i, residual) if self.chunked else self.mean[:,i]@residual)+var_gradient[i]+alpha*(candidate[i]-weights[i,channel])
                        value = np.clip(np.rint((candidate[i]-derivative/diagonal[i])*steps[i,channel])/steps[i,channel],0,1)
                        delta = value-candidate[i]
                        if delta and 2*delta*derivative+delta*delta*diagonal[i] < -1e-12:
                            candidate[i] = value
                            if hessian is not None:
                                gradient += delta*hessian[:,i]
                            else:
                                if self.chunked:
                                    self.mean.add_column(residual, i, delta*self.gain)
                                else:
                                    residual += delta*self.gain*self.mean[:,i]
                                sl,cov,j = groups[i]
                                var_gradient[sl] += delta*factor*cov[:,j]
                            changed = True
                    if not changed:
                        break
            best[:,channel] = min([weights[:,channel]]+candidates,key=score)
        for channel in range(self.channel_count):
            i = 0
            for design in result:
                if design['channel'] != channel:
                    continue
                for site in design['image_sites']:
                    drops = int(round(best[i,channel]*steps[i,channel]))
                    fraction = float(drops/steps[i,channel])
                    set_site_fraction(design, site, fraction)
                    i += 1
        before,after = self.evaluate(designs,target),self.evaluate(result,target)
        deviation = float(np.mean((self.weights(result)-weights)**2))
        penalty = regularization_strength * deviation
        if after['objective_mse'] + penalty > before['objective_mse']:
            result,after = deepcopy(designs),before
            penalty = deviation = 0.0
        metrics = dict(method='bounded expected-error optimization with within-origami covariance and droplet refinement',
            alignment_backend='chunked' if self.chunked else 'dense',
            gaussian_tail_cutoff_sigma=KERNEL_CUTOFF_SIGMA if self.chunked else None,
            objective='mean image squared error + (1 + variability penalty) * finite-count alignment variance + regularization * mean squared departure from direct assignments',
            regularization_strength=regularization_strength, regularization_penalty=penalty,
            mean_squared_assignment_change=deviation, baseline_regularized_objective=before['objective_mse'],
            optimized_regularized_objective=after['objective_mse']+penalty,
            baseline_rmse=math.sqrt(before['expected_mse']),optimized_rmse=math.sqrt(after['expected_mse']),
            mse_reduction_fraction=(before['expected_mse']-after['expected_mse'])/before['expected_mse'] if before['expected_mse'] else 0.,
            training_baseline=before,training_optimized=after,alignment=asdict(self.settings),
            training_seed=self.seed,error_samples=ERROR_SAMPLES,barcode_bits_fixed=True,
            fwhm_nm=self.fwhm,exposure=self.exposure,linear_light=self.linear_light,normalize_designs=self.normalize_designs,compensate_area=self.compensate_area)
        return result,metrics
