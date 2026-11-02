"""Tk-free alignment jobs; UI callers pass snapshots and receive complete results."""
from concurrent.futures import CancelledError
from dataclasses import asdict, replace

from alignment_reconstruction import AlignmentCapacityError, AlignmentModel, TRAIN_SEED, VALIDATION_SEED
from gaussian_optimization import fitted_target, optimize_designs
from gaussian_reconstruction import render_gaussian_overlay


def cached_alignment_model(cache, designs, settings, fwhm, exposure, light,
                           seed=TRAIN_SEED, checkpoint=None, progress=None):
    geometry_settings = replace(settings, origami_per_design=1, variability_penalty=0)
    signature = tuple((d['name'], d['encoding_scheme'], d.get('origami_rotation_degrees_clockwise', 0),
                       tuple(d.get('active_image_columns', ())), tuple(d.get('active_canvas_offset_nm', ())),
                       d['x_shift_nm'], d['y_shift_nm'],
                       tuple((s['name'], s['x_nm'], s['y_nm']) for s in d['image_sites'])) for d in designs)
    key = (signature, geometry_settings, fwhm, exposure, tuple(sorted(light.items())), seed)
    if key not in cache:
        model = AlignmentModel(designs, settings, fwhm, exposure, seed=seed,
                               checkpoint=checkpoint, progress=progress, **light)
        # Keep the current training and validation geometry.
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[key] = model
    return cache[key].with_settings(settings)


def simulate_alignment(designs, settings, fwhm, exposure, light, image_path,
                       fit_mode, optimize, regularization, example_seed, cache,
                       cancel, progress):
    def checkpoint():
        if cancel.is_set():
            raise CancelledError()

    def report(value, label):
        checkpoint()
        progress(value, label)

    report(5, 'Building alignment model…')
    try:
        model = cached_alignment_model(cache, designs, settings, fwhm, exposure, light,
                                       checkpoint=checkpoint,
                                       progress=lambda fraction: report(5 + 19*fraction, 'Building alignment chunks…'))
    except AlignmentCapacityError as exc:
        report(20, 'Rendering Gaussian preview; alignment exceeds the size limit…')
        reconstruction = gaussian_capacity_preview(designs, settings, fwhm, exposure, light, str(exc))
        baseline, metrics = None, None
        if optimize:
            baseline = reconstruction
            report(35, 'Optimizing Gaussian overlap without alignment uncertainty…')
            target = fitted_target(image_path, fit_mode, baseline.rgb.size,
                                   linear_light=light['linear_light'],
                                   monochrome=baseline.metadata['image_mode'] == 'grayscale')
            designs, metrics = optimize_designs(designs, target, fwhm, exposure, **light,
                                               regularization_strength=regularization,
                                               checkpoint=checkpoint)
            metrics.update(alignment_included=False, alignment_fallback=str(exc))
            report(90, 'Rendering optimized Gaussian preview…')
            reconstruction = gaussian_capacity_preview(designs, settings, fwhm, exposure, light,
                                                       str(exc), optimized=True)
        report(100, 'Gaussian preview ready; alignment unavailable for this layout')
        return dict(designs=designs, baseline=baseline, metrics=metrics,
                    reconstruction=reconstruction, ideal=None, example=None, cache=cache)
    baseline, metrics = None, None
    if optimize:
        report(25, 'Optimizing assignments…')
        baseline = model.expected(designs)
        target = fitted_target(image_path, fit_mode, baseline.rgb.size, linear_light=light['linear_light'],
                               monochrome=model.channel_count == 1)
        original = designs
        designs, metrics = model.optimize(designs, target, regularization_strength=regularization,
                                         checkpoint=checkpoint)
        report(55, 'Checking optimized assignments…')
        validation = cached_alignment_model(cache, original, settings, fwhm, exposure, light,
                                            seed=VALIDATION_SEED, checkpoint=checkpoint,
                                            progress=lambda fraction: report(55 + 14*fraction, 'Checking alignment chunks…'))
        metrics.update(validation_seed=VALIDATION_SEED,
                       validation_baseline=validation.evaluate(original, target),
                       validation_optimized=validation.evaluate(designs, target),
                       validation_used_for_selection=False)
    report(70, 'Rendering predictions…')
    reconstruction = model.expected(designs)
    ideal = render_gaussian_overlay(designs, fwhm, exposure, **light)
    example = model.example(designs, example_seed,
                            progress=lambda fraction: report(75 + 24*fraction, 'Simulating origami…'))
    report(100, 'Simulation complete')
    return dict(designs=designs, baseline=baseline, metrics=metrics,
                reconstruction=reconstruction, ideal=ideal, example=example, cache=cache)


def gaussian_capacity_preview(designs, settings, fwhm, exposure, light, reason, *, optimized=False):
    """Preserve the physical Gaussian width; explicitly omit alignment effects."""
    result = render_gaussian_overlay(designs, fwhm, exposure, **light)
    result.metadata.update(alignment_fallback=reason,
                           requested_alignment=asdict(settings),
                           preview_kind='ideal Gaussian overlay; alignment size limit exceeded',
                           alignment_included=False, optimization_skipped=not optimized)
    return result
