"""Render measured site fractions using the picklist's isotropic Gaussian model.

This is a display/forward model, not an occupancy estimator. The analytic kernel,
fixed reference-area gain, design weighting and optional sRGB transfer match
picklist_generator/gaussian_reconstruction.py and color_light.py. No dependency
on the picklist GUI or its mutable encoding scheme registry is required: saved
manifest geometry and rendering metadata are the source of truth.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from origami_composite import _orient_composite_raster


DEFAULT_FWHM_NM = 7.5
REFERENCE_AREA_NM2 = (9.9 / 4) * (5.1 / 4)
MISSING_SUPPORT_SIGMA = 6.0


def rendering_settings(manifest, fwhm_nm=None):
    saved = manifest.get('gaussian_reconstruction') or {}
    geometry = manifest['geometry']
    normalize = bool(saved.get('normalize_designs', False))
    reference = float(saved.get('reference_area_nm2', REFERENCE_AREA_NM2))
    abundance = float(saved.get('design_abundance_scale',
                      48.0 / len(manifest['designs']) if normalize else 1.0))
    area = float(saved.get('canvas_area_gain',
                 geometry.get('tile_count', 1) if saved.get('compensate_area') else 1.0))
    exposure = float(saved.get('exposure', 1.0))
    fwhm = float(fwhm_nm if fwhm_nm is not None else saved.get('fwhm_nm', DEFAULT_FWHM_NM))
    gain = float(saved.get('linear_rgb_gain', reference * exposure * abundance * area))
    if not np.isfinite([fwhm, gain, reference, exposure, abundance, area]).all() or min(fwhm, gain, reference, exposure, abundance, area) <= 0:
        raise ValueError('Gaussian width and saved rendering gains must be finite and positive.')
    return dict(fwhm_nm=fwhm, sigma_nm=fwhm / np.sqrt(8 * np.log(2)),
                width_source='override' if fwhm_nm is not None else ('manifest' if 'fwhm_nm' in saved else 'default'),
                linear_rgb_gain=gain, reference_area_nm2=reference, exposure=exposure,
                design_abundance_scale=abundance, canvas_area_gain=area,
                normalize_designs=normalize, linear_light=bool(saved.get('linear_light', False)),
                channel_count=int(geometry.get('channel_count', 3)),
                model='sum_of_unit_integral_isotropic_gaussians',
                signal_weight='Measured ON/valid fraction per site, times the saved per-design weight; no sample-count weighting.',
                normalization='Saved fixed linear_rgb_gain, then clipping and optional sRGB display encoding; no image-dependent maximum.',
                edge_handling='Full analytic Gaussians summed at saved physical site coordinates; tails outside the image field cropped without renormalization.',
                missing_support_sigma=MISSING_SUPPORT_SIGMA,
                missing_policy='Pixels within six sigma of an unmeasured site are missing in that channel. More distant unknown tails are neglected for the mask only (kernel amplitude below exp(-18)). No missing fraction is inferred; raw per-site values remain NaN.',
                alignment_model='Uses the saved Gaussian width; does not add simulated alignment jitter to measured, already aligned origami.')


def gaussian_rgb(signal, linear_light=False):
    finite = np.isfinite(signal)
    values = np.clip(np.where(finite, signal, 0), 0, 1)
    if linear_light:
        values = np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1/2.4) - .055)
    rgb = np.rint(values * 255).astype(np.uint8)
    rgb[~finite.all(axis=2)] = [255, 0, 255]
    return rgb


def _sum_kernels(xs, ys, centers, weights, sigma):
    """Separable analytic evaluation, bounded for irregular manual placements."""
    if not len(centers):
        return np.zeros((len(ys), len(xs)))
    unique_x, ix = np.unique(centers[:, 0], return_inverse=True)
    unique_y, iy = np.unique(centers[:, 1], return_inverse=True)
    if (len(unique_x) * len(unique_y) <= 1_000_000
            and len(xs)*len(unique_x) + len(ys)*len(unique_y) <= 8_000_000):
        masses = np.zeros((len(unique_y), len(unique_x)))
        np.add.at(masses, (iy, ix), weights)
        kx = np.exp(-.5 * ((xs[:, None] - unique_x) / sigma)**2)
        ky = np.exp(-.5 * ((ys[:, None] - unique_y) / sigma)**2)
        result = ky @ masses @ kx.T
    else:
        result = np.zeros((len(ys), len(xs)))
        batch = max(1, min(128, 4_000_000 // (len(xs) + len(ys))))
        for start in range(0, len(centers), batch):
            block = centers[start:start+batch]
            kx = np.exp(-.5 * ((xs[:, None] - block[:, 0]) / sigma)**2)
            ky = np.exp(-.5 * ((ys[:, None] - block[:, 1]) / sigma)**2)
            result += (ky * weights[start:start+batch]) @ kx.T
    return result / (2 * np.pi * sigma**2)


def _missing_mask(xs, ys, centers, sigma):
    mask = np.zeros((len(ys), len(xs)), bool)
    radius = MISSING_SUPPORT_SIGMA * sigma
    for x, y in centers:
        x0, x1 = np.searchsorted(xs, [x-radius, x+radius], side='left')
        y0, y1 = np.searchsorted(ys, [y-radius, y+radius], side='left')
        mask[y0:y1, x0:x1] |= ((ys[y0:y1, None]-y)**2 + (xs[None, x0:x1]-x)**2) <= radius**2
    return mask


def gaussian_raster(populations, settings, rendering, progress=None):
    cells = np.concatenate([p['cells'] for p in populations.values()])
    lower, upper = cells[:, 0].min(axis=0), cells[:, 1].max(axis=0)
    span = upper - lower
    shape_float = np.ceil(span / settings.pixel_nm - 1e-9)
    if not np.isfinite(shape_float).all() or np.prod(shape_float) > 4_000_000 or min(shape_float) <= 0:
        raise ValueError('Gaussian image exceeds four million pixels. Increase Output pixel (nm).')
    width, height = shape_float.astype(int)
    dx, dy = span / [width, height]
    if max(dx, dy) > rendering['fwhm_nm'] / 4 + 1e-9:
        raise ValueError('Output pixels are too coarse for the Gaussian width. Set Output pixel (nm) to at most FWHM / 4.')
    xs = lower[0] + (np.arange(width) + .5) * dx
    ys = lower[1] + (np.arange(height) + .5) * dy
    density = np.zeros((height, width, 3))
    target = np.zeros_like(density)
    for channel in range(3):
        selected = [p for p in populations.values() if p['channel'] == channel]
        if not selected:
            density[:, :, channel] = target[:, :, channel] = np.nan
            continue
        centers = np.concatenate([p['gaussian_centers'] for p in selected])
        design_weights = np.concatenate([np.full(len(p['ids']), p['gaussian_weight']) for p in selected])
        for field, array in [('occupancy', density), ('targets', target)]:
            values = np.concatenate([p[field] for p in selected])
            measured = np.isfinite(values)
            values = np.where(measured, values, 0) * design_weights
            emitting = values != 0
            array[:, :, channel] = _sum_kernels(xs, ys, centers[emitting], values[emitting], rendering['sigma_nm'])
            array[:, :, channel][_missing_mask(xs, ys, centers[~measured], rendering['sigma_nm'])] = np.nan
            if not measured.any():
                array[:, :, channel] = np.nan
        if progress:
            progress(f'Rendering Gaussian occupancy: {("Red", "Green", "Blue")[channel]}')
    if rendering['channel_count'] == 1:
        density = np.repeat(density[:, :, :1], 3, axis=2)
        target = np.repeat(target[:, :, :1], 3, axis=2)
    orient = lambda a: _orient_composite_raster(a, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    density, target = orient(density), orient(target)
    signal = density * rendering['linear_rgb_gain']
    metadata = dict(rendering, canonical_origin_nm=lower.tolist(), canonical_pixel_nm=[float(dx), float(dy)],
                    canonical_shape=[int(height), int(width)], canonical_field_size_nm=span.tolist(),
                    density_units='occupancy-weighted integrated site signal / nm^2',
                    clipped_pixel_fraction=float(np.mean(np.any(signal > 1 + 1e-6, axis=2))))
    return dict(gaussian_density=density, gaussian_signal=signal,
                gaussian_target_signal=target * rendering['linear_rgb_gain'],
                rgb=gaussian_rgb(signal, rendering['linear_light']), metadata=metadata)


def overlay_on_gaussian_grid(overlay, settings, rendering, scale):
    """Compare continuous images at matching physical coordinates, before gamma."""
    origin = np.asarray(rendering['canonical_origin_nm'])
    dx, dy = rendering['canonical_pixel_nm']
    height, width = rendering['canonical_shape']
    previous = overlay['report']['settings']
    source_origin = np.asarray(overlay['report']['pre_rotation_origin_nm'])
    pixel = overlay['report']['output_pixel_nm']
    xs = (origin[0] + (np.arange(width)+.5)*dx - source_origin[0]) / pixel - .5
    ys = (origin[1] + (np.arange(height)+.5)*dy - source_origin[1]) / pixel - .5
    xx, yy = np.meshgrid(xs, ys)
    output = np.empty((height, width, 3))
    for c in range(3):
        plane = overlay['density'][:, :, c]
        if previous.get('mirror_y'):
            plane = np.flipud(plane)
        if previous.get('mirror_x'):
            plane = np.fliplr(plane)
        plane = np.rot90(plane, int(previous.get('rotation_deg', 0)) // 90)
        output[:, :, c] = map_coordinates(plane, [yy, xx], order=1, mode='constant', cval=np.nan) / scale
    return _orient_composite_raster(output, settings.rotation_deg, settings.mirror_x, settings.mirror_y)


def template_heatmap_data(population, settings, rendering):
    """Compare one barcode in its aligned physical frame, before image placement.

    Both images share the diagnostic histogram's exact bins. The localization
    overlay retains its usual per-section smoothing/averaging; occupancy renders
    the measured fractions at physical analog sites, independent of localization
    multiplicity, RGB gains, decoded placement, or other barcode populations.
    """
    from scipy.ndimage import gaussian_filter
    extent = np.asarray(population['overlay_extent'], float)
    overlay = np.asarray(population['localization_overlay'], float).copy()
    height, width = overlay.shape
    dx = (extent[1]-extent[0]) / width
    dy = (extent[3]-extent[2]) / height
    xs = extent[0] + (np.arange(width)+.5)*dx
    ys = extent[2] + (np.arange(height)+.5)*dy
    count = len(population['states'])
    if count == 0:
        overlay[:] = np.nan
    else:
        if settings.sigma_nm:
            overlay = gaussian_filter(overlay, (settings.sigma_nm/dy, settings.sigma_nm/dx), mode='constant')
        overlay /= dx*dy
        if settings.average_species:
            overlay /= count
    values = np.asarray(population['occupancy'])
    sites = np.asarray(population['sites'])
    measured = np.isfinite(values)
    emitting = measured & (values > 0)
    occupancy = _sum_kernels(xs, ys, sites[emitting], values[emitting], rendering['sigma_nm'])
    occupancy[_missing_mask(xs, ys, sites[~measured], rendering['sigma_nm'])] = np.nan
    if not measured.any():
        occupancy[:] = np.nan
    # Fixed reference for all templates at this FWHM, including overlapping spots.
    occupancy_reference = _sum_kernels(xs, ys, sites, np.ones(len(sites)), rendering['sigma_nm'])
    return dict(overlay=overlay, occupancy=occupancy, extent=extent,
                occupancy_vmax=float(occupancy_reference.max()),
                overlay_units='localizations / nm²' + (' / origami' if settings.average_species else ''),
                occupancy_units='occupancy-weighted signal / nm²',
                grid=np.asarray(population['grid_nm']), xs=xs, ys=ys)


def draw_template_heatmaps(figure, population, settings, rendering, *, show_grid=True,
                          show_values=True, show_target=False, full_grid=False):
    """Heatmaps, grid and values in a common reference; gray means unmeasured."""
    from matplotlib import colormaps
    from matplotlib.patches import Rectangle
    data = template_heatmap_data(population, settings, rendering)
    panels = [('Localization overlay', data['overlay'], data['overlay_units'], None),
              ('Gaussian occupancy', data['occupancy'], data['occupancy_units'], data['occupancy_vmax'])]
    if show_target:
        values = np.asarray(population['targets'])
        finite = np.isfinite(values)
        target = _sum_kernels(data['xs'], data['ys'], population['sites'][finite], values[finite], rendering['sigma_nm'])
        target[_missing_mask(data['xs'], data['ys'], population['sites'][~finite], rendering['sigma_nm'])] = np.nan
        if not finite.any():
            target[:] = np.nan
        panels.append(('Programmed target', target, data['occupancy_units'], data['occupancy_vmax']))
    engine = figure.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0, 0, 1, .92))
    else:
        figure.subplots_adjust(top=.80)
    axes = figure.subplots(1, len(panels), sharex=True, sharey=True)
    cmap = colormaps['viridis'].copy()
    cmap.set_bad('#d9d9d9')
    grid = data['grid']
    xs, ys = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    # Show the full origami for context. The saved image region is drawn in place;
    # missing barcode-side localizations are never invented for compact sessions.
    pad_x = np.min(np.diff(xs))/2
    pad_y = np.min(np.diff(ys))/2
    extent = data['extent']
    limits = [min(xs[0]-pad_x, extent[0]), max(xs[-1]+pad_x, extent[1]),
              min(ys[0]-pad_y, extent[2]), max(ys[-1]+pad_y, extent[3])]
    if not full_grid:
        limits = extent
    for index, (ax, (title, values, units, vmax)) in enumerate(zip(axes, panels)):
        im = ax.imshow(values, origin='lower', extent=extent, cmap=cmap,
                       interpolation='nearest', vmin=0, vmax=vmax)
        ax.set_facecolor('#d9d9d9')
        ax.set(xlim=limits[:2], ylim=limits[2:], aspect='equal',
               xlabel='Aligned X (nm)', title=title)
        if index == 0:
            ax.set_ylabel('Aligned Y (nm)')
        figure.colorbar(im, ax=ax, orientation='horizontal', pad=.12, shrink=.9, label=units)
        ax.add_patch(Rectangle((extent[0], extent[2]), extent[1]-extent[0], extent[3]-extent[2],
                               fill=False, edgecolor='tab:orange', linewidth=1.2))
        if show_grid:
            # Outlined lattice sites and connecting lines remain visible over both
            # dark heatmap values and the gray, unmeasured half of the origami.
            for x in xs:
                ax.plot([x,x], [ys[0],ys[-1]], color='white', alpha=.35, linewidth=.5)
            for y in ys:
                ax.plot([xs[0],xs[-1]], [y,y], color='white', alpha=.35, linewidth=.5)
            ax.scatter(grid[:, 0], grid[:, 1], s=13, facecolors='none', edgecolors='#555555', linewidths=.45)
            ax.scatter(population['sites'][:, 0], population['sites'][:, 1], s=17,
                       facecolors='none', edgecolors='white', linewidths=.6)
        if show_values and index > 0:
            fractions = population['occupancy'] if index == 1 else population['targets']
            for (x, y), value in zip(population['sites'], fractions):
                ax.annotate('?' if not np.isfinite(value) else f'{value:.2f}', (x,y),
                    xytext=(0,4), textcoords='offset points', ha='center', va='bottom',
                    color='white', fontsize=6,
                    bbox=dict(facecolor='black', edgecolor='none', alpha=.45, pad=.3))
    name = population['template'] or population['design_name']
    n = len(population['states'])
    valid = population['valid']
    figure.suptitle(f'{name} · barcode {population["code"]} · {n} accepted origami\n'
                   f'Valid observations/site: {int(valid.min())}–{int(valid.max())}; '
                   f'occupancy FWHM {rendering["fwhm_nm"]:g} nm\n'
                   'Orange: image extraction region · gray: unmeasured · site labels: ON/valid (? = missing)', fontsize=10)
    return data
