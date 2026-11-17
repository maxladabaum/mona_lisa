"""Display-ready views of the learned library, prioritizing realized stocks."""
import numpy as np


def vector_panels(result):
    recipe = result.get('recipe')
    if recipe is not None:
        concentrations = np.asarray(recipe['vector_concentrations_nm'])
        return dict(panels=[('ON concentration', concentrations[..., 0]),
                            ('OFF concentration', concentrations[..., 1])],
                    unit='Concentration (nM)', kind='concentration',
                    note='Realized vector wells after 25 nL preparation and dilution; not the continuous fit.')
    if result['model'] == 'pca':
        return dict(panels=[('PCA components', np.asarray(result['parameters']))],
                    unit='Component loading', kind='signed',
                    note='PCA components are a signed reference; their count is the fitted rank, independent of K.')
    library = result.get('library_parameters')
    if library is None:
        return None
    b = np.asarray(library['basis_vectors'])
    if result['model'] == 'on_off':
        panels = [('ON probability', b[..., 0]), ('OFF probability', b[..., 1]), ('Unoccupied probability', b[..., 2])]
        kind, unit = 'probability', 'Conditional probability'
    elif result['model'] == 'on_only':
        panels = [('ON probability', b)]; kind, unit = 'probability', 'Conditional probability'
    else:
        panels = [('Signed additive vectors', b)]; kind, unit = 'signed', 'Signed amplitude'
    return dict(panels=panels, unit=unit, kind=kind,
                note='Learned shared library. Branch coefficients are separate; additive baseline is not counted as a vector.')


def site_canvas(values, sites=None):
    """Map a vector to actual site coordinates, or a labeled schematic matrix."""
    values = np.asarray(values)
    if sites and len(sites) == len(values):
        if all('row' in s and 'column' in s for s in sites):
            coordinates = [(s['row'], s['column']) for s in sites]
            note = 'Origami site layout'
        elif all('x_nm' in s and 'y_nm' in s for s in sites):
            coordinates = [(s['y_nm'], s['x_nm']) for s in sites]
            note = 'Origami coordinates (nm)'
        else:
            coordinates = None
        if coordinates:
            rows = sorted({r for r, _ in coordinates}); columns = sorted({c for _, c in coordinates})
            if len(set(coordinates)) != len(coordinates):
                raise ValueError('Analog sites must have unique canvas coordinates.')
            canvas = np.full((len(rows), len(columns)), np.nan)
            for (r, c), value in zip(coordinates, values):
                canvas[rows.index(r), columns.index(c)] = value
            return canvas, rows, columns, note
    columns = max(i for i in range(1, int(np.sqrt(len(values)))+1) if len(values) % i == 0)
    rows = len(values)//columns
    return values.reshape(rows, columns), list(range(rows)), list(range(columns)), 'Schematic matrix layout (no spatial metadata)'


def draw_site_canvas(ax, values, sites=None, rotation=0, **colors):
    """Use the full physical origami frame for every spatial canvas."""
    from image_encoding import ROWS, rotate_point
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    physical = sites and len(sites) == len(values) and all(
        site.get('row') in ROWS and str(site.get('column', '')).zfill(2) in
        {f'{column:02d}' for column in range(1, 13)} for site in sites)
    if physical:
        cells = []
        for site in sites:
            c = 12-int(site['column']); r = ROWS.index(site['row'])
            x, y = rotate_point(c*9.9+(10 if c >= 6 else 0), r*5.1, 118.9, 35.7, rotation)
            width, height = (5.1, 9.9) if rotation in (90, 270) else (9.9, 5.1)
            cells.append(Rectangle((x-width/2, y-height/2), width, height))
        artist = PatchCollection(cells, cmap=colors.get('cmap', 'viridis'), edgecolors='none')
        artist.set_array(np.asarray(values)); artist.set_clim(colors.get('vmin'), colors.get('vmax'))
        ax.add_collection(artist)
        width, height = (35.7, 118.9) if rotation in (90, 270) else (118.9, 35.7)
        ax.set_xlim(-.2*width, 1.2*width); ax.set_ylim(1.25*height, -.25*height)
        ax.set_aspect('equal')
    else:
        canvas, _, _, _ = site_canvas(values, sites)
        artist = ax.imshow(np.rot90(canvas, -(rotation//90)), interpolation='nearest', aspect='equal', **colors)
    ax.set(xticks=[], yticks=[])
    ax._site_canvas_artist = artist
    ax._site_canvas_geometry = (sites, rotation, bool(physical))
    return artist


def update_site_canvas(ax, values):
    """Keep artists and layout intact while stepping through construction."""
    sites, rotation, physical = ax._site_canvas_geometry
    artist = ax._site_canvas_artist
    if physical:
        artist.set_array(np.asarray(values))
    else:
        canvas, _, _, _ = site_canvas(values, sites)
        artist.set_data(np.rot90(canvas, -(rotation//90)))
    return artist


def draw_mixed_canvas(ax, on, off, sites=None, rotation=0, scale=1.):
    """One map for a premixed dose or the jointly occupied state of each site."""
    on, off = np.asarray(on), np.asarray(off)
    fractions = np.stack((on, off), axis=-1)/max(float(scale), 1e-12)
    background = np.array([.08, .09, .11])
    rgb = (fractions[:, :1]*np.array([0., .8, 1.])
           +fractions[:, 1:]*np.array([1., .45, .05])
           +np.maximum(0, 1-fractions.sum(axis=1))[:, None]*background)
    rgb = np.clip(rgb, 0, 1)
    artist = draw_site_canvas(ax, on+off, sites, rotation, vmin=0, vmax=scale)
    if ax._site_canvas_geometry[2]:
        artist.set_array(None); artist.set_facecolors(rgb)
    else:
        canvas = np.stack([site_canvas(rgb[:, c], sites)[0] for c in range(3)], axis=-1)
        artist.set_data(np.rot90(canvas, -(rotation//90)))
    # Preserve quantitative components for inspection and regression checks.
    ax._mixed_components = (on.copy(), off.copy())
    return artist
