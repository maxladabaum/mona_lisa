"""One fitted population, with color-specific views for image reconstruction."""
import numpy as np


def combined_targets(datasets, metadata):
    """Stack designs, preserving one common analog site order across colors."""
    layouts = [entry['sites'] for entry in metadata.values()]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError('A shared split–pool population requires the same analog site layout across colors.')
    return np.concatenate(list(datasets.values()), axis=0)


def color_views(shared, datasets):
    """Slice assigned rows, never refit or duplicate the shared K/branch recipe."""
    views = {}; offset = 0
    assignment = np.asarray(shared['assignment'])
    for color, targets in datasets.items():
        selected = (assignment >= offset) & (assignment < offset + len(targets))
        view = dict(shared)
        view['assignment'] = assignment[selected] - offset
        view['prediction'] = np.asarray(shared['prediction'])[selected]
        view['address_indices'] = np.asarray(shared['address_indices'])[selected]
        view['mse'] = float(np.mean((view['prediction'] - targets[view['assignment']])**2))
        if 'pca_scores' in shared:
            view['pca_scores'] = np.asarray(shared['pca_scores'])[selected]
        view['shared_pool_fit'] = shared
        views[color] = view
        offset += len(targets)
    return views
