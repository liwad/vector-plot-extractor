# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:09:47 2024

@author: Yu-Chen Wang

helper functions for filtering out not useful elements
"""

import numpy as np
from itertools import repeat

_RECT_MODE_ALIASES = {
    'touch': 'touch',
    'keep': 'touch',
    'include': 'touch',
    'intersect': 'touch',
    'subtract': 'subtract',
    'remove': 'subtract',
    'exclude': 'subtract',
}


def normalize_rect_mode(mode):
    """Normalize region filtering mode names.

    Parameters
    ----------
    mode : str or None
        User supplied mode name. Accepts a handful of aliases so callers can
        use descriptive terms without worrying about the canonical value.

    Returns
    -------
    str
        Either ``'touch'`` (keep only objects that overlap with the rectangle)
        or ``'subtract'`` (remove overlapped objects).
    """

    if mode is None:
        key = 'touch'
    else:
        try:
            key = str(mode).lower()
        except Exception as exc:
            raise ValueError(f'unknown rect filter mode {mode!r}') from exc

    try:
        return _RECT_MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f'unknown rect filter mode {mode!r}') from exc

def eq(ar0, ar1, eta=1e-2):
    ar0 = np.array(ar0)
    ar1 = np.array(ar1)
    if ar0.shape != ar1.shape:
        return False
    else:
        return np.all(np.abs(ar0 - ar1) < eta)


def color_eq(col0, col1, eta=5e-3):
    col0 = np.array(col0, dtype=float)
    col1 = np.array(col1, dtype=float)

    if col0.shape != col1.shape:
        return False

    if not (np.all(np.isfinite(col0)) and np.all(np.isfinite(col1))):
        return np.array_equal(col0, col1)

    return np.all(np.abs(col0 - col1) <= eta)


def select_paths(target_feature, path_features, modes='s'):
    if isinstance(modes, (tuple, list)) and len(modes) != len(path_features):
        raise ValueError(f'expected {len(path_features)} or 1 modes, got {len(path_features)}')
    if isinstance(modes, str):
        modes = repeat(modes)

    idx = []
    for i, (path_feature, mode) in enumerate(zip(path_features, modes)):
        if mode in 'sl' and not eq(target_feature['rel_pos'], path_feature['rel_pos']):
            continue # not matched
        elif mode in 'ol' and not (color_eq(target_feature['color'], path_feature['color']) and color_eq(target_feature['fill'], path_feature['fill'])):
            continue # not matched
        idx.append(i)
    return idx

def rect_filter_objects(objects, x0, x1, y0, y1, mode='touch'):
    """Return a boolean mask of objects affected by a rectangular selection."""

    mode = normalize_rect_mode(mode)
    selected = {}

    for typ, typ_objs in objects.items():
        selected[typ] = np.full(len(typ_objs), False, dtype=bool)

        for i, obj in enumerate(typ_objs):
            x, y = obj['coords']
            x, y = np.array(x), np.array(y)
            if np.any((x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)):
                selected[typ][i] = True

    return selected
    
def get_filtered_objects(objects, selection):
    filtered_objects = {}
    
    for typ, typ_objs in objects.items():
        filtered_objects[typ] = []

        for idx in np.where(selection[typ])[0]:
            filtered_objects[typ].append(typ_objs[idx])
    
    return filtered_objects
