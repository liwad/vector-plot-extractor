# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:09:47 2024

@author: Yu-Chen Wang

helper functions for filtering out not useful elements
"""

import numpy as np
from itertools import repeat, islice

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

DEFAULT_SHAPE_TOL = 1e-2
DEFAULT_COLOR_TOL = 5e-3

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

def eq(ar0, ar1, eta=None):
    if eta is None:
        eta = DEFAULT_SHAPE_TOL
    ar0 = np.array(ar0)
    ar1 = np.array(ar1)
    if ar0.shape != ar1.shape:
        return False
    else:
        return np.all(np.abs(ar0 - ar1) < eta)


def color_eq(col0, col1, eta=DEFAULT_COLOR_TOL):
    col0 = np.array(col0, dtype=float)
    col1 = np.array(col1, dtype=float)

    if col0.shape != col1.shape:
        return False

    if not (np.all(np.isfinite(col0)) and np.all(np.isfinite(col1))):
        return np.array_equal(col0, col1)

    return np.all(np.abs(col0 - col1) <= eta)


def _expand_parameter(value, count, name):
    """Return a list of ``count`` values for broadcasting parameters."""

    if count == 0:
        return []

    if isinstance(value, str):
        return [value] * count

    try:
        iterator = iter(value)
    except TypeError:
        return [value] * count

    result = list(islice(iterator, count))
    if not result:
        raise ValueError(f'expected at least one value for {name}')
    if len(result) == 1 and count > 1:
        result = result * count
    elif len(result) != count:
        raise ValueError(f'expected {count} values for {name}, got {len(result)}')
    return result


def select_paths(target_feature, path_features, modes='s', pos_tol=DEFAULT_SHAPE_TOL, color_tol=DEFAULT_COLOR_TOL):
    count = len(path_features)
    if isinstance(modes, str):
        mode_list = [modes] * count
    else:
        try:
            mode_list = list(modes)
        except TypeError as exc:
            raise ValueError('modes must be a string or an iterable of mode codes') from exc
        if not mode_list:
            raise ValueError('expected at least one mode')
        if len(mode_list) == 1 and count > 1:
            mode_list = mode_list * count
        elif len(mode_list) != count:
            raise ValueError(f'expected {count} modes, got {len(mode_list)}')

    pos_tols = _expand_parameter(pos_tol, count, 'pos_tol')
    color_tols = _expand_parameter(color_tol, count, 'color_tol')

    idx = []
    for i, (path_feature, mode, eta_pos, eta_color) in enumerate(zip(path_features, mode_list, pos_tols, color_tols)):
        if mode in 'sl' and not eq(target_feature['rel_pos'], path_feature['rel_pos'], eta=eta_pos):
            continue # not matched
        if mode in 'ol' and not (color_eq(target_feature['color'], path_feature['color'], eta=eta_color) and color_eq(target_feature['fill'], path_feature['fill'], eta=eta_color)):
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
