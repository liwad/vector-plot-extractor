# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:09:47 2024

@author: Yu-Chen Wang

helper functions for filtering out not useful elements
"""

import numpy as np
from itertools import repeat


DEFAULT_TOLERANCE = 1e-2

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

def eq(ar0, ar1, eta=DEFAULT_TOLERANCE):
    ar0 = np.array(ar0)
    ar1 = np.array(ar1)
    if ar0.shape != ar1.shape:
        return False
    else:
        return np.all(np.abs(ar0 - ar1) < eta)

def _normalize_color(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        inner = arr.item()
        if inner is None:
            return None
        return _normalize_color(inner)
    if arr.size == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    try:
        return arr.astype(float)
    except (TypeError, ValueError):
        return None


def _colors_close(color0, color1, tol):
    arr0 = _normalize_color(color0)
    arr1 = _normalize_color(color1)
    if arr0 is None or arr1 is None:
        return arr0 is None and arr1 is None
    n = min(arr0.size, arr1.size)
    if n == 0:
        return False
    return np.all(np.abs(arr0[:n] - arr1[:n]) < tol)


def is_background_like(path_feature, background_color, tol):
    if background_color is None:
        return False
    bg = _normalize_color(background_color)
    if bg is None:
        return False
    color_match = _colors_close(path_feature.get('color'), bg, tol)
    fill = path_feature.get('fill')
    if fill is None:
        return color_match
    fill_match = _colors_close(fill, bg, tol)
    return color_match and fill_match


def _contrast(value, background_color):
    bg = _normalize_color(background_color)
    if bg is None:
        return 0.0
    arr = _normalize_color(value)
    if arr is None:
        return 0.0
    n = min(arr.size, bg.size)
    if n == 0:
        return 0.0
    diff = arr[:n] - bg[:n]
    return float(np.linalg.norm(diff))


def marker_contrast(path_feature, background_color):
    """Return a contrast score for ``path_feature`` against ``background_color``."""

    color_contrast = _contrast(path_feature.get('color'), background_color)
    fill_contrast = _contrast(path_feature.get('fill'), background_color)
    return max(color_contrast, fill_contrast)


def dedupe_by_contrast(matched_idxs, path_features, tol, background_color):
    """Remove duplicate matches that share geometry, keeping the highest contrast."""

    if len(matched_idxs) <= 1:
        return matched_idxs

    deduped = []
    groups = []

    for idx in matched_idxs:
        feature = path_features[idx]
        contrast = marker_contrast(feature, background_color)

        for group in groups:
            if eq(feature['rel_pos'], group['rel_pos'], eta=tol):
                if contrast > group['contrast']:
                    group['contrast'] = contrast
                    group['index'] = idx
                    deduped[group['order']] = idx
                break
        else:
            order = len(deduped)
            groups.append({'rel_pos': feature['rel_pos'],
                           'contrast': contrast,
                           'index': idx,
                           'order': order})
            deduped.append(idx)

    return deduped


def select_paths(target_feature, path_features, modes='s', *, tolerance=DEFAULT_TOLERANCE, background_color=None):
    if isinstance(modes, (tuple, list)) and len(modes) != len(path_features):
        raise ValueError(f'expected {len(path_features)} or 1 modes, got {len(modes)}')
    if isinstance(modes, str):
        modes = repeat(modes)

    if isinstance(tolerance, (tuple, list, np.ndarray)):
        if len(tolerance) != len(path_features):
            raise ValueError(f'expected {len(path_features)} or 1 tolerances, got {len(tolerance)}')
        tolerances = iter(tolerance)
    else:
        tolerances = repeat(float(tolerance))

    idx = []
    for i, (path_feature, mode, tol) in enumerate(zip(path_features, modes, tolerances)):
        if is_background_like(path_feature, background_color, tol):
            continue
        if mode in 'sl' and not eq(target_feature['rel_pos'], path_feature['rel_pos'], eta=tol):
            continue  # not matched
        if mode in 'ol':
            if not (_colors_close(target_feature['color'], path_feature['color'], tol) and
                    _colors_close(target_feature['fill'], path_feature['fill'], tol)):
                continue
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
