# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:28:30 2024

@author: Yu-Chen Wang
"""

import math
from collections import OrderedDict

import numpy as np
from .filter import select_paths, rect_filter_objects, get_filtered_objects, normalize_rect_mode
from copy import copy, deepcopy
from .drawing import add, plot_objects, get_color, Line2D
import matplotlib.pyplot as plt
from .utils import pause_and_warn, save_pickle, annotate, dedup
import os
import json
from matplotlib.widgets import TextBox, Button
from itertools import chain
from . import __version__
from matplotlib.collections import LineCollection, PathCollection, PatchCollection
from matplotlib.patches import Patch
from matplotlib import colors as mcolors

class ConsistencyError(Exception):
    pass

class BaseEventHandler():
    def __init__(self, fig=None, **kwargs):
        self.pressed_down = False
        self.key_pressed_down = False
        self.cids = []
        self.finished = False
        
        if fig is not None:
            self.connect(fig)
       
        self.init(**kwargs)
    
    def init(self, **kwargs):
        pass
        
    def connect(self, fig):
        self.fig = fig
        self.cids += [
            fig.canvas.mpl_connect('button_press_event', self._onpress),
            fig.canvas.mpl_connect('button_release_event', self._onrelease),
            fig.canvas.mpl_connect('key_press_event', self._onkeypress),
            fig.canvas.mpl_connect('key_release_event', self._onkeyrelease),
            fig.canvas.mpl_connect('motion_notify_event', self._onmove),
            fig.canvas.mpl_connect('pick_event', self.onpick),
            fig.canvas.mpl_connect('close_event', self.onclose),
            ]
        
    def disconnect(self):
        for cid in self.cids:
            self.fig.canvas.mpl_disconnect(cid)
        self.cids.clear()
    
    def _onpress(self, event):
        self.pressed_down = True
        self.onpress(event)
    
    def onpress(self, event):
        pass
    
    def _onrelease(self, event):
        self.pressed_down = False
        self.onrelease(event)
    
    def onrelease(self, event):
        pass
    
    def _onkeypress(self, event):
        self.key_pressed_down = True
        self.onkeypress(event)
    
    def onkeypress(self, event):
        pass
    
    def _onkeyrelease(self, event):
        self.key_pressed_down = False
        self.onkeyrelease(event)
    
    def onkeyrelease(self, event):
        pass
    
    def _onmove(self, event):
        if self.pressed_down:
            self.onmove_down(event)
        else:
            self.onmove_up(event)
    
    def onmove(self, event):
        pass
    
    def onmove_up(self, event):
        self.onmove(event)
        
    def onmove_down(self, event):
        self.onmove(event)
        
    def onpick(self, event):
        pass
    
    def onclose(self, event):
        if not self.finished:
            self.finished = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, *exc):
        self.disconnect()
        self.finalize()
        
    def finalize(self):
        pass
        
    def wait(self, interval=.1):
        # wait until self.finish
        while True:
            if self.finished:
                return
            plt.pause(interval)
            

class RectSelector(BaseEventHandler):
    def init(self, finish=True):
        # finish: if set to True, will be considered finished whenever a region is selected
        self.rects = {}
        for ax in self.fig.axes:
            rect, = ax.plot([], [], linestyle='--', color='r')
            self.rects[ax] = rect
        self.finish = finish

    def onpress(self, event):
        if event.inaxes not in self.rects:
            self.ax = None
            return

        if event.xdata is None or event.ydata is None:
            self.ax = None
            return

        self.finished = False
        self.ax = event.inaxes
        self.x0, self.y0 = event.xdata, event.ydata

    def onmove_down(self, event):
        if event.inaxes == self.ax:
            self.x1, self.y1 = event.xdata, event.ydata

            x, y  = self.get_xydata()
            self.rects[self.ax].set_data(x, y)
            self.rects[self.ax].set_marker('')
            self.fig.canvas.draw()
            
    def onrelease(self, event):
        if self.ax is None or self.ax not in self.rects:
            return

        x1 = getattr(self, 'x1', self.x0)
        y1 = getattr(self, 'y1', self.y0)

        self.x0, x1 = np.sort((self.x0, x1))
        self.y0, y1 = np.sort((self.y0, y1))
        self.x1, self.y1 = x1, y1

        self.rects[self.ax].set_marker('s')
        self.fig.canvas.draw()

        if self.finish:
            self.finished = True
        
    def get_xydata(self, closed=True):
        x0, x1 = np.sort((self.x0, self.x1))
        y0, y1 = np.sort((self.y0, self.y1))
        if closed:
            return [x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0]
        else:
            return [x0, x0, x1, x1], [y0, y1, y1, y0]

class ObjectChecker(BaseEventHandler):
    '''
    click an object and show information in terminal
    '''
    
    def init(self, ax, artists, artists_in_plot, path_features):
        self.artists_in_plot = artists_in_plot
        self.path_features = path_features
    
    def onpick(self, event):
        artist = event.artist
        idx = self.artists_in_plot.index(artist)
        self.path_feature = self.path_features[idx]
        
        print(f'idx = {idx}')
        print(artist)
        print(f'path_feature = {self.path_feature}')

class ElementIdentifier(BaseEventHandler):
    _EXTENT_DECIMALS = 6
    _AUTO_DISCARD_CONTRAST = 0.05
    _MIN_SHAPE_TOL = 1e-5
    _MIN_COLOR_TOL = 1e-5

    def init(self, ax, artists, artists_in_plot, path_features):
        self.ax = ax
        self.known_markers = []
        self.matches = []
        self.state = 0
        self._marker_preview_idx = None
        self._group_preview_idxs = None
        self.matched_idxs = []
        self.shape_tol = DEFAULT_SHAPE_TOL
        self.color_tol = DEFAULT_COLOR_TOL
        self._tol_update_guard = False
        self._shape_tol_box = None
        self._color_tol_box = None
        self._auto_discard_members = ()
        self._auto_discard_count = 0
        self._hidden_duplicates = 0

        self._background_rgb = self._determine_background_rgb(ax)
        self._background_contrast_threshold = self._AUTO_DISCARD_CONTRAST

        prepared = self._prepare_artists(artists, artists_in_plot, path_features)
        self.artists = prepared['artists']
        self.artists_in_plot = prepared['artists_in_plot']
        self.path_features = prepared['path_features']
        self.indexes = np.array(prepared['base_indices'], dtype=int)
        self._duplicate_lookup = prepared['duplicate_lookup']
        self._hidden_duplicates = prepared['hidden_duplicates']
        self._auto_discard_members = tuple(prepared.get('auto_discard', ()))
        self._auto_discard_count = sum(len(group) for group in self._auto_discard_members)
        self.types = np.full((prepared['original_count'],), fill_value='u', dtype='S1') # [S]catter, [L]ine, [D]iscard. u means "not marked"
        if self._auto_discard_members:
            for group in self._auto_discard_members:
                self.types[np.array(group, dtype=int)] = b'd'

        self._setup_tolerance_controls()

        self._idle_title = self._compose_idle_title()
        self.fig.suptitle(self._idle_title)

        self.cids.append(self.fig.canvas.mpl_connect('resize_event', self._on_resize))
    
    def onpick(self, event):
        artist = event.artist
        # print(artist)
        if self.state == 0 and artist in self.artists_in_plot: #event.inaxes == self.ax['main']:
            self.picked = True
            idx = self.artists_in_plot.index(artist)
            self.path_feature = self.path_features[idx]
            # print(idx, self.path_feature)
            self.fig.suptitle('object type: [S]catter, [L]ine, [D]iscard, [O]thers, [Del] remove, or [C]ancel')
            self._draw_marker_preview(idx)
            self.state = 1

            self.fig.canvas.draw()
        
    # match_mode_dict = { # keyboard shortcut: mode code in code
    #     's': 's',
    #     'o': 'c',
    #     'l': 'sc',
    #     }
    
    type_names = { # type names to display
        's': 'scatter',
        'l': 'line',
        'd': 'discard',
        'o': 'others',
        }

    def _compose_idle_title(self):
        base = 'click element to identify'
        if self._hidden_duplicates:
            dup = self._hidden_duplicates
            plural = 's' if dup != 1 else ''
            base += f' ({dup} duplicate{plural} hidden)'
        if self._auto_discard_count:
            auto = self._auto_discard_count
            plural = 's' if auto != 1 else ''
            base += f' ({auto} low-contrast auto-discard{plural})'
        return base + ', or [F]inish'

    def _format_tol(self, value):
        return f'{float(value):.4g}'

    def _parse_tol_input(self, text, current, minimum):
        try:
            value = float(text)
        except (TypeError, ValueError):
            return current, False
        if not np.isfinite(value) or value <= 0:
            return current, False
        value = max(value, minimum)
        if math.isclose(value, current, rel_tol=1e-9, abs_tol=1e-12):
            return current, False
        return value, True

    def _update_textbox(self, box, value):
        if box is None:
            return
        if self._tol_update_guard:
            return
        self._tol_update_guard = True
        try:
            box.set_val(self._format_tol(value))
        finally:
            self._tol_update_guard = False

    def _setup_tolerance_controls(self):
        if not hasattr(self, 'fig'):
            return

        group_ax = self.ax.get('group') if isinstance(self.ax, dict) else None
        bbox = group_ax.get_position() if group_ax is not None else self.fig.axes[-1].get_position()
        width = min(0.22, max(0.12, bbox.width))
        height = 0.05
        pad = 0.01
        y_color = bbox.y0 - height - pad
        y_shape = y_color - height - pad
        if y_shape < 0.02:
            y_shape = min(0.9 - height, bbox.y1 + pad)
            y_color = y_shape + height + pad
            if y_color + height > 0.95:
                y_color = max(0.02, y_shape - height - pad)
        y_shape = min(max(y_shape, 0.02), 0.95 - height)
        y_color = min(max(y_color, 0.02), 0.95 - height)
        x = min(max(bbox.x0, 0.02), 1 - width - pad)

        shape_ax = self.fig.add_axes([x, y_shape, width, height])
        color_ax = self.fig.add_axes([x, y_color, width, height])
        shape_ax._selector_ignore = True
        color_ax._selector_ignore = True

        self._shape_tol_box = TextBox(shape_ax, 'shape tol', initial=self._format_tol(self.shape_tol))
        self._color_tol_box = TextBox(color_ax, 'color tol', initial=self._format_tol(self.color_tol))
        self._shape_tol_box.on_submit(self._on_shape_tol_submit)
        self._color_tol_box.on_submit(self._on_color_tol_submit)

    def _on_shape_tol_submit(self, text):
        if self._tol_update_guard:
            return
        new_value, changed = self._parse_tol_input(text, self.shape_tol, self._MIN_SHAPE_TOL)
        self.shape_tol = new_value
        self._update_textbox(self._shape_tol_box, self.shape_tol)
        if changed and self.state == 0:
            self.fig.suptitle(self._idle_title)
        self.fig.canvas.draw_idle()

    def _on_color_tol_submit(self, text):
        if self._tol_update_guard:
            return
        new_value, changed = self._parse_tol_input(text, self.color_tol, self._MIN_COLOR_TOL)
        self.color_tol = new_value
        self._update_textbox(self._color_tol_box, self.color_tol)
        if changed and self.state == 0:
            self.fig.suptitle(self._idle_title)
        self.fig.canvas.draw_idle()

    def _quantize_rel_pos(self, rel_pos):
        arr = np.array(rel_pos, dtype=np.float64, copy=True)
        if arr.size == 0:
            return arr
        finite = np.isfinite(arr)
        if finite.any():
            tol = max(float(self.shape_tol), self._MIN_SHAPE_TOL)
            arr[finite] = np.round(arr[finite] / tol) * tol
        return arr

    def _quantize_color(self, color):
        arr = np.array(color, dtype=np.float64, copy=True)
        if arr.size == 0:
            return arr
        finite = np.isfinite(arr)
        if finite.any():
            tol = max(float(self.color_tol), self._MIN_COLOR_TOL)
            arr[finite] = np.round(arr[finite] / tol) * tol
        return arr

    def _round_tuple(self, values, decimals):
        arr = np.array(values, dtype=np.float64, copy=True)
        if arr.size:
            finite = np.isfinite(arr)
            if finite.any():
                arr[finite] = np.round(arr[finite], decimals)
        return tuple(arr.tolist())

    def _feature_signature(self, feature):
        rel_pos = self._quantize_rel_pos(feature['rel_pos'])
        color = self._quantize_color(feature['color'])
        fill = self._quantize_color(feature['fill'])
        extent = self._round_tuple(feature.get('extent', (0.0, 0.0)), self._EXTENT_DECIMALS)
        bbox = self._round_tuple(feature.get('bbox', (0.0, 0.0, 0.0, 0.0)), self._EXTENT_DECIMALS)
        return (
            feature.get('type'),
            rel_pos.tobytes(),
            color.tobytes(),
            fill.tobytes(),
            feature.get('artist_class'),
            extent,
            bbox,
            bool(feature.get('closed', False)),
            bool(feature.get('has_stroke', False)),
            bool(feature.get('has_fill', False)),
        )

    def _geometry_signature(self, feature):
        rel_pos = self._quantize_rel_pos(feature['rel_pos'])
        extent = self._round_tuple(feature.get('extent', (0.0, 0.0)), self._EXTENT_DECIMALS)
        bbox = self._round_tuple(feature.get('bbox', (0.0, 0.0, 0.0, 0.0)), self._EXTENT_DECIMALS)
        return (
            feature.get('type'),
            rel_pos.tobytes(),
            feature.get('artist_class'),
            extent,
            bbox,
            bool(feature.get('closed', False)),
        )

    def _determine_background_rgb(self, axes):
        page_rgb = np.ones(3, dtype=float)
        fig_rgba = self._to_rgba(self.fig.get_facecolor() if getattr(self, 'fig', None) is not None else None)
        fig_rgb = self._composite_rgb(fig_rgba, page_rgb)
        main_ax = axes.get('main') if isinstance(axes, dict) else axes
        axis_face = None
        if main_ax is not None:
            axis_face = getattr(main_ax, 'get_facecolor', lambda: None)()
        axis_rgba = self._to_rgba(axis_face)
        background_rgb = self._composite_rgb(axis_rgba, fig_rgb)
        return background_rgb

    def _to_rgba(self, color):
        if color is None:
            return None
        try:
            rgba = np.asarray(mcolors.to_rgba(color), dtype=float)
        except (TypeError, ValueError):
            arr = np.asarray(color, dtype=float).ravel()
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return None
            rgb = arr[:3]
            alpha = arr[3] if arr.size > 3 and np.isfinite(arr[3]) else 1.0
            rgba = np.concatenate([rgb, [alpha]])
        if rgba.shape != (4,) or not np.all(np.isfinite(rgba)):
            return None
        return np.clip(rgba, 0.0, 1.0)

    def _composite_rgb(self, top_rgba, bottom_rgb):
        base = np.asarray(bottom_rgb, dtype=float) if bottom_rgb is not None else np.ones(3, dtype=float)
        if top_rgba is None:
            return base
        alpha = float(np.clip(top_rgba[3], 0.0, 1.0))
        top_rgb = top_rgba[:3]
        return top_rgb * alpha + base * (1.0 - alpha)

    def _effective_color(self, value):
        rgba = self._to_rgba(value)
        if rgba is None:
            return None
        alpha = float(np.clip(rgba[3], 0.0, 1.0))
        rgb = rgba[:3]
        background = self._background_rgb if self._background_rgb is not None else np.ones(3, dtype=float)
        if alpha >= 1.0 - 1e-6:
            return rgb
        return rgb * alpha + background * (1.0 - alpha)

    def _color_contrast(self, value):
        if value is None:
            return 0.0
        effective = self._effective_color(value)
        if effective is None:
            return 0.0
        background = self._background_rgb if self._background_rgb is not None else np.ones(3, dtype=float)
        return float(np.linalg.norm(effective - background))

    def _style_score(self, feature):
        scores = []
        if feature.get('has_stroke', False):
            scores.append(self._color_contrast(feature.get('color')))
        if feature.get('has_fill', False):
            scores.append(self._color_contrast(feature.get('fill')))
        if not scores:
            return 0.0
        return max(scores)

    def _choose_geometry_representative(self, base_indices, path_features):
        best_idx = None
        best_score = -np.inf
        for idx in base_indices:
            feature = path_features[idx]
            score = self._style_score(feature)
            if best_idx is None or score > best_score + 1e-12:
                best_idx = idx
                best_score = score
        return best_idx if best_idx is not None else base_indices[0]

    def _should_merge_geometry_group(self, base_indices, path_features):
        if len(base_indices) <= 1:
            return False

        has_fill = False
        has_stroke = False
        for idx in base_indices:
            feature = path_features[idx]
            if not feature.get('closed', False):
                return False
            has_fill = has_fill or feature.get('has_fill', False)
            has_stroke = has_stroke or feature.get('has_stroke', False)

        if not has_fill:
            return False
        if not has_stroke:
            return False

        return True

    def _prepare_artists(self, artists, artists_in_plot, path_features):
        signature_map = OrderedDict()
        for idx, feature in enumerate(path_features):
            signature = self._feature_signature(feature)
            signature_map.setdefault(signature, []).append(idx)

        base_indices = []
        duplicate_lookup = {}

        for group in signature_map.values():
            base_idx = group[0]
            base_indices.append(base_idx)
            dupset = duplicate_lookup.setdefault(base_idx, set())
            for dup_idx in group:
                dupset.add(dup_idx)
            for dup_idx in group[1:]:
                preview_artist = artists_in_plot[dup_idx]
                if getattr(preview_artist, 'axes', None) is not None:
                    preview_artist.remove()

        geometry_groups = OrderedDict()
        for base_idx in base_indices:
            feature = path_features[base_idx]
            geometry_signature = self._geometry_signature(feature)
            geometry_groups.setdefault(geometry_signature, []).append(base_idx)

        final_indices = []
        for group in geometry_groups.values():
            if len(group) == 1 or not self._should_merge_geometry_group(group, path_features):
                for idx in group:
                    final_indices.append(idx)
                continue

            representative = self._choose_geometry_representative(group, path_features)
            final_indices.append(representative)
            members = set()
            for idx in group:
                members.update(duplicate_lookup.get(idx, {idx}))
                if idx == representative:
                    continue
                preview_artist = artists_in_plot[idx]
                if getattr(preview_artist, 'axes', None) is not None:
                    preview_artist.remove()
                duplicate_lookup.pop(idx, None)
            duplicate_lookup[representative] = members

        hidden_duplicates = len(path_features) - len(final_indices)

        visible_indices = []
        visible_lookup = {}
        visible_artists = []
        visible_features = []
        visible_plot_artists = []
        auto_discard = []

        for idx in final_indices:
            members = duplicate_lookup.get(idx, {idx})
            if not isinstance(members, set):
                members = set(members)
            member_tuple = tuple(sorted(members))
            feature = path_features[idx]
            score = self._style_score(feature)
            if score < self._background_contrast_threshold:
                preview_artist = artists_in_plot[idx]
                if getattr(preview_artist, 'axes', None) is not None:
                    preview_artist.remove()
                auto_discard.append(member_tuple)
                continue

            visible_indices.append(idx)
            visible_lookup[idx] = member_tuple
            visible_artists.append(artists[idx])
            visible_features.append(feature)
            preview_artist = artists_in_plot[idx]
            self._tweak_artist_for_preview(preview_artist)
            visible_plot_artists.append(preview_artist)

        return {
            'artists': visible_artists,
            'artists_in_plot': visible_plot_artists,
            'path_features': visible_features,
            'base_indices': visible_indices,
            'duplicate_lookup': visible_lookup,
            'hidden_duplicates': hidden_duplicates,
            'auto_discard': tuple(auto_discard),
            'original_count': len(path_features),
        }

    def _on_resize(self, _event):
        if self._marker_preview_idx is not None:
            if self._marker_preview_idx < len(self.artists):
                self._draw_marker_preview(self._marker_preview_idx, redraw_only=True)
            else:
                self._marker_preview_idx = None
                self.ax['marker'].clear()

        if self._group_preview_idxs:
            valid = [idx for idx in self._group_preview_idxs if idx < len(self.artists)]
            if valid:
                self._group_preview_idxs = valid
                self._draw_group_preview(valid, redraw_only=True)
            else:
                self._group_preview_idxs = None
                self.ax['group'].clear()
                self.ax['group'].set_title('')
                self.fig.canvas.draw_idle()

    def _make_preview_artist(self, artist, axis):
        preview = copy(artist)
        preview.set_transform(axis.transData)
        self._tweak_artist_for_preview(preview)
        return preview

    def _tweak_artist_for_preview(self, artist):
        lw_min, lw_max = 0.3, 2.5
        try:
            if isinstance(artist, Line2D):
                lw = artist.get_linewidth()
                if np.isfinite(lw):
                    artist.set_linewidth(min(max(lw, lw_min), lw_max))
                ms = artist.get_markersize()
                if np.isfinite(ms):
                    artist.set_markersize(min(ms, 12))
            elif isinstance(artist, LineCollection):
                lws = artist.get_linewidths()
                if lws is not None and len(lws):
                    artist.set_linewidths(np.clip(lws, lw_min, lw_max))
            elif isinstance(artist, (PatchCollection, PathCollection)):
                lws = artist.get_linewidths()
                if lws is not None and len(lws):
                    artist.set_linewidths(np.clip(lws, lw_min, lw_max))
            elif isinstance(artist, Patch):
                lw = artist.get_linewidth()
                if lw is not None and np.isfinite(lw):
                    artist.set_linewidth(min(max(lw, lw_min), lw_max))
        except Exception:
            pass

    def _draw_marker_preview(self, idx, redraw_only=False):
        if not redraw_only:
            self._marker_preview_idx = idx
        ax = self.ax['marker']
        ax.clear()
        if idx is None or idx >= len(self.artists):
            self.fig.canvas.draw_idle()
            return
        preview = self._make_preview_artist(self.artists[idx], ax)
        add(ax, preview)
        ax.autoscale(True)
        ax.invert_yaxis()
        self.fig.canvas.draw_idle()

    def _draw_group_preview(self, indices, redraw_only=False):
        if not redraw_only:
            self._group_preview_idxs = list(indices)
        elif self._group_preview_idxs is not None:
            indices = self._group_preview_idxs
        ax = self.ax['group']
        ax.clear()
        if not indices:
            ax.set_title('')
            self.fig.canvas.draw_idle()
            return
        count = 0
        for idx in indices:
            if idx is None or idx >= len(self.artists):
                continue
            preview = self._make_preview_artist(self.artists[idx], ax)
            add(ax, preview)
            count += 1
        ax.autoscale(True)
        ax.invert_yaxis()
        ax.set_title(f'found {count}')
        self.fig.canvas.draw_idle()

    def _safe_extent_ratio(self, extent):
        width, height = extent
        eps = 1e-6
        if width < eps or height < eps:
            return None
        return width / height

    def _extent_compatible(self, base_extent, candidate_extent):
        base_extent = np.asarray(base_extent, dtype=float)
        candidate_extent = np.asarray(candidate_extent, dtype=float)
        if base_extent.shape != (2,) or candidate_extent.shape != (2,):
            return True
        eps = 1e-6
        base_diag = math.hypot(base_extent[0], base_extent[1])
        cand_diag = math.hypot(candidate_extent[0], candidate_extent[1])
        if base_diag < eps:
            return cand_diag < 3 * eps
        ratio = cand_diag / base_diag if base_diag else np.inf
        if ratio < 0.5 or ratio > 2.0:
            return False
        base_ratio = self._safe_extent_ratio(base_extent)
        cand_ratio = self._safe_extent_ratio(candidate_extent)
        if base_ratio is None or cand_ratio is None:
            return True
        rel = cand_ratio / base_ratio if base_ratio else np.inf
        return 0.5 <= rel <= 2.0

    def _filter_matches(self, matched_idxs):
        if self.type != 's':
            return matched_idxs
        base_extent = self.path_feature.get('extent')
        base_artist_class = self.path_feature.get('artist_class')
        filtered = []
        for idx in matched_idxs:
            feature = self.path_features[idx]
            if base_artist_class and feature.get('artist_class') != base_artist_class:
                continue
            if base_extent is not None and feature.get('extent') is not None:
                if not self._extent_compatible(base_extent, feature.get('extent')):
                    continue
            filtered.append(idx)
        return filtered

    def _find_line_like_match(self, indices):
        for idx in indices:
            artist = self.artists[idx]
            if isinstance(artist, Line2D) and (artist.get_marker() in (None, '', ' ')):
                return artist
        return None

    def _expand_duplicate_indices(self, base_indices):
        expanded = []
        for base_idx in np.atleast_1d(base_indices):
            base_idx = int(base_idx)
            expanded.extend(self._duplicate_lookup.get(base_idx, (base_idx,)))
        return np.array(expanded, dtype=int)

    def _remove_local_indices(self, remove_indices):
        if not remove_indices:
            return
        remove_set = {int(idx) for idx in remove_indices if 0 <= int(idx) < len(self.artists)}
        if not remove_set:
            return

        new_artists = []
        new_artists_in_plot = []
        new_path_features = []
        new_indexes = []

        for local_idx, (artist, preview, feature, base_idx) in enumerate(zip(self.artists, self.artists_in_plot, self.path_features, self.indexes)):
            if local_idx in remove_set:
                if getattr(preview, 'axes', None) is not None:
                    preview.remove()
                self._duplicate_lookup.pop(int(base_idx), None)
            else:
                new_artists.append(artist)
                new_artists_in_plot.append(preview)
                new_path_features.append(feature)
                new_indexes.append(int(base_idx))

        self.artists = new_artists
        self.artists_in_plot = new_artists_in_plot
        self.path_features = new_path_features
        self.indexes = np.array(new_indexes, dtype=int) if new_indexes else np.array([], dtype=int)

    def _commit_selection(self, matched_local_indices, label, match_mode=None):
        local_indices = sorted({int(idx) for idx in np.atleast_1d(matched_local_indices) if 0 <= int(idx) < len(self.artists)})
        if not local_indices:
            self.state = 0
            self.fig.suptitle(self._idle_title)
            self.ax['group'].clear()
            self.ax['group'].set_title('')
            self.fig.canvas.draw_idle()
            return False

        base_indices = self.indexes[local_indices]
        target_indices = self._expand_duplicate_indices(base_indices)
        self.types[target_indices] = label

        if label == 's' and match_mode is not None:
            self.known_markers.append({
                'match_by': match_mode,
                'feature': self.path_feature,
                'shape_tol': self.shape_tol,
                'color_tol': self.color_tol,
            })

        self._remove_local_indices(local_indices)
        self._marker_preview_idx = None
        self._group_preview_idxs = None
        self.ax['marker'].clear()
        self.ax['group'].clear()
        self.ax['group'].set_title('')
        self.matched_idxs = []
        self.state = 0
        self.fig.suptitle(self._idle_title)
        self.fig.canvas.draw_idle()
        return True
        
    def onkeyrelease(self, event):
        if self.state == 0: #
            if event.key == 'f': # finish?
                self.state = 99
                self.fig.suptitle('[F]inish? (press "f" again to confirm)')
        elif self.state == 99:
            if event.key == 'f':
                plt.close(self.fig)
                self.finished = True
            else:
                self.state = 0
                self.fig.suptitle(self._idle_title)
        elif self.state >= 1 and self.state <= 9: # currently handling an object
            if event.key in ('c', 'escape'): # cancelled
                self.state = 0
                self.fig.suptitle(self._idle_title)
                self._marker_preview_idx = None
                self._group_preview_idxs = None
                self.matched_idxs = []
                self.ax['marker'].clear()
                self.ax['group'].clear()
                self.ax['group'].set_title('')

            elif self.state == 1 and event.key in ('delete', 'del', 'backspace', 'n'):
                if self._marker_preview_idx is not None:
                    self._commit_selection([self._marker_preview_idx], 'd')

            elif self.state == 1 and event.key in 'sldo': # have just chosen object type
                self.type = event.key
                # next step, choose how to match similar objects
                self.fig.suptitle('chosen "{}". match [S]hape, c[O]lor, co[L]or+shape, or [C]ancel'.format(self.__class__.type_names[self.type]))
                self.state = 2

            elif self.state == 2 and event.key in 'sol':
                self.match_mode = event.key
                self.matched_idxs = select_paths(
                    self.path_feature,
                    self.path_features,
                    modes=self.match_mode,
                    pos_tol=self.shape_tol,
                    color_tol=self.color_tol,
                )
                self.matched_idxs = self._filter_matches(self.matched_idxs)
                warntxt = ''
                if self.type == 's':
                    warn_artist = self._find_line_like_match(self.matched_idxs)
                    if warn_artist is not None:
                        warntxt = '\n(WARNING: elements labelled as "scatter", but at least one is line-like)'
                self._draw_group_preview(self.matched_idxs)
                self.fig.suptitle(f'press any key to continue or [C]ancel{warntxt}')
                self.state = 3

            elif self.state == 3:
                self._commit_selection(self.matched_idxs, self.type, match_mode=self.match_mode)
        else:
            return

        self.fig.canvas.draw()
            
    def save(self, basepath, yes=False):
        # save information to file
        type_path = basepath + '.typ'
        if not yes and os.path.exists(type_path):
            pause_and_warn('File "{}" already exists!'.format(type_path), choose='overwrite existing files?',
                           default='n', yes_message='overwritten', no_message='raise')
        with open(type_path, 'wb') as f:
            f.write(self.types.tobytes())
        
        marker_path = basepath + '.mkr'
        if not yes and os.path.exists(marker_path):
            pause_and_warn('File "{}" already exists!'.format(type_path), choose='overwrite existing files?',
                           default='n', yes_message='overwritten', no_message='raise')
        
        with open(marker_path, 'w') as f:
            json.dump(self.known_markers, f, #ensure_ascii=True, indent=2,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                      )
    
    @staticmethod
    def load(basepath):
        type_path = basepath + '.typ'
        # with open(type_path, 'rb') as f:
        #     types = np.frombuffer(f.read(), dtype='S1')
        with open(type_path) as f:
            types = f.read()
        
        marker_path = basepath + '.mkr'
        with open(marker_path) as f:
            known_markers = json.load(f)
            
        return types, known_markers
    
class RectObjectSelector(RectSelector):
    MODE_STYLES = {
        'touch': {'title': 'Keep touched objects', 'color': '#2ca02c'},
        'subtract': {'title': 'Remove touched objects', 'color': '#d62728'},
    }

    def init(self, objects, ax=None, mode='touch'):
        super().init()
        if ax is None:
            ax = getattr(self.fig, 'ax', None)
            if ax is None:
                if not self.fig.axes:
                    raise ValueError('expected an Axes for RectObjectSelector')
                ax = self.fig.axes[0]

        self.display_ax = ax
        self.objects = deepcopy(objects)
        self.orig_objects = objects
        self.selected = {}
        for typ, typ_objs in objects.items():
            self.selected[typ] = np.full(len(typ_objs), True, dtype=bool)

        self._held_mode = None
        self.mode = None

        plot_objects(self.objects, ax=self.display_ax)

        self._set_mode(mode)

    def _set_mode(self, mode, *, force=False):
        normalized = normalize_rect_mode(mode)
        if not force and normalized == self.mode:
            return

        self.mode = normalized
        style = self.MODE_STYLES[self.mode]
        for rect in self.rects.values():
            rect.set_color(style['color'])
        self._update_title()

    def _update_title(self):
        style = self.MODE_STYLES[self.mode]
        if self.mode == 'touch':
            hint = 'Press [M] to remove, hold Alt to temporarily remove, [R] to reset.'
        else:
            hint = 'Press [M] to keep instead, [R] to reset.'
        self.display_ax.set_title(f"Mode: {style['title']}. {hint}")
        self.fig.canvas.draw_idle()

    def _toggle_mode(self):
        next_mode = 'subtract' if self.mode != 'subtract' else 'touch'
        self._set_mode(next_mode)

    def onrelease(self, event):
        super().onrelease(event)

        touched = rect_filter_objects(self.objects, self.x0, self.x1, self.y0, self.y1, mode=self.mode)
        self.last_selected = touched

        for typ, typ_objs in self.objects.items():
            if self.mode == 'subtract':
                to_hide = np.where(self.selected[typ] & touched[typ])[0]
            else:
                to_hide = np.where(self.selected[typ] & ~touched[typ])[0]
            for idx in to_hide:
                typ_objs[idx]['artist'].set_visible(False)
                self.selected[typ][idx] = False

        self.fig.canvas.draw_idle()

    def onkeypress(self, event):
        if event.key == 'alt' and self.mode != 'subtract' and self._held_mode is None:
            self._held_mode = self.mode
            self._set_mode('subtract')

    def onkeyrelease(self, event):
        if event.key == 'm':
            self._held_mode = None
            self._toggle_mode()
        elif event.key == 'r':
            self._held_mode = None
            for typ, typ_objs in self.objects.items():
                self.selected[typ] = np.full(len(typ_objs), True, dtype=bool)
                for typ_obj in typ_objs:
                    typ_obj['artist'].set_visible(True)
            self.fig.canvas.draw_idle()
            self._update_title()
        elif event.key == 'alt' and self._held_mode is not None:
            self._set_mode(self._held_mode)
            self._held_mode = None
            
    def get_filtered_objects(self):
        # print(self.selected)
        return get_filtered_objects(self.orig_objects, self.selected)

    def _finish_selection(self, _event=None):
        if self.finished:
            return

        self.finished = True
        self.display_ax.set_title('Selection complete. Close window to continue.')
        self.fig.canvas.draw_idle()
        plt.close(self.fig)
    
    
class DataExtractor(BaseEventHandler):
    def init(self, objects, ax0, ax1, axbox, pdf_path=None):
        self.exportpath = pdf_path + '.out'
        if os.path.exists(self.exportpath):
            pause_and_warn('File "{}" already exists: this file contains data you have exported.'.format(self.exportpath), choose='overwrite existing file?',
                           default='n', yes_message='', no_message='raise', warn=False)
        
        self.objects = objects
        self.ax0 = ax0
        self.ax1 = ax1
        self.axbox = axbox
        
        self.textbox = TextBox(self.axbox, '', textalignment="center")
        self.textbox.on_submit(self.onsubmit)
        
        # calibration line artists
        self.xcals = []
        self.ycals = []
        self.select_rect = None

        self.xscale = None
        self.yscale = None
        self._calibration_suggestions = {'x': None, 'y': None}
        
        self.export_data = {
            'meta': {
                'vpextractor_version': __version__,
                },
            }
        
        self.axes = {} # data axes information, not real axes for plot
        self._ca = None # currect data axis number
        self._next_axis = None # the next axis to be changed to

        self.select_mode = 'touch'
        
        if pdf_path is not None:
            self.savepath = pdf_path + '.axes'
            # if os.path.exists(self.savepath):
            #     pause_and_warn(f"'{self.savepath}' already exists!", choose='overwrite?')
            self.axes.update(self.load())
        else:
            raise NotImplementedError('please input pdf_path')
        
        self._display_objects = deepcopy(self.objects)
        plot_objects(self._display_objects, ax=self.ax0, optimize_preview=True)

        self.set_status(-1)
        
    @property
    def ca(self): # currect data axis dict
        return self.axes[self._ca]
    
    status_title = {
        -1: '[A]dd an axis, input number of any saved axis, [E]xport all axes, or [S]ave',
        # 100: 'axis #%ca: click on an axis tick/data plot, or manally set [X]-axis/[Y]-axis calibration, \n'\
        #     'set [A]xis region, [S]ave, [E]xport, d[U]plicate, or e[X]it axis',
        100: 'axis #%ca: click on an axis tick/data plot, or: \n'\
            'set [A]xis region, [S]ave, [E]xport, d[U]plicate, or e[X]it axis',
        110: 'input x value in textbox, or [C]ancel',
        111: 'input y value in textbox, or [C]ancel',
        120: 'change x value, [D]elete, or [C]ancel',
        121: 'change y value, [D]elete, or [C]ancel',
        130: 'axis #%ca copied to axis #%na. press Enter to change to #%na',
        140: 'drag to select',
        }
    
    def set_status(self, code, **kwargs):
        self.status = code
        title = self.__class__.status_title[code]
        title = title.replace('%ca', str(self._ca))
        title = title.replace('%na', str(self._next_axis))
        for old, new in kwargs:
            title = title.replace(old, new)
        self.fig.suptitle(title)
    
    def onpick(self, event):
        if self.status == 100: # default state with axes activated
            for i, cal_artists in enumerate(self.xcals):
                # print(cal_artists.values())
                # print(self.xcals)
                if event.artist in cal_artists.values():
                    self.textbox.set_active(True)
                    # print(i, self.ca['x_cal'])
                    self.textbox.set_val(self.ca['x_cal']['data'][i])
                    self.set_status(120)
                    self.fig.canvas.draw()
                    self.changecal_idx = i # index of the activage cal
                    return
                
            for i, cal_artists in enumerate(self.ycals):
                if event.artist in cal_artists.values():
                    self.textbox.set_active(True)
                    self.textbox.set_val(self.ca['y_cal']['data'][i])
                    self.set_status(121)
                    self.fig.canvas.draw()
                    self.changecal_idx = i # index of the activage cal
                    return
        
            for obj in self.objects['u']:
                if event.artist is obj['artist']: # is it an axis label?
                    x, y = obj['coords']
                    x, y = np.unique(x), np.unique(y)
                    if x.size == 1: # x-axis
                        self.x = x[0]
                        self.textbox.label.set_text('x value:')
                        self.textbox.set_active(True)
                        # self.textbox._rendercursor()
                        # self.textbox.begin_typing()
                        self.set_status(110)
                    elif y.size == 1: # y-axis
                        self.y = y[0]
                        self.textbox.label.set_text('x value:')
                        self.textbox.set_active(True)
                        # self.textbox._rendercursor()
                        # self.textbox.begin_typing()
                        self.set_status(111)
                    self.fig.canvas.draw()
                    return
                
    def onkeypress(self, event):
        if self.status == -1: # initial state
            if event.key in '0123456789': # axis number
                self.fig.suptitle('available axes numbers include: ' + ' '.join(self.axes.keys()))
        
            self.fig.canvas.draw()
    
    def _change_current_axis(self, n):
        self._ca = n
                    
        for pos, data in zip(self.ca['x_cal']['pos'], self.ca['x_cal']['data']):
            self.xcals.append(annotate(x=pos, xtxt=f'{data:.2g}', ax=self.ax0))
        for pos, data in zip(self.ca['y_cal']['pos'], self.ca['y_cal']['data']):
            self.ycals.append(annotate(y=pos, ytxt=f'{data:.2g}', ax=self.ax0))
        
        self.set_status(100)
        
        self.calibrate()
        self.plot_data()
        
    def _exit_current_axis(self):
        for cal in chain(self.xcals, self.ycals):
            for artist in cal.values():
                artist.remove()
        self.xcals.clear()
        self.ycals.clear()
        if self.select_rect is not None:
            self.select_rect.remove()
        self.select_rect = None
        self.set_status(-1)
        self.xscale = None
        self.yscale = None
    
    def onkeyrelease(self, event):
        if self.status == -1: # initial state
            if event.key in '0123456789': # axis number
                if event.key not in self.axes: 
                    self.set_status(-1)
                else: # load one saved axis
                    self._change_current_axis(event.key)
                    
            elif event.key == 'a':
                for n in '0123456789':
                    if n not in self.axes:
                        self.axes[n] = {
                            'x_cal': {
                                'pos': [], # x position on the plot
                                'data': [], # real data
                                },
                            'y_cal': {
                                'pos': [], # y position on the plot
                                'data': [], # real data
                                },
                            'xlim': [-np.inf, np.inf],
                            'ylim': [-np.inf, np.inf],
                            }
                        self._ca = n
                        self.set_status(100)
                        break
                else:
                    raise NotImplementedError('maximum number of axes exceeded')
            elif event.key == 's':
                self.save()
                self.fig.suptitle(f"axis information saved to '{self.savepath}'")
                plt.pause(2)
                self.set_status(self.status)
            elif event.key == 'e': # export all
                self.export()
                self.fig.suptitle(f"data exported to '{self.exportpath}'")
                plt.pause(2)
                self.set_status(self.status)
            self.fig.canvas.draw()
        elif self.status // 100 == 1: # in axis mode
            if self.status in [110, 111] and event.key == 'c':
                self.set_status(100)

            elif self.status in [120, 121]: # editing calibration
                if event.key == 'c': # cancel
                    pass
                elif event.key == 'd': # delete
                    i = self.changecal_idx
                    if self.status == 120:
                        self.ca['x_cal']['data'].pop(i)
                        self.ca['x_cal']['pos'].pop(i)
                        for artist in self.xcals.pop(i).values():
                            artist.remove()
                    elif self.status == 121:
                        self.ca['y_cal']['data'].pop(i)
                        self.ca['y_cal']['pos'].pop(i)
                        for artist in self.ycals.pop(i).values():
                            artist.remove()
                    self.calibrate()
                    self.plot_data()
                
                if event.key in 'cd':
                    self.textbox.set_active(False)
                    self.set_status(100)
                    self.textbox.set_val('')
            elif self.status == 100:
                if event.key == 'a': # set axis region
                    self.set_status(140)
                    with RectSelector(fig=self.fig) as rs:
                        rs.wait()
                    self.set_status(100)
                    
                    if self.select_rect is not None:
                        self.select_rect.remove()
                    self.select_rect = rs.rects[rs.ax] # the selection rectangle artist
                    self.ca['xlim'] = [rs.x0, rs.x1]
                    self.ca['ylim'] = [rs.y0, rs.y1]
                    
                    self.plot_data()
                elif event.key == 'f':
                    for axis in ('x', 'y'):
                        if self._apply_calibration_suggestion(axis):
                            break
                elif event.key == 'u': # duplicate axis
                    for n in '0123456789':
                        if n not in self.axes:
                            self.axes[n] = deepcopy(self.ca)
                            self._next_axis = n
                            self.set_status(130)
                            break
                    else:
                        raise NotImplementedError('maximum number of axes exceeded')
                elif event.key == 'e': # export all
                    self.export()
                    self.fig.suptitle(f"data exported to '{self.exportpath}'")
                    plt.pause(2)
                    self.set_status(self.status)
            elif self.status == 130:
                if event.key == 'enter':
                    self._exit_current_axis()
                    self._change_current_axis(self._next_axis)
                
            if event.key == 'x': # exit axis
                self._exit_current_axis()
                
            elif event.key == 's':
                self.save()
                self.fig.suptitle(f"axis information saved to '{self.savepath}'")
                plt.pause(2)
                self.set_status(self.status)
            self.fig.canvas.draw()
            
    def onsubmit(self, expression):
        if self.status in [110, 111, 120, 121]:
            try:
                num = float(expression)
            except ValueError:
                if expression:
                    print(f'expected a number, got "{expression}"')
                return
            
            if self.status == 110: # setting x value
                xdata = num
                self.ca['x_cal']['pos'].append(self.x)
                self.ca['x_cal']['data'].append(xdata)
                self.xcals.append(annotate(x=self.x, xtxt=f'{xdata:.2g}', ax=self.ax0))
            elif self.status == 111: # setting y value
                ydata = num
                self.ca['y_cal']['pos'].append(self.y)
                self.ca['y_cal']['data'].append(ydata)
                self.ycals.append(annotate(y=self.y, ytxt=f'{ydata:.2g}', ax=self.ax0))
            elif self.status == 120: # editing x value
                xdata = num
                self.ca['x_cal']['data'][self.changecal_idx] = xdata
                self.xcals[self.changecal_idx]['vtext'].set_text(xdata)
            elif self.status == 121: # editing y value
                ydata = num
                self.ca['y_cal']['data'][self.changecal_idx] = ydata
                # print(self.ycals[self.changecal_idx]['htext'])
                self.ycals[self.changecal_idx]['htext'].set_text(ydata)
            
            self.set_status(100)
            self.textbox.set_active(False)
            self.textbox.set_val('')
            
            self.calibrate()
            self.plot_data()

            self.fig.canvas.draw()
    
    def calibrate(self):
        self.xscale = None
        self.yscale = None
        self._calibration_suggestions['x'] = None
        self._calibration_suggestions['y'] = None
        error_messages = []
        # calibrate axes
        try:
            xs, xds = self.ca['x_cal']['pos'], self.ca['x_cal']['data']
            self.xk, self.xb, self.xscale = self.__class__.get_coeffs_auto(xs, xds)
            print(f'calibration: got {self.xk} x + {self.xb}, {self.xscale} scale')
        except ConsistencyError:
            errmsg = self._handle_calibration_failure('x', xs, xds)
            error_messages.append(errmsg)
        try:
            ys, yds = self.ca['y_cal']['pos'], self.ca['y_cal']['data']
            self.yk, self.yb, self.yscale = self.__class__.get_coeffs_auto(ys, yds)
            print(f'calibration: got {self.yk} y + {self.yb}, {self.yscale} scale')
            # print(self.yk, self.yb, self.yscale)
        except ConsistencyError:
            errmsg = self._handle_calibration_failure('y', ys, yds)
            error_messages.append(errmsg)

        if error_messages:
            errmsg = '\n'.join(error_messages)
            self.fig.suptitle(f'ERROR: {errmsg}')
            self.fig.canvas.draw()
        else:
            self.set_status(self.status)
           
    scale_func = {
        'linear': lambda x: x,
        'log': np.log10,
        } 
    scale_inv_func = {
        'linear': lambda x: x,
        'log': lambda x: 10**np.array(x),
        } 

    @classmethod
    def get_coeffs_auto(cls, xs, xds, err=1e-5):
        xs = np.asarray(xs, dtype=float)
        xds = np.asarray(xds, dtype=float)
        if xs.size != xds.size:
            raise ValueError('expected xs, xds with the same shape')
        if xs.size < 2:
            return None, None, None

        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(xds))):
            raise ConsistencyError(f'inconsistent data: {xs} and {xds}')

        dx = np.diff(xs)
        if np.any(dx == 0):
            raise ConsistencyError(f'inconsistent data: {xs} and {xds}')

        # TODO: support interpolation calibration?
        # automatically choose linear or log scale, and check consistency
        for scale, xfunc in cls.scale_func.items():
            if scale == 'log' and np.any(xds <= 0):
                continue
            try:
                transformed = xfunc(xds)
            except (FloatingPointError, ValueError):
                continue
            if not np.all(np.isfinite(transformed)):
                continue
            ks = np.diff(transformed) / dx
            if not np.all(np.isfinite(ks)):
                continue

            # 1: unique
            # k = np.unique(ks)
            # if k.size == 1:
            #     k = k[0]
            #     b = xds[0] - k * xs[0]
            #     return k, b, scale

            # 2: allow error
            k = np.mean(ks)
            if k == 0 or not np.isfinite(k):
                continue
            spread = np.max(ks) - np.min(ks)
            if spread / np.abs(k) < err:
                b = np.mean(transformed[:-1] - k * xs[:-1])
                if not np.isfinite(b):
                    continue
                return k, b, scale

        else:
            raise ConsistencyError(f"inconsistent data: {xs} and {xds}")

    def _handle_calibration_failure(self, axis, xs, xds):
        errmsg = f'inconsistent calibration for {axis} axis: {xs}, {xds}'
        suggestion = self._compute_calibration_suggestion(axis, xs, xds)
        self._calibration_suggestions[axis] = suggestion
        if suggestion is not None:
            idx = suggestion['index'] + 1
            current = suggestion['current']
            proposed = suggestion['suggested']
            delta = proposed - current
            scale = suggestion['scale']
            suggestion_msg = (f"Suggested {axis}-axis tick #{idx}: {current:.6g} → "
                              f"{proposed:.6g} (Δ={delta:.2g}, scale={scale}). "
                              "Press [F] to apply.")
            full_msg = f"{errmsg}\n{suggestion_msg}"
        else:
            suggestion_msg = 'click on calibration line to edit'
            full_msg = f"{errmsg}\n{suggestion_msg}"
        print(full_msg)
        return full_msg

    def _compute_calibration_suggestion(self, axis, xs, xds):
        xs = np.asarray(xs, dtype=float)
        xds = np.asarray(xds, dtype=float)
        if xs.size < 2:
            return None

        best = None
        for scale, xfunc in self.scale_func.items():
            if scale == 'log' and np.any(xds <= 0):
                continue
            try:
                transformed = xfunc(xds)
            except (FloatingPointError, ValueError):
                continue
            if not np.all(np.isfinite(transformed)):
                continue
            A = np.vstack([xs, np.ones_like(xs)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, transformed, rcond=None)
            except np.linalg.LinAlgError:
                continue
            fitted = A @ coeffs
            residuals = transformed - fitted
            rms = np.sqrt(np.mean(residuals**2))
            idx = int(np.argmax(np.abs(residuals)))
            suggested_value = self.scale_inv_func[scale](fitted[idx])
            suggested_value = np.asarray(suggested_value)
            if suggested_value.size != 1:
                try:
                    suggested_value = float(suggested_value.item())
                except ValueError:
                    continue
            else:
                suggested_value = float(suggested_value)
            if scale == 'log' and suggested_value <= 0:
                continue
            suggestion = {
                'axis': axis,
                'scale': scale,
                'index': idx,
                'current': float(xds[idx]),
                'suggested': suggested_value,
                'residual': float(residuals[idx]),
                'rms': float(rms),
            }
            if best is None or suggestion['rms'] < best['rms']:
                best = suggestion
        return best

    def _apply_calibration_suggestion(self, axis):
        suggestion = self._calibration_suggestions.get(axis)
        if suggestion is None:
            return False
        idx = suggestion['index']
        new_value = suggestion['suggested']
        cal_key = f'{axis}_cal'
        self.ca[cal_key]['data'][idx] = new_value
        if axis == 'x':
            if idx < len(self.xcals) and 'vtext' in self.xcals[idx]:
                self.xcals[idx]['vtext'].set_text(f'{new_value:.2g}')
        elif axis == 'y':
            if idx < len(self.ycals) and 'htext' in self.ycals[idx]:
                self.ycals[idx]['htext'].set_text(f'{new_value:.2g}')
        self._calibration_suggestions[axis] = None
        print(f"applied suggested correction to {axis}-axis tick #{idx + 1}: {new_value:.6g}")
        self.calibrate()
        self.plot_data()
        self.fig.canvas.draw_idle()
        return True
        
    @staticmethod
    def get_coeffs(x1, x2, xd1, xd2, scale='linear'):
        if scale == 'linear':
            pass
        elif scale == 'log':
            x1, x2 = np.log10(x1), np.log(x2)
        else:
            raise ValueError(f"unknown scale '{scale}'")
        
        k = (xd2 - xd1) / (x2 - x1)
        b = xd1 - k * x1
    
        return k, b
    
    def get_data(self):
        # get calibrated data
        x0, x1 = self.ca['xlim']
        y0, y1 = self.ca['ylim']
        selected = rect_filter_objects(self.objects, x0, x1, y0, y1, mode=self.select_mode)
        
        out_data = {'l': [], 's': []}
        out_info = {'l': [], 's': []}
        
        self.export_data[self._ca] = {'lines': [], 'scatters': []}
        typecode_translate = {'l': 'lines', 's': 'scatters'}
        
        for typ in ['l', 's']: # line, scatter
            for obj, sel in zip(self.objects[typ], selected[typ]):
                if sel:
                    data_coords = self.transform(*obj['coords'])
                    out_data[typ].append(data_coords)
                    info = {'linestyle': dedup(obj['artist'].get_linestyle()),
                            'linewidth': dedup(obj['artist'].get_linewidth())}
                    info.update(get_color(obj['artist']))
                    # if typ == 's':
                    #     info.update({'s': obj['artist'].get_sizes()})
                    out_info[typ].append(info)
                    
                    export_dict = {
                        'x': data_coords[0],
                        'y': data_coords[1],
                        }
                    export_dict.update(info)
                    self.export_data[self._ca][typecode_translate[typ]].append(export_dict)
                    
        # self.export_data[self._ca] = {
        #     'axis_number': self._ca,
        #     'lines_data': out_data['l'],
        #     'lines_info': out_info['l'],
        #     'scatters_data': out_data['s'],
        #     'scatters_info': out_info['s'],
        #     }
        return out_data, out_info
    
    def plot_data(self):
        # plot calibrated data
        if self.xscale is not None and self.yscale is not None:
            self.ax1.clear()
            out_data, out_info = self.get_data()
            # print(out_data, out_info)
            for (x, y), info in zip(out_data['s'], out_info['s']):
                self.ax1.scatter(x, y, fc=info['facecolor'], ec=info['edgecolor']) # , s=info['s']
            for (x, y), info in zip(out_data['l'], out_info['l']):
                self.ax1.plot(x, y, color=info['color'], linestyle=info['linestyle'], linewidth=info['linewidth'])
            self.ax1.set_xscale(self.xscale)
            self.ax1.set_yscale(self.yscale)
            self.ax1.grid()
        
    def transform(self, x, y):
        x, y = np.array(x), np.array(y)
        func = self.__class__.scale_inv_func
        return [func[self.xscale](self.xk * x + self.xb),
                func[self.yscale](self.yk * y + self.yb)]
    
    def save(self):
        # print(self.axes)
        with open(self.savepath, 'w') as f:
            json.dump(self.axes, f,
                      indent=2,
                      # default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                      )
        print(f"axis information saved to '{self.savepath}'")
            
    def load(self):
        if not os.path.exists(self.savepath):
            return {}
        with open(self.savepath) as f:
            return json.load(f)
        
    def export(self):
        for ca in self.axes:
            self._ca = ca
            self.calibrate()
            if self.xscale and self.yscale:
                self.get_data()
        with open(self.exportpath, 'w') as f:
            json.dump(self.export_data, f,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                      )
        print(f"data exported to '{self.exportpath}'")
        
            