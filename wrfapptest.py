#!/usr/bin/env python3

'''
WRF Viewer - Single file desktop app (PySide6 + Matplotlib + Cartopy)

Features (MVP)
    - Open one or many wrfout_* NetCDF files (concatenated in time order).
    - Variables: MDBZ, RAINNC, RAINC, WSPD10, REFL1KM, 2 m Temp (°F).
    - Time slider with smooth live redraw (debounced) + play/pause animation.
    - Colormap dropdown
    - Export current frame as PNG.
    - Inline **Status Bar** progress for preloading (no popup)
    - Fast Scrubbing: resuses pcolormesh + 
Tested libs: wrf-python, netCDF4, numpy, xarray (optional), cartopy, matplotlib, pint, PySide6

Pack to an EXE (Windows) or app bundle (macOS) with PyInstaller, see bottom of file for notes.
'''

from __future__ import annotations
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
import numpy as np
import os
import shutil
import sys
import typing as T

from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from numpy import float32
from netCDF4 import Dataset
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider, QSplitter, QToolBox, QVBoxLayout, QWidget)
from wrf import to_np # getvar, latlon_coords, ALL_TIMES, interplevel


# ------------------------
# Data Structure
# ------------------------
@dataclass
class WRFFrame:
    path: str
    time_index: int
    timestamp_str: str


@dataclass(frozen=True)
class UpperAirSpec:
    canonical: str
    display_name: str
    level_hpa: float
    shading_field: str
    colorbar_label: str
    vmin: T.Optional[float] = None
    vmax: T.Optional[float] = None
    contour_field: T.Optional[str] = None
    contour_levels: T.Optional[T.Sequence[float]] = None
    contour_color: str = 'black'
    contour_width: float =  0.8
    barb_stride: int = 12
    barb_length: float = 6.0
    barb_color: str = 'black'
    title: str = ''


@dataclass
class UpperAirData:
    scalar: np.ndarray
    contour: T.Optional[np.ndarray]
    u: np.ndarray
    v: np.ndarray
    
    
# Use nearly a full step for intensity shading inside each precipitation
# category so the colorbar ticks (0.0, 0.01, 0.05, 0.25, 0.5 in/hr) land at the
# expected quarter/half/three-quarter heights of each band (0.0 at the base,
# 0.5 near the top). Leave a tiny margin below 1.0 to avoid crossing the next
# category boundary when values are clipped.
PTYPE_INTENSITY_SPAN = 0.995
PTYPE_MAX_RATE_INHR = 0.5


def _ptype_rate_offset(rate: np.ndarray | float) -> np.ndarray | float:
    ''' Map precipitation rate (in/hr) to an intensity offset inside the band.
    
    Rates are clamped to [0.0, 0.5] in/hr and then interpolated so 0.01, 0.05, 0.25,
    and 0.5 in/hr land at 1/4, 1/2, 3/4, and full intensity respectively.
    '''
    
    rate_arr = np.asarray(rate, dtype=float32)
    break_rates = np.array([0.0, 0.01, 0.05, 0.25, PTYPE_MAX_RATE_INHR], dtype=float32)
    break_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float32) * PTYPE_INTENSITY_SPAN
    clamped = np.clip(rate_arr, break_rates[0], break_rates[-1])
    offset = np.interp(clamped, break_rates, break_positions)
    if np.isscalar(rate):
        return float(offset)
    return offset.astype(float32)


UPPER_AIR_SPECS: dict[str, UpperAirSpec] = {
    'HGT500': UpperAirSpec(
        canonical='HGT500',
        display_name='500 mb Hgt/Wind',
        level_hpa=500.0,
        shading_field='height',
        colorbar_label='500 hPa Geopotential Height (m)',
        vmin=4800.0,
        vmax=6000.0,
        contour_field='height',
        contour_levels=np.arange(4800.0, 6000.1, 60.0),
        contour_color='black',
        contour_width=0.9,
        barb_stride=16,
        barb_length=5.5,
        barb_color='black',
        title='500 hPa Height & Wind',
    ),
    'RH700': UpperAirSpec(
        canonical='RH700',
        display_name='700 mb RH/Wind',
        level_hpa=700.0,
        shading_field='rh',
        colorbar_label='700 hPa Relative Humidity (%)',
        vmin=0.0,
        vmax=100.0,
        contour_field='height',
        contour_levels=np.arange(2800.0, 3400.1, 60.0),
        contour_color='black',
        contour_width=0.8,
        barb_stride=16,
        barb_length=5.5,
        barb_color='black',
        title='700 hPa Relative Humidity & Wind',
    ),
    'TEMP850': UpperAirSpec(
        canonical='TEMP850',
        display_name='850 mb Temp/Wind',
        level_hpa=850.0,
        shading_field='temperature',
        colorbar_label='850 hPa Temperature (°C)',
        vmin=-30.0,
        vmax=30.0,
        contour_field='height',
        contour_levels=np.arange(1300.0, 1700.1, 30.0),
        contour_color='white',
        contour_width=1.0,
        barb_stride=16,
        barb_length=5.5,
        barb_color='black',
        title='850 hPa Temperature & Wind',
    ),
}

class WRFLoader(QtCore.QObject):
    '''
    Loads wrfout files, make a timeline of frames, and provide 2D fields.
    Supports optional full preloading for smooth scrubbing.
    '''
    def __init__(self):
        super().__init__()
        self.files: list[str] = []
        self.frames: list[WRFFrame] = []
        self._cache: dict[tuple[str, str, int], T.Any] = {}
        self._geo_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Preload store: {(VAR, level_or_None): [object | None per frame]}
        self.preloaded: dict[tuple[str, T.Optional[float]], list[T.Optional[T.Any]]] = {}
        self._pressure_orientation: dict[str, str] = {}
        self._upper_base_cache: OrderedDict[tuple[str, int], dict[str, np.ndarray]] = OrderedDict()
        self._upper_base_cache_limit: int = 3
    
    @staticmethod
    def _log_debug(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)
    
    @staticmethod
    def _format_timestamp(ts: str) -> str:
        ts_clean = ts.replace('_', ' ')
        try:
            dt = datetime.strptime(ts_clean, '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            return ts
    
    # --- Loading and Indexing ---
    def open_files(self, paths: list[str]) -> None:
        if not paths:
            return
        # Expand directories and globs
        expanded: list[str] = []
        for p in paths:
            if os.path.isdir(p):
                expanded.extend(sorted(glob.glob(os.path.join(p, 'wrfout_*'))))
            else:
                expanded.extend(glob.glob(p))
        self.files = [p for p in expanded if os.path.exists(p)]
        if not self.files:
            raise FileNotFoundError('No wrfout files found.')
        
        frames: list[WRFFrame] = []
        for fp in self.files:
            with Dataset(fp) as nc:
                if 'Times' not in nc.variables:
                    raise RuntimeError(f'Times variable missing in {fp}')
                times_arr = nc.variables['Times'][:]
                for ti in range(times_arr.shape[0]):
                    ts = times_arr[ti].tobytes().decode('utf-8').strip().replace('\x00', '')
                    formatted_ts = self._format_timestamp(ts)
                    frames.append(WRFFrame(path=fp, time_index=ti, timestamp_str=formatted_ts))
        frames.sort(key=lambda fr: (os.path.getmtime(fr.path), fr.time_index))
        self.frames = frames
        self._cache.clear()
        self._geo_cache.clear()
        self.preloaded.clear()
        self._pressure_orientation.clear()
        self._upper_base_cache.clear()
        
    # --- Geometry ---
    def get_geo(self, frame: WRFFrame) -> tuple[np.ndarray, np.ndarray]:
        fp = frame.path
        if fp in self._geo_cache:
            return self._geo_cache[fp]
        with Dataset(fp) as nc:
            if 'XLAT' in nc.variables and 'XLONG' in nc.variables:
                lat = np.array(nc.variables['XLAT'][0, :, :])
                lon = np.array(nc.variables['XLONG'][0, :, :])
            else:
                # Rare fallback: derive via wrf-python (avoid caching)
                from wrf import latlon_coords, getvar
                pres = getvar(nc, 'pressure', timeidx=frame.time_index, cache=False)
                lats, lons = latlon_coords(pres)
                lat = to_np(lats)
                lon = to_np(lons)
        self._geo_cache[fp] = (lat, lon)
        return lat, lon
    
    # --- Utilities for raw access ---
    def _slice_time_var(self, var_obj, time_index: int) -> np.ndarray:
        dims = tuple(getattr(var_obj, 'dimensions', ()))
        if 'Time' in dims:
            axis = dims.index('Time')
            slicer = [slice(None)] * var_obj.ndim
            slicer[axis] = time_index
            data = np.array(var_obj[tuple(slicer)])
        else:
            data = np.array(var_obj[:])
        return data
    
    def _destagger(self, arr: np.ndarray, axis: int) -> np.ndarray:
        slicer1 = [slice(None)] * arr.ndim
        slicer2 = [slice(None)] * arr.ndim
        slicer1[axis] = slice(0, -1)
        slicer2[axis] = slice(1, None)
        return 0.5 * (arr[tuple(slicer1)] + arr[tuple(slicer2)])
    
    def _calc_pressure(self, nc: Dataset, time_index: int) -> np.ndarray:
        p = self._slice_time_var(nc.variables['P'], time_index)
        pb = self._slice_time_var(nc.variables['PB'], time_index)
        return (p + pb).astype(float32)
    
    def _calc_height(self, nc: Dataset, time_index: int) -> np.ndarray:
        ph = self._slice_time_var(nc.variables['PH'], time_index)
        phb = self._slice_time_var(nc.variables['PHB'], time_index)
        z_full = (ph + phb) / 9.81
        return 0.5 * (z_full[:-1, :, :] + z_full[1:, :, :])
    
    def _calc_temperature(self, nc: Dataset, pressure: np.ndarray, time_index: int) -> np.ndarray:
        theta = self._slice_time_var(nc.variables['T'], time_index) + 300.0
        temp_k = theta * (pressure / 100000.0) ** (287.0 / 1004.0)
        return temp_k.astype(float32)
    
    def _calc_relative_humidity(self, temp_k: np.ndarray, pressure: np.ndarray, qv: np.ndarray) -> np.ndarray:
        eps = 0.622
        vap_press = (qv * pressure) / (eps + qv + 1e-12)
        temp_c = temp_k - 273.15
        es = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        es = np.maximum(es, 1.0)
        rh = (vap_press / es) * 100.0
        return np.clip(rh, 0.0, 100.0).astype(float32)
    
    def _get_upper_base_fields(self, frame: WRFFrame) -> dict[str, np.ndarray]:
        key = (frame.path, frame.time_index)
        cached = self._upper_base_cache.get(key)
        if cached is not None:
            self._upper_base_cache.move_to_end(key)
            return cached
            
        with Dataset(frame.path) as nc:
            pressure = self._calc_pressure(nc, frame.time_index).astype(float32)
            height = self._calc_height(nc, frame.time_index).astype(float32)
            u_stag = self._slice_time_var(nc.variables['U'], frame.time_index)
            v_stag = self._slice_time_var(nc.variables['V'], frame.time_index)
            u_mass = self._destagger(u_stag, axis=-1).astype(float32)
            v_mass = self._destagger(v_stag, axis=-2).astype(float32)
            temp_k = self._calc_temperature(nc, pressure, frame.time_index)
            temp_c = (temp_k - 273.15).astype(float32)
            qv = self._slice_time_var(nc.variables['QVAPOR'], frame.time_index)
            rh = self._calc_relative_humidity(temp_k, pressure, qv).astype(float32)
            wspd = np.hypot(u_mass, v_mass).astype(float32)
        
        # Some WRF outputs carry different vertical counts across staggered fields
        # (e.g., PH-derived height can be double the mass-level pressure count). Trim
        # everything to the shallowest shared depth so downstream products (such as
        # precipitation type) operate on aligned columns.
        vert_depths = [
            pressure.shape[0],
            height.shape[0],
            temp_c.shape[0],
            rh.shape[0],
            u_mass.shape[0],
            v_mass.shape[0],
            wspd.shape[0],
        ]
        common_levels = min(vert_depths)
        if common_levels < 2:
            raise RuntimeError('Insufficient vertical levels for upper-air calculations')
        
        def _trim_to_levels(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] == common_levels:
                return arr
            return arr[:common_levels, ...].astype(float32, copy=False)
        
        pressure = _trim_to_levels(pressure)
        height = _trim_to_levels(height)
        temp_c = _trim_to_levels(temp_c)
        rh = _trim_to_levels(rh)
        u_mass = _trim_to_levels(u_mass)
        v_mass = _trim_to_levels(v_mass)
        wspd = _trim_to_levels(wspd)
        
        fields = {
            'pressure': pressure,
            'height': height,
            'temperature': temp_c,
            'rh': rh,
            'u': u_mass,
            'v': v_mass,
            'wspd': wspd,
        }
        self._upper_base_cache[key] = fields
        self._upper_base_cache.move_to_end(key)
        while len(self._upper_base_cache) > self._upper_base_cache_limit:
            self._upper_base_cache.popitem(last=False)
        return fields
    
    def _ensure_pressure_orientation(self, frame_path: str, pressure: np.ndarray) -> str:
        orient = self._pressure_orientation.get(frame_path)
        if orient:
            return orient
        sample = pressure[:, pressure.shape[1] // 2, pressure.shape[2]  // 2]
        orient = 'ascending' if sample[0] <= sample[-1] else 'descending'
        self._pressure_orientation[frame_path] = orient
        return orient
    
    @staticmethod
    def _interp_column(target_pa: float, p_col: np.ndarray, f_col: np.ndarray) -> float:
        valid = np.isfinite(p_col) & np.isfinite(f_col)
        if valid.sum() < 2:
            return np.nan
        p_valid = p_col[valid]
        f_valid = f_col[valid]
        if p_valid[0] > p_valid[-1]:
            order = np.argsort(p_valid)
            p_valid = p_valid[order]
            f_valid = f_valid[order]
        if target_pa < p_valid[0]:
            if np.isclose(target_pa, p_valid[0]):
                return float(f_valid[0])
            return np.nan
        if target_pa > p_valid[-1]:
            return np.nan
        return float(np.interp(target_pa, p_valid, f_valid, left=np.nan, right=np.nan))
    
    def _interp_to_pressure(self, field: np.ndarray, pressure: np.ndarray, level_hpa: float, frame_path: str) -> np.ndarray:
        target_pa = level_hpa * 100.0
        orient = self._ensure_pressure_orientation(frame_path, pressure)
        if orient == 'descending':
            pressure = pressure[::-1, :, :]
            field = field[::-1, :, :]
        if field.shape != pressure.shape:
            min_dims = tuple(min(fs, ps) for fs, ps in zip(field.shape, pressure.shape))
            field = field[tuple(slice(0, m) for m in min_dims)]
            pressure = pressure[tuple(slice(0, m) for m in min_dims)]
        
        pressure = np.ascontiguousarray(pressure, dtype=float32)
        field = np.ascontiguousarray(field, dtype=float32)
        nz, ny, nx = pressure.shape
        cols = np.moveaxis(pressure, 0, -1).reshape(-1, nz)
        vals = np.moveaxis(field, 0, -1).reshape(-1, nz)
        
        finite_mask = np.all(np.isfinite(cols), axis=1) & np.all(np.isfinite(vals), axis=1)
        monotonic_mask = np.all(np.diff(cols, axis=1) >= -1e13, axis=1)
        valid_cols = finite_mask & monotonic_mask
        
        out_flat = np.full(cols.shape[0], np.nan, dtype=float32)
        if np.any(valid_cols):
            p_valid = cols[valid_cols]
            f_valid = vals[valid_cols]
            upper_mask = p_valid >= target_pa
            has_upper = upper_mask.any(axis=1)
            idx_upper = np.argmax(upper_mask, axis=1)
            
            valid_results = np.full(p_valid.shape[0], np.nan, dtype=float32)
            
            # Exact matches at the first level.
            exact_first = has_upper & (idx_upper == 0) & np.isclose(p_valid[:, 0], target_pa)
            if np.any(exact_first):
                valid_results[exact_first] = f_valid[exact_first, 0].astype(float32)
            
            usable = has_upper & (idx_upper > 0)
            if np.any(usable):
                sel_rows = np.where(usable)[0]
                upper_idx = idx_upper[usable]
                lower_idx = upper_idx - 1
                p1 = p_valid[sel_rows, lower_idx]
                p2 = p_valid[sel_rows, upper_idx]
                f1 = f_valid[sel_rows, lower_idx]
                f2 = f_valid[sel_rows, upper_idx]
                denom = p2 - p1
                interp_vals = np.empty_like(p1, dtype=float32)
                with np.errstate(divide='ignore', invalid='ignore'):
                    frac = (target_pa - p1) / denom
                    interp_vals = f1 + frac * (f2 - f1)
                zero_denom = np.abs(denom) < 1e-6
                if np.any(zero_denom):
                    interp_vals = interp_vals.astype(float32)
                    interp_vals[zero_denom] = f1[zero_denom].astype(float32)
                valid_results[sel_rows] = interp_vals.astype(float32)
            
            out_flat[valid_cols] = valid_results
        
        invalid_cols = ~valid_cols
        if np.any(invalid_cols):
            idxs = np.where(invalid_cols)[0]
            for i, idx in enumerate(idxs):
                out_flat[idx] = float32(
                    self._interp_column(target_pa, cols[idxs[i]], vals[idxs[i]])
                )
        
        return out_flat.reshape(ny, nx)
    
    def _dbz_to_rate_inhr(self, dbz: np.ndarray) -> np.ndarray:
        '''Approximate precipitation rate (in/hr) from reflectivity (dBZ).
        
        Uses a Marshall-Palmer Z-R relationship (Z = 200 R^1.6), converts to
        mm/hr, then inches/hr. Negative/invalid dBZ values are clipped to keep
        rates non-negative.
        '''
        dbz = np.asarray(dbz, dtype=float32)
        with np.errstate(over='ignore'):
            z_lin = np.power(10.0, dbz * 0.1)
        z_lin = np.clip(z_lin, 0.0, None)
        rain_rate_mmhr = np.power(z_lin / 200.0, 1.0 / 1.6, dtype=float32)
        return rain_rate_mmhr / 25.4
    
    def _precip_type_field(self, frame: WRFFrame) -> np.ndarray:
        key = (frame.path, 'PTYPE', frame.time_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        base_fields = self._get_upper_base_fields(frame)
        pressure = base_fields['pressure']
        temp_c = base_fields['temperature']
        height = base_fields['height']
        
        self._log_debug(
            f'PTYPE base shapes: pressure={pressure.shape}, temp={temp_c.shape}, height={height.shape}'
        )
        
        orient = self._ensure_pressure_orientation(frame.path, pressure)
        surface_first = orient == 'descending'
        self._log_debug(f'PTYPE orientation: {orient} (surface_first={surface_first})')
        if not surface_first:
            pressure = pressure[::-1, :, :]
            temp_c = temp_c[::-1, :, :]
            height = height[::-1, :, :]
        
        # Ensure dimensions agree between height-derived thickness and temperature profiles.
        # Some WRF outputs carry off-by-one counts vertically or horizontally between
        # staggered grids. Align everything to the shallowest shared depth and smallest
        # horizontal extents before computing totals.
        nz = min(temp_c.shape[0], height.shape[0], pressure.shape[0])
        ny = min(temp_c.shape[1], height.shape[1], pressure.shape[1])
        nx = min(temp_c.shape[2], height.shape[2], pressure.shape[2])
        if nz <= 1 or ny <= 1 or nx <= 1:
            raise RuntimeError('Insufficient vertical levels to determine precipitation type')
        
        slicer = (slice(0, nz), slice(0, ny), slice(0, nx))
        temp_c = np.ascontiguousarray(temp_c[slicer], dtype=float32)
        height = np.ascontiguousarray(height[slicer], dtype=float32)
        pressure = np.ascontiguousarray(pressure[slicer], dtype=float32)
        
        self._log_debug(
            f'PTYPE aligned shapes: pressure={pressure.shape}, temp={temp_c.shape}, height={height.shape}, nz={nz}, ny={ny}, nx={nx}'
        )
        
        layer_thickness = np.diff(height, axis=0, append=height[::-1, :, :])
        layer_thickness = np.clip(layer_thickness, 0.0, None)
        
        if layer_thickness.shape != temp_c.shape:
            nz_energy = min(layer_thickness.shape[0], temp_c.shape[0])
            ny_energy = min(layer_thickness.shape[1], temp_c.shape[1])
            nx_energy = min(layer_thickness.shape[2], temp_c.shape[2])
            layer_thickness = layer_thickness[:nz_energy, :ny_energy, :nx_energy]
            temp_c = temp_c[:nz_energy, :ny_energy, :nx_energy]
            pressure = pressure[:nz_energy, :ny_energy, :nx_energy]
            self._log_debug(
                f'PTYPE energy align: layer_thickness shape={layer_thickness.shape}, temp shape={temp_c.shape}, pressure shape={pressure.shape}'
            )
        
        warm_energy = np.sum(np.clip(temp_c, 0.0, None) * layer_thickness, axis=0)
        cold_energy = np.sum(np.clip(-temp_c, 0.0, None) * layer_thickness, axis=0)
        surface_temp = temp_c[0, :, :]
        max_temp = temp_c.max(axis=0)
        
        self._log_debug(
            f'PTYPE energies: warm_energy shape={warm_energy.shape}, cold_energy shape={cold_energy.shape}, surface_temp shape={surface_temp.shape}, max_temp shape={max_temp.shape}'
        )
        
        ptype = np.zeros_like(surface_temp, dtype=np.int8) # rain default
        
        snow_mask = (max_temp <= 0.5) | (warm_energy < 5.0)
        
        warm_ratio = warm_energy / np.clip(warm_energy + cold_energy, 1e-3, None)
        warm_near_surface = surface_temp >= 1.0
        strong_surface_warm = surface_temp >= 2.0
        deep_warm = max_temp >= 1.5
        
        sleet_mask = (
            ~snow_mask
            & (warm_energy > 8.0)
            & (cold_energy >= warm_energy * 0.55)
            & (surface_temp <= 0.5)
        )
        
        rain_mask = (
            ~snow_mask
            & ~sleet_mask
            & (
                strong_surface_warm
                | (warm_near_surface & (warm_ratio >= 0.5))
                | (deep_warm & (warm_ratio >= 0.45))
                | (warm_energy >= cold_energy * 0.5)
            )
        )
        
        mix_mask = ~(snow_mask | sleet_mask | rain_mask)
        
        warm_lean_mix = mix_mask & (warm_ratio >= 0.45) & (surface_temp >= 0.5)
        mix_mask = mix_mask & ~warm_lean_mix
        
        ptype[snow_mask] = 1
        ptype[mix_mask] = 2
        ptype[sleet_mask] = 3
        
        # Bias toward rain for low-elevation warm surfaces unless strong ice signals exist
        surface_elev = height[0, :, :]
        lowland_rain = (surface_elev <= 1200.0) & (surface_temp >= 0.5)
        ptype[lowland_rain & ~snow_mask & ~sleet_mask]
        
        # Only classify where precipitation is present (MDBZ > 0); Leave others transparent
        try:
            mdbz = self.get2d(frame, 'MDBZ')
            if mdbz.shape != ptype.shape:
                ny = min(mdbz.shape[0], ptype.shape[0])
                nx = min(mdbz.shape[1], ptype.shape[1])
                mdbz = np.ascontiguousarray(mdbz[:ny, :nx], dtype=float32)
                ptype = ptype[:ny, :nx]
            precip_mask = np.isfinite(mdbz) & (mdbz > 0.0)
            rate_inhr = self._dbz_to_rate_inhr(mdbz)
            rate_inhr = np.clip(rate_inhr, 0.0, PTYPE_MAX_RATE_INHR)
            intensity = (rate_inhr / PTYPE_MAX_RATE_INHR).astype(float32) * PTYPE_INTENSITY_SPAN
            ptype = np.where(precip_mask, ptype.astype(float32) + intensity, np.nan)
        except Exception as exc:
            self._log_debug(f'PTYPE: Unable to apply precipitation mask ({exc})')
        
        self._cache[key] = ptype.astype(float32)
        return self._cache[key]
    
    def is_upper_air(self, var: str) -> bool:
        return var.upper() in UPPER_AIR_SPECS
    
    def get_upper_air_data(self, frame: WRFFrame, var: str) -> UpperAirData:
        v = var.upper()
        if v not in UPPER_AIR_SPECS:
            raise RuntimeError(f'Unknown upper air variable "{var}"')
        key = (frame.path, f'UPPER:{v}', frame.time_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        spec = UPPER_AIR_SPECS[v]
        base_fields = self._get_upper_base_fields(frame)
        pressure = base_fields['pressure']
        
        scalar3d = base_fields.get(spec.shading_field)
        if scalar3d is None:
            raise RuntimeError(f'Missing shading field "{spec.shading_field}" for {var}.')
        scalar2d = self._interp_to_pressure(scalar3d, pressure, spec.level_hpa, frame.path)
        
        contour2d = None
        if spec.contour_field:
            contour3d = base_fields.get(spec.contour_field)
            if contour3d is None:
                raise RuntimeError(f'Missing contour field "{spec.contour_field}" for {var}.')
            contour2d = self._interp_to_pressure(contour3d, pressure, spec.level_hpa, frame.path)
        
        u2d = self._interp_to_pressure(base_fields['u'], pressure, spec.level_hpa, frame.path)
        v2d = self._interp_to_pressure(base_fields['v'], pressure, spec.level_hpa, frame.path)
        
        data = UpperAirData(
            scalar=scalar2d.astype(float32),
            contour=None if contour2d is None else contour2d.astype(float32),
            u=u2d.astype(float32),
            v=v2d.astype(float32),
        )
        self._cache[key] = data
        return data
    
    # --- Preload API (helpers) ---
    def allocate_preload(self, var: str, level_hpa: T.Optional[float]) -> None:
        key = (var.upper(), float(level_hpa) if level_hpa is not None else None)
        if key not in self.preloaded:
            self.preloaded[key] = [None] * len(self.frames)
            
    def set_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int, data: T.Any) -> None:
        key = (var.upper(), float(level_hpa) if level_hpa is not None else None)
        self.preloaded.setdefault(key, [None] * len(self.frames))
        self.preloaded[key][idx] = data
    
    def get_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int) -> T.Optional[T.Any]:
        key = (var.upper(), float(level_hpa) if level_hpa is not None else None)
        li = self.preloaded.get(key)
        return None if li is None else li[idx]
        
    def clear_preloaded(self):
        self.preloaded.clear()
    
    # --- Data Access ---
    def get2d(self, frame: WRFFrame, var: str, level_hpa: T.Optional[float] = None) -> np.ndarray:
        key = (frame.path, f'{var}@{level_hpa}', frame.time_index)
        if key in self._cache:
            return self._cache[key]
        
        with Dataset(frame.path) as nc:
            v = var.upper()
            if v in ('MDBZ', 'MAXDBZ'):
                # Compute MDBZ as column max of reflectivity (no wrf.getvar to avoid pickling)
                if 'REFL_10CM' in nc.variables:
                    refl = nc.variables['REFL_10CM'] # (Time, bottom_top, y, x)
                elif 'dbz' in nc.variables:
                    refl = nc.variables['dbz']
                else:
                    raise RuntimeError('Need "dbz" or "REFL_10CM" in wrfout to compute MDBZ.')
                
                dims = tuple(getattr(refl, 'dimensions', ()))
                if 'Time' in dims:
                    t_axis = dims.index('Time')
                    slicer = [slice(None)] * refl.ndim
                    slicer[t_axis] = frame.time_index
                    arr = np.array(refl[tuple(slicer)]) # -> (bottom_top, y, x)
                else:
                    arr = np.array(refl[:])
                arr = np.squeeze(arr)
                if arr.ndim == 3:
                    data2d = np.nanmax(arr, axis=0)
                elif arr.ndim == 2:
                    data2d = arr
                else:
                    raise RuntimeError(f'Reflectivity shape {arr.shape} not understood for MDBZ')
            elif v == 'RAINNC':
                data2d = np.array(nc.variables['RAINNC'][frame.time_index, :, :])
            elif v == 'RAINC':
                data2d = np.array(nc.variables['RAINC'][frame.time_index, :, :])
            elif v == 'WSPD10':
                u10 = np.array(nc.variables['U10'][frame.time_index, :, :])
                v10 = np.array(nc.variables['V10'][frame.time_index, :, :])
                data2d = np.hypot(u10, v10)
            elif v == 'T2F':
                if 'T2' not in nc.variables:
                    raise RuntimeError('Variable "T2" not found; cannot compute 2 m temperature')
                t2_var = nc.variables['T2']
                dims = tuple(getattr(t2_var, 'dimensions', ()))
                if 'Time' in dims:
                    t2k = np.array(t2_var[frame.time_index, :, :])
                else:
                    t2k = np.array(t2_var[:, :])
                t2c = t2k - 273.15
                data2d = (t2c * 9.0 / 5.0) + 32.0
            elif v == 'PTYPE':
                data2d = self._precip_type_field(frame)
            elif v == 'REFL1KM':
                if 'REFL_10CM' in nc.variables:
                    refl_var = nc.variables['REFL_10CM']
                elif 'dbz' in nc.variables:
                    refl_var = nc.variables['dbz']
                else:
                    raise RuntimeError('Need "REFL_10CM" or "dbz" to compute REFL1KM')
                
                r_dims = tuple(getattr(refl_var, 'dimensions', ()))
                if 'Time' in r_dims:
                    t_axis = r_dims.index('Time')
                    slicer = [slice(None)] * refl_var.ndim
                    slicer[t_axis] = frame.time_index
                    refl = np.array(refl_var[tuple(slicer)]) # -> (z, y, x)
                else:
                    refl = np.array(refl_var[:])
                relf = np.squeeze(refl)
                if refl.ndim != 3:
                    raise RuntimeError(f'Reflectivity has shape {relf.shape}, expected 3D (z,y,x)')
                nz, ny, nx = refl.shape
                
                if 'PH' not in nc.variables or 'PHB' not in nc.variables:
                    raise RuntimeError('PH/PHB not found; cannot compute heights for REFL1KM')
                PH = nc.variables['PH'] 
                PHB = nc.variables['PHB']
                w_dims = tuple(getattr(PH, 'dimensions', ()))
                if 'Time' in w_dims:
                    slicer = [slice(None)] * PH.ndim
                    slicer[w_dims.index('Time')] = frame.time_index
                    ph = np.array(PH[tuple(slicer)])
                    phb = np.array(PHB[tuple(slicer)])
                else:
                    ph = np.array(PH[:])
                    phb = np.array(PHB[:])
                z_w = (ph + phb) / 9.81
                if z_w.shape[0] != nz + 1:
                    raise RuntimeError('Vertical dimension mismatch between REFL_10CM and PH/PHB')
                z_mass = 0.5 * (z_w[:-1, :, :] + z_w[1:, :, :])
                
                if 'HGT' not in nc.variables:
                    raise RuntimeError('HGT not found; cannot compute AGL.')
                
                HGTv = nc.variables['HGT']
                h_dims = tuple(getattr(HGTv, 'dimensions', ()))
                if 'Time' in h_dims:
                    hgt = np.array(HGTv[frame.time_index, :, :])
                else:
                    hgt = np.array(HGTv[:, :])
                
                z_agl = z_mass - hgt[None, :, :]
                
                target = 1000.0
                mask_up = z_agl >= target
                has = mask_up.any(axis = 0)
                k = mask_up.argmax(axis=0)
                
                k1 = np.clip(k, 1, nz - 1)
                k0 = k1 - 1
                
                z0 = z_agl[k0, np.arange(ny)[:,None], np.arange(nx)]
                z1 = z_agl[k1, np.arange(ny)[:,None], np.arange(nx)]
                r0 = refl[k0, np.arange(ny)[:,None], np.arange(nx)]
                r1 = relf[k1, np.arange(ny)[:,None], np.arange(nx)]
                
                denom = (z1 - z0)
                denom[denom == 0] = 1.0
                w = (target - z0) / denom
                w = np.clip(w, 0.0, 1.0)
                
                data2d = r0 + w * (r1 - r0)
                
                topcol = refl[-1, :, :]
                data2d = np.where(has, data2d, topcol)
            else:
                if v not in nc.variables:
                    raise RuntimeError(f'Variable "{var}" not found')
                var_obj = nc.variables[v]
                dims = tuple(getattr(var_obj, 'dimensions', ()))
                base = np.array(var_obj[frame.time_index]) if 'Time' in dims else np.array(var_obj[:])
                base = np.squeeze(base)
                if base.ndim == 3:
                    data2d = base[0, :, :]
                elif base.ndim == 2:
                    data2d = base
                else:
                    raise RuntimeError(f'Unsupported shape {base.shape} for "{var}".')
        
        self._cache[key] = data2d
        return data2d

# ------------------------------
# Worker (thread) for preloading
# ------------------------------
class PreloadWorker(QtCore.QObject):
    progress = QtCore.Signal(int) # percent
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)
    
    def __init__(self, loader: 'WRFLoader', var: str, level_hpa: T.Optional[float]):
        super().__init__()
        self.loader = loader
        self.var = var
        self.level_hpa = level_hpa
    
    @QtCore.Slot()
    def run(self):
        try:
            n = len(self.loader.frames)
            self.loader.allocate_preload(self.var, self.level_hpa)
            for i, fr in enumerate(self.loader.frames):
                if self.loader.is_upper_air(self.var):
                    data = self.loader.get_upper_air_data(fr, self.var)
                else:
                    data = self.loader.get2d(fr, self.var, self.level_hpa)
                self.loader.set_preloaded(self.var, self.level_hpa, i, data)
                self.progress.emit(int((i + 1) * 100 / max(1, n)))
            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))

# ------------------------------
# GUI
# ------------------------------
class WRFViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WRF Viewer - PySide6')
        #self.resize(1200, 800)
        
        self.loader = WRFLoader()
        self.upper_air_specs = UPPER_AIR_SPECS
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100) # ms
        self.timer.timeout.connect(self._tick)
        self._stepping = False
        
        # --- App Settings ---
        self.settings = QtCore.QSettings('WRFViewer1', 'WRFViewer')
        self.user_cmap_dir = Path.home() / '.wrfviewer' / 'colormaps'
        self.project_cmap_dir = Path.cwd() / 'colormaps'
        for d in (self.user_cmap_dir, self.project_cmap_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.cmap_registry: dict[str, LinearSegmentedColormap] = {}
        
        # --- Central ---
        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # --- Variable categories (accordion data) ---
        self.var_categories: dict[str, list[tuple[str, str]]] = {
            'Surface': [
                ('Composite Reflectivity', 'MDBZ'),
                ('Total Rain Accumulation', 'RAINNC'),
                ('RAINC', 'RAINC'),
                ('10 m AGL Wind', 'WSPD10'),
                ('2 m Temperature (°F)', 'T2F'),
                ('Precipitation Type', 'PTYPE'),
            ],
            'Severe': [
                ('MDBZ', 'MDBZ'),
                ('REFL1KM', 'REFL1KM'),
            ],
            'Upper Air': [
                *[(spec.display_name, spec.canonical) for spec in UPPER_AIR_SPECS.values()],
                ('REFL1KM', 'REFL1KM'),
            ],
        }
        self._var_aliases: dict[str, str] = {
            label.upper(): canonical.upper()
            for items in self.var_categories.values()
            for label, canonical in items
        }

        # ensure canonical names include themselves
        for canonical in {c for items in self.var_categories.values() for _, c in items}:
            self._var_aliases.setdefault(canonical.upper(), canonical.upper())

        
        # --- Status Bar Progress (inline) ---
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setVisible(False)
        self.status.addPermanentWidget(self.pb)
        
        # --- Controls Row ---
        controls = QHBoxLayout()
        self.btn_open = QPushButton('Open wrfout...')
        self.btn_open.clicked.connect(self.on_open)
        controls.addWidget(self.btn_open)
        
        seen_labels: set[str] = set()
        self._variable_labels: list[str] = []
        for items in self.var_categories.values():
            for label, _ in items:
                if label not in seen_labels:
                    seen_labels.add(label)
                    self._variable_labels.append(label)

        self.current_var_label: str = self._variable_labels[0] if self._variable_labels else ''
        
        # --- Colormap Picker ---
        controls.addWidget(QLabel('Colormap:'))
        self.cmb_cmap = QComboBox()
        #self.cmb_cmap.addItems(['viridis', 'plasma', 'cividis', 'turbo' if hasattr(plt.cm, 'turbo') else 'plasma'])
        self.cmb_cmap.currentTextChanged.connect(self.on_cmap_changed)
        controls.addWidget(self.cmb_cmap)
        
        self.btn_open_cmap_dir = QPushButton('Open cmap folder...')
        #self.btn_open_cmap_dir.setToolTip('Opens ~/.wrfviewer/colormaps (and ./colormaps). Drop .cpt/.ct files there to auto-load next run.')
        self.btn_open_cmap_dir.clicked.connect(self.on_open_cmap_folder)
        controls.addWidget(self.btn_open_cmap_dir)
        
        self.btn_load_cpt = QPushButton('Load CPT...')
        self.btn_load_cpt.clicked.connect(self.on_load_cpt)
        controls.addWidget(self.btn_load_cpt)
        
        self.btn_preload = QPushButton('Preload current variable')
        self.btn_preload.clicked.connect(self.on_preload)
        controls.addWidget(self.btn_preload)
        
        self.btn_export = QPushButton('Export PNG...')
        self.btn_export.clicked.connect(self.on_export_png)
        controls.addWidget(self.btn_export)
        
        controls.addStretch(1)
        
        self.btn_play = QPushButton('▶ Play')
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.on_toggle_play)
        controls.addWidget(self.btn_play)
        
        vbox.addLayout(controls)
        
        # --- Time Row ---
        time_row = QHBoxLayout()
        self.lbl_time = QLabel('Time: -')
        time_row.addWidget(self.lbl_time)
        
        self.sld_time = QSlider(Qt.Horizontal)
        self.sld_time.setRange(0,0)
        # Redraw while dragging (debounced)
        self.sld_time.setTracking(True)
        self.sld_time.valueChanged.connect(self._on_slider_changed)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(25)
        self._debounce.timeout.connect(self._do_redraw_from_slider)
        time_row.addWidget(self.sld_time, stretch=1)
        vbox.addLayout(time_row)
        
        # --- Figure & Accordion Layout ---
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = plt.axes(projection=ccrs.PlateCarree())
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)
        self._timestamp_text = self.fig.text(
            0.02,
            0.02,
            '',
            transform=self.fig.transFigure,
            ha='left',
            va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.25')
        )
        
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)

        self.var_toolbox = QToolBox()
        self.var_toolbox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._category_lists: list[QListWidget] = []
        self._last_category_index: T.Optional[int] = None
        for category, items in self.var_categories.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(4)
            list_widget = QListWidget()
            list_widget.setSelectionMode(QListWidget.SingleSelection)
            for label, canonical in items:
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, label)
                item.setData(Qt.UserRole + 1, canonical)
                list_widget.addItem(item)
            list_widget.itemClicked.connect(self._on_category_var_selected)
            page_layout.addWidget(list_widget)
            page_layout.addStretch(1)
            self.var_toolbox.addItem(page, category)
            self._category_lists.append(list_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(QLabel('Variable Categories'))
        left_layout.addWidget(self.var_toolbox, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        left_panel.setMinimumWidth(200)

        vbox.addWidget(self.main_splitter, stretch=1)
        
        # -- Load colormaps (builtins + user folders) and pick last used.
        self._init_colormaps()
        saved = self.settings.value('cmap_name', 'Viridis')
        self.current_cmap = self.cmap_registry.get(saved, plt.get_cmap('viridis'))
        self.cmb_cmap.setCurrentText(saved if saved in self.cmap_registry else 'Viridis')
        
        #self.current_cmap = plt.get_cmap('viridis')
        self._img_art = None
        self._cbar = None
        self._img_shape = None
        self._contour_sets: list = []
        self._contour_labels: list = []
        self._barb_art = None
        self._value_labels: list[matplotlib.text.Text] = []
        
        # Basemap Features
        self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        self.ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4)
        self.ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6)

        # Sync default selection with accordion
        if self.current_var_label:
            self._sync_category_selection(self.current_var_label)
    
    # ---------------
    # Event handlers
    # ---------------
    def on_open(self):
        dlg = QFileDialog(self, 'Select wrfout files', os.getcwd(), 'WRF NetCDF (wrfout_*);;All files (*)')
        dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if not dlg.exec():
            return
        paths = dlg.selectedFiles()
        try:
            self.loader.open_files(paths)
        except Exception as e:
            QMessageBox.critical(self, 'Open failed', str(e))
            return
        n = len(self.loader.frames)
        self.sld_time.setRange(0, max(0, n -1))
        self.sld_time.setValue(0)
        self._extent_set = False
        self.loader.clear_preloaded()
        self.update_plot()
    
    def on_var_changed(self, text: str):
        if not text:
            return
        self.current_var_label = text
        self.loader.clear_preloaded()
        self._sync_category_selection(text)
        self.update_plot()

    def _on_category_var_selected(self, item: QListWidgetItem):
        if item is None:
            return
        label = item.data(Qt.UserRole) or item.text()
        if not label:
            return
        list_widget = item.listWidget()
        if list_widget in self._category_lists:
            self._last_category_index = self._category_lists.index(list_widget)
        else:
            self._last_category_index = None
        if self.current_var_label == label:
            # Ensure downstream actions happen even if the same label is re-selected
            self.on_var_changed(label)
            return
        self.on_var_changed(label)

    def _sync_category_selection(self, label: str) -> None:
        if not label:
            return
        matches: dict[int, QListWidgetItem] = {}
        for idx, lst in enumerate(self._category_lists):
            items = lst.findItems(label, Qt.MatchExactly)
            if items:
                matches[idx] = items[0]

        if not matches:
            for lst in self._category_lists:
                lst.blockSignals(True)
                lst.clearSelection()
                lst.blockSignals(False)
            self._last_category_index = None
            return

        preferred = self._last_category_index
        target_index: T.Optional[int] = preferred if preferred in matches else next(iter(matches))

        for idx, lst in enumerate(self._category_lists):
            lst.blockSignals(True)
            if idx == target_index:
                item = matches[idx]
                lst.setCurrentItem(item, QtCore.QItemSelectionModel.ClearAndSelect)
                lst.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
            else:
                lst.clearSelection()
            lst.blockSignals(False)

        if target_index is not None:
            self.var_toolbox.setCurrentIndex(target_index)
        self._last_category_index = None

    def _canonical_var(self, var: str) -> str:
        if not var:
            return ''
        return self._var_aliases.get(var.upper(), var.upper())
    
    def on_toggle_play(self, checked: bool):
        if checked:
            self.btn_play.setText('⏸ Pause')
            self.timer.start()
        else:
            self.btn_play.setText('▶ Play')
            self.timer.stop()
        
    def _tick(self):
        if self.sld_time.maximum() <= 0:
            return
        self._stepping = True
        try:
            val = self.sld_time.value() + 1
            if val > self.sld_time.maximum():
                val = 0
            self.sld_time.setValue(val)
            self.update_plot()
        finally:
            self._stepping = False
    
    def _on_slider_changed(self, val: int):
        if getattr(self, '_stepping', False):
            self.update_plot()
            return
        
        self._pending_slider_value = val
        self._debounce.start()
    
    def _do_redraw_from_slider(self):
        self.update_plot()
    
    def on_open_cmap_folder(self):
        #folder = os.path.expanduser('~/.wrfviewer/colormaps')
        folder = str(self.user_cmap_dir)
        if QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
            return
        
        for exe, args in [('xdg-open', [folder]), ('gio', ['open', folder]), ('nautilus', [folder])]:
            if shutil.which(exe):
                QtCore.QProcess.startDetached(exe, args)
                
        QtWidgets.QFileDialog.getExisitingDirectory(self, 'Colormap folder', folder, QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontUseNativeDialog)
        #self._open_folder_portable(folder)
    
    def on_load_cpt(self):
        start_dir = str(self.user_cmap_dir if self.user_cmap_dir.exists() else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, 'Load CPT colormap', os.getcwd(), 'CPT (*.ct *.cpt);;All files (*)')
        if not path:
            return
        try:
            cmap = self._load_cpt(path)
            name = Path(path).stem
            self.cmap_registry[name] = cmap
            self._populate_cmap_combo(select=name)
            self.current_cmap = cmap
            self.settings.setValue('cmap_name', name)
            if self._img_art is not None:
                self._img_art.set_cmap(self.current_cmap)
                if self._cbar is not None:
                    self._cbar.update_normal(self._img_art)
                self.canvas.draw_idle()
            self.update_plot()
            #name = os.path.splittext(os.path.basename(path))[0]
            #if self.cmb_cmap.findText(name) == -1:
            #    self.cmb_cmap.addItems(name)
            #self.cmb_cmap.setCurrentText(name)
            #self.canvas.draw_idle()
            #self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, 'Colormap error', f'Failed to load CPT: {e}')
    
    def on_cmap_changed(self, name: str):
        if not name:
            return
        cmap = self.cmap_registry.get(name)
        if cmap is None:
            return
        self.current_cmap = cmap
        self.settings.setValue('cmap_name', name)
        if self._img_art is not None:
                self._img_art.set_cmap(self.current_cmap)
                if self._cbar is not None:
                    self._cbar.update_normal(self._img_art)
                self.canvas.draw_idle()
        #self.update_plot()
    
    def on_preload(self):
        if not self.loader.frames:
            return
        display_var = self.current_var_label
        var = self._canonical_var(display_var)

        # show inline progress bar
        self.pb.setVisible(True)
        self.pb.setValue(0)
        self.status.showMessage(f'Preloading {display_var}')

        self.worker_thread = QtCore.QThread(self)
        level = UPPER_AIR_SPECS[var].level_hpa if var in UPPER_AIR_SPECS else None
        self.worker = PreloadWorker(self.loader, var, level)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.pb.setValue)
        self.worker.failed.connect(self._preload_failed)
        self.worker.finished.connect(self._preload_done)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()
        
    def _preload_failed(self, msg: str):
        self.status.clearMessage()
        self.pb.setVisible(False)
        QtWidgets.QMessageBox.critical(self, 'Preload error', msg)
        
    def _preload_done(self):
        self.pb.setValue(100)
        self.status.showMessage('Preload complete', 1500)
        QtCore.QTimer.singleShot(500, lambda: self.pb.setVisible(False))
        self.update_plot()
        
    def on_export_png(self):
        if not self.loader.frames:
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Export PNG', os.getcwd(), 'PNG (*.png)')
        if not out:
            return
        self.fig.savefig(out, dpi=150, bbox_inches='tight')
        
    # ---------------
    # Plotting
    # ---------------
    def update_plot(self):
        if not self.loader.frames:
            return
        idx = self.sld_time.value()
        frame = self.loader.frames[idx]
        self.lbl_time.setText(f'Time: {frame.timestamp_str}')
        self._timestamp_text.set_text(f'WRF Gabe Zago   {frame.timestamp_str}')
        
        display_var = self.current_var_label
        var = self._canonical_var(display_var)
        spec = self.upper_air_specs.get(var)
        level = spec.level_hpa if spec else None

        data_obj = self.loader.get_preloaded(var, level, idx)
        if data_obj is None:
            try:
                if spec:
                    data_obj = self.loader.get_upper_air_data(frame, var)
                else:
                    data_obj = self.loader.get2d(frame, var, None)
            except Exception as e:
                QMessageBox.critical(self, 'Plot error', str(e))
                return
        
        lat, lon = self.loader.get_geo(frame)
        
        if not hasattr(self, '_extent_set') or not self._extent_set:
            ymin, ymax = float(np.nanmin(lat)), float(np.nanmax(lat))
            xmin, xmax = float(np.nanmin(lon)), float(np.nanmax(lon))
            dy = (ymax - ymin) * 0.05
            dx = (xmax - xmin) * 0.05
            self.ax.set_extent([xmin - dx, xmax + dx, ymin - dy, ymax + dy], crs=ccrs.PlateCarree())
            self._extent_set = True
        
        if spec:
            if not isinstance(data_obj, UpperAirData):
                data_obj = self.loader.get_upper_air_data(frame, var)
            data = np.asarray(data_obj.scalar)
        else:
            data = np.asarray(data_obj)
        
        data = np.squeeze(data)
        plot_lat = lat
        plot_lon = lon
        if data.shape != lat.shape:
            ny = min(data.shape[0], lat.shape[0]) if data.ndim >= 1 else lat.shape[0]
            nx = min(data.shape[1], lat.shape[1]) if data.ndim >= 2 else lat.shape[1]
            plot_lat = lat[:ny, :nx]
            plot_lon = lon[:ny, :nx]
            if data.ndim >= 2:
                data = data[:ny, :nx]
            elif data.ndim == 1:
                data = data[:ny]
            else:
                data = np.resize(data, (ny, nx))
        else:
            ny, nx = plot_lat.shape
        
        if spec:
            vmin, vmax = spec.vmin, spec.vmax
            label = spec.colorbar_label
        else:
            vmin, vmax, label = self._default_range(var)
        
        cmap_to_use = self.current_cmap
        norm = None
        tick_values: T.Optional[list[int]] = None
        tick_labels: T.Optional[list[str]] = None
        if var.upper() == 'PTYPE':
            cmap_to_use, norm, tick_values, tick_labels = self._precip_type_style()
            self.loader._log_debug(
                f'PTYPE plot alignment: data shape={data.shape}, lat shape={plot_lat.shape}, lon shape={plot_lon.shape}, ticks={tick_values}'
            )
        
        # Create once /then resure for speed
        if self._img_art is None or self._img_shape != data.shape:
            if self._img_art is not None:
                try:
                    self._img_art.remove()
                except Exception:
                    pass
            
            self._img_art = self.ax.pcolormesh(
                plot_lon, plot_lat, data,
                transform=ccrs.PlateCarree(),
                cmap=cmap_to_use,
                norm=norm,
                shading='nearest',
                vmin=vmin, vmax=vmax,
                antialiased=False,
                rasterized=True,
            )
            self._img_shape = data.shape
            
            if self._cbar is None:
                self._cbar = self.fig.colorbar(self._img_art, ax=self.ax, orientation='vertical', shrink=0.8, pad=0.02)
            self._cbar.set_label(label)
            if tick_values is not None and tick_labels is not None:
                self._cbar.set_ticks(tick_values)
                self._cbar.set_ticklabels(tick_labels)
                if var.upper() == 'PTYPE':
                    self._nudge_precip_type_ticks()
            else:
                self._reset_colorbar_ticks()
        else:
            # Update only face colors and limits
            # pcolormesh set_array expects one value per face; averaging corners is okay for speed.
            self._img_art.set_array(np.asarray(data).ravel())
            self._img_art.set_cmap(cmap_to_use)
            if norm is not None:
                self._img_art.set_norm(norm)
            else:
                self._img_art.set_norm(Normalize(vmin=vmin, vmax=vmax))
            if vmin is not None and vmax is not None:
                self._img_art.set_clim(vmin, vmax)
            self._cbar.update_normal(self._img_art)
            self._cbar.set_label(label)
            if tick_values is not None and tick_labels is not None:
                self._cbar.set_ticks(tick_values)
                self._cbar.set_ticklabels(tick_labels)
                if var.upper() == 'PTYPE':
                    self._nudge_precip_type_ticks()
            else:
                self._reset_colorbar_ticks()
        
        if spec:
            self._draw_upper_overlays(plot_lat, plot_lon, data_obj, spec)
        else:
            self._clear_upper_air_artists()
            
        self.ax.set_title(self._title_text(display_var, var), loc='center', fontsize=12, fontweight='bold')
        self._draw_value_labels(plot_lat, plot_lon, data, var)
        self.canvas.draw_idle()

    def _default_range(self, var: str) -> tuple[T.Optional[float], T.Optional[float], str]:
        v = var.upper()
        if v in ('MDBZ', 'MAXDBZ'):
            return 0.0, 70.0, 'Reflectivity (dBZ)'
        if v in ('RAINNC', 'RAINC'):
            return 0.0, None, 'Accumulated Precip (mm)'
        if v in ('WSPD10', 'WIND10'):
            return 0.0, 40.0, '10-m wind speed (m s$^{-1}$)'
        if v == 'T2F':
            return -60.0, 120.0, '2-m temperature (°F)'
        if v == 'REFL1KM':
            return 0.0, 70.0, 'Reflectivity @ 1km AGL (dBZ)'
        if v == 'PTYPE':
            # Keep the precipitation-type colorbar aligned to fixed 0-4 boundaries so
            # each category begins at an integer tick (0=rain, 1=snow, 2=mix, 3=sleet)
            # regardless of the intensity span we use inside each bucket.
            return -0.5, 4.0, 'Precipitation Type (in/hr rates)'
        return None, None, var
    
    def _title_text(self, display_var: str, canonical_var: str) -> str:
        v = canonical_var.upper()
        if v in self.upper_air_specs:
            spec = self.upper_air_specs[v]
            return spec.title or display_var
        if v in ('MDBZ', 'MAXDBZ'):
            return 'MDBZ (Column Max dBZ)'
        if v == 'REFL1KM':
            return 'Reflectivity @ 1 km AGL (dBZ)'
        if v == 'PTYPE':
            return 'Precipitation Type'
        if v == 'T2F':
            return '2 m Temperature (°F)'
        return display_var
    
    def _precip_type_style(self) -> tuple[ListedColormap, BoundaryNorm, list[int], list[str]]:
        if not hasattr(self, '_ptype_cmap'):
            steps_per_type = 96
            type_gradients = [
                ['#8eff8c', '#138527', '#fff200'], # Rain
                ['#e5f0ff', '#74add1', '#6a51a3'], # Snow
                ['#fde0ef', '#f768a1', '#ae017e'], # Mix / Freezing
                ['#f8e7ff', '#c994c7', '#6a3d9a'], # Sleet
            ]
            
            def _ramp_list(colors: list[str], n: int) -> list[tuple[float, float, float]]:
                cmap = mcolors.LinearSegmentedColormap.from_list('ptype_tmp', colors, N=n)
                return [tuple(c[:3]) for c in cmap(np.linspace(0.0, 1.0, n))]
            
            color_list: list[tuplep[float, float, float]] = []
            for colors in type_gradients:
                color_list.extend(_ramp_list(colors, steps_per_type))
                
            self._ptype_cmap = ListedColormap(color_list, name='PrecipTypeIntensity')
            
            # Use a boundary norm so category starts always line up with integer values
            # (0=rain, 1=snow, 2=mix, 3=sleet, 4=top). This prevents rain intensities
            # from bleeding into the snow colors simply because the intensity span is
            # narrower than the 1.0 spacing between category bases.
            ncolors = len(color_list)
            boundaries = np.linspace(0.0, 4.0, ncolors + 1)
            self._ptype_norm = mcolors.BoundaryNorm(boundaries, ncolors=ncolors, clip=True)
        
        # Keep the legend readable by labeling each precipitation type once and
        # showing a handful of rate ticks beneath it. This avoids the cluttered
        # "Type value" repetition that made the previous colorbar hard to read.
        rate_ticks = [0.0, 0.01, 0.05, 0.25, 0.5]
        
        def _offset(rate: float) -> float:
            return float(_ptype_rate_offset(rate))
        
        tick_values: list[float] = []
        tick_labels: list[str] = []
        for base, name in enumerate(['Rain', 'Snow', 'Mix', 'Sleet']):
            for rate in rate_ticks:
                tick_values.append(base + _offset(rate))
                if rate == 0.0:
                    tick_labels.append(f'{name} (in/hr)')
                else:
                    tick_labels.append(f'{rate:.2g}')
                
        return self._ptype_cmap, self._ptype_norm, tick_values, tick_labels
    
    def _nudge_precip_type_ticks(self) -> None:
        '''Shift the base (0.0) rate labels so they don't sit on neighboring ticks.

        Users reported the 0.0 labels blending into adjacent 0.5 tick labels on the
        precipitation-type colorbar. Move any 0.0-like tick text slightly to the
        right and keep precipitation-type ticks on the right side of the colorbar so
        they don't collide with the colorbar label or default range text.
        '''
        
        if not self._cbar:
            return
        
        ax = self._cbar.ax
        # Keep ticks on the right but avoid shoving every label away from the bar;
        # only the long category labels should shift. Give numeric ticks a tiny
        # nudge so they don't sit on top of the colorbar.
        ax.tick_params(axis='y', which='both', labelright=True, labelleft=False, pad=3)
        ax.yaxis.set_label_position('right')
        ax.yaxis.labelpad = 12
        
        # Matplotlib returns (transform, valign, halign); we only need the transform
        base_transform = ax.get_yaxis_text2_transform(0)[0]
        numeric_offset = mtransforms.offset_copy(base_transform, fig=ax.figure, x=6, units='points')
        category_offset = mtransforms.offset_copy(base_transform, fig=ax.figure, x=26, units='points')
        moved = False
        
        for lbl in ax.get_yticklabels():
            txt = lbl.get_text().strip()
            # Reset every label to the default right-hand transform so numeric ticks
            # stay pinned to the colorbar even if we nudge category labels.
            lbl.set_transform(base_transform)
            if 'in/hr' in txt:
                lbl.set_transform(category_offset)
                lbl.set_horizontalalignment('left')
                moved = True
            else:
                lbl.set_transform(numeric_offset)
                lbl.set_horizontalalignment('left')
            
        if moved:
            ax.figure.canvas.draw_idle()

    def _clear_value_labels(self) -> None:
        if not self._value_labels:
            return
        for label in self._value_labels:
            try:
                label.remove()
            except Exception:
                pass
        self._value_labels.clear()

    def _draw_value_labels(self, lat: np.ndarray, lon: np.ndarray, data: np.ndarray, var: str) -> None:
        self._clear_value_labels()
        if var.upper() != 'T2F':
            return

        arr = np.asarray(data)
        if arr.ndim < 2:
            return
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            return

        lat_arr = np.asarray(lat)
        lon_arr = np.asarray(lon)
        if lat_arr.shape != arr.shape or lon_arr.shape != arr.shape:
            ny = min(arr.shape[0], lat_arr.shape[0], lon_arr.shape[0])
            nx = min(arr.shape[1], lat_arr.shape[1], lon_arr.shape[1])
            arr = arr[:ny, :nx]
            lat_arr = lat_arr[:ny, :nx]
            lon_arr = lon_arr[:ny, :nx]

        stride = 5
        start_y = stride if arr.shape[0] > 1 else 0
        start_x = stride if arr.shape[1] > 1 else 0
        for iy in range(start_y, arr.shape[0], stride):
            for ix in range(start_x, arr.shape[1], stride):
                val = arr[iy, ix]
                if not np.isfinite(val):
                    continue
                txt = self.ax.text(
                    lon_arr[iy, ix],
                    lat_arr[iy, ix],
                    f'{val:.0f}',
                    transform=ccrs.PlateCarree(),
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='black',
                )
                self._value_labels.append(txt)
    
    def _reset_colorbar_ticks(self) -> None:
        if not self._cbar:
            return
        locator = mticker.AutoLocator()
        formatter = mticker.ScalarFormatter()
        formatter.set_powerlimits((-6, 6))
        self._cbar.ax.yaxis.set_major_locator(locator)
        self._cbar.ax.yaxis.set_major_formatter(formatter)
        self._cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
        self._cbar.update_ticks()
    
    def _clear_upper_air_artists(self):
        if self._contour_sets:
            for cs in self._contour_sets:
                try:
                    remover = getattr(cs, 'remove', None)
                    if callable(remover):
                        remover()
                        continue
                except Exception:
                    pass
                try:
                    for coll in getattr(cs, 'collections', []) or []:
                        try:
                            coll.remove()
                        except Exception:
                            pass
                    for txt in getattr(cs, 'labelTexts', []) or []:
                        try:
                            txt.remove()
                        except Exception:
                            pass
                except Exception:
                    pass
            self._contour_sets.clear()
        if self._contour_labels:
            for label in self._contour_labels:
                try:
                    label.remove()
                except Exception:
                    pass
            self._contour_labels.clear()
        if self._barb_art is not None:
            try:
                self._barb_art.remove()
            except Exception:
                pass
            self._barb_art = None
    
    def _draw_upper_overlays(self, lat: np.ndarray, lon: np.ndarray, data: UpperAirData, spec: UpperAirData) -> None:
        self._clear_upper_air_artists()
        
        contour_data = data.contour
        if contour_data is not None and spec.contour_field:
            contour_arr = np.asarray(contour_data)
            contour_arr = np.squeeze(contour_arr)
            if contour_arr.shape != lat.shape:
                ny = min(contour_arr.shape[0], lat.shape[0]) if contour_arr.ndim >= 1 else lat.shape[0]
                nx = min(contour_arr.shape[1], lat.shape[1]) if contour_arr.ndim >= 2 else lat.shape[1]
                contour_arr = contour_arr[:ny, :nx]
                lat = lat[:ny, :nx]
                lon = lon[:ny, :nx]
            contour_kwargs = {
                'colors': spec.contour_color,
                'linewidths':  spec.contour_width,
                'transform': ccrs.PlateCarree(),
            }
            if spec.contour_levels is not None:
                contour_kwargs['levels'] = spec.contour_levels
            try:
                cs = self.ax.contour(
                    lon,
                    lat,
                    contour_arr,
                    **contour_kwargs,
                )
            except ValueError:
                cs = None
            if cs is not None:
                self._contour_labels.append(cs)
                if spec.contour_field == 'height':
                    try:
                        labels = self.ax.clabel(cs, fmt='%d', fontsize=8)
                    except Exception:
                        labels = []
                    if labels:
                        self._contour_labels.extend(labels)
        
        stride = max(1, spec.barb_stride)
        u = np.asarray(data.u)
        v = np.asarray(data.v)
        u = np.squeeze(u)
        v = np.squeeze(v)
        if u.shape != lat.shape:
            ny = min(u.shape[0], lat.shape[0]) if u.ndim >= 1 else lat.shape[0]
            nx = min(u.shape[1], lat.shape[1]) if u.ndim >= 2 else lat.shape[1]
            u = u[:ny, :nx]
            v = v[:ny, :nx]
            lat = lat[:ny, :nx]
            lon = lon[:ny, :nx]
        sl = (slice(None, None, stride), slice(None, None, stride))
        barb_lon = lon[sl]
        barb_lat = lat[sl]
        barb_u = u[sl]
        barb_v = v[sl]
        if barb_lon.size and barb_lat.size:
            try:
                self._barb_art = self.ax.barbs(
                    barb_lon,
                    barb_lat,
                    barb_u,
                    barb_v,
                    length=spec.barb_length,
                    color=spec.barb_color,
                    linewidth=0.6,
                    transform=ccrs.PlateCarree(),
                )
            except Exception:
                self._barb_art = None
        
    # ---------------
    # Utilities
    # ---------------
    def _load_cpt(self, path: str) -> LinearSegmentedColormap:
        xs, cols = [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                parts = s.split()
                try:
                    if len(parts) >= 8:
                        x1, r1, g1, b1, x2, r2, g2, b2 = map(float, parts[:8])
                        xs.extend([x1, x2])
                        cols.extend([[r1/255.0, g1/255.0, b1/255.0], [r2/255.0, g2/255.0, b2/255.0]])
                    elif len(parts) >= 4:
                        x, r, g, b = map(float, parts[:4])
                        xs.append(x)
                        cols.append([r/255.0, g/255.0, b/255.0])
                except ValueError:
                    continue
        if not xs:
            raise ValueError('No Color stops parsed from CPT.')
        xs = np.array(xs)
        cols = np.array(cols)
        xr = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
        stops = list(zip(xr.tolist(), cols.tolist()))
        if stops[0][0] != 0.0:
            stops.insert(0, (0.0, cols[0].tolist()))
        if stops[-1][0] != 1.0:
            stops.append((1.0, cols[-1].tolist()))
        return LinearSegmentedColormap.from_list(Path(path).stem, stops)
    
    def _init_colormaps(self):
        base = {
            'Cividis': plt.get_cmap('cividis'),
            'Cool': plt.get_cmap('cool'),
            'Gist Rainbow': plt.get_cmap('gist_rainbow'),
            'Hot': plt.get_cmap('hot'),
            'Jet': plt.get_cmap('jet'),
            'PiYG': plt.get_cmap('PiYG'),
            'Plasma': plt.get_cmap('plasma'),
            'Spectral': plt.get_cmap('Spectral'),
            'Turbo': plt.get_cmap('turbo') if hasattr(plt.cm, 'turbo') else plt.get_cmap('plasma'),
            'Viridis': plt.get_cmap('viridis'),
        }
        self.cmap_registry.update(base)
        
        for d in (self.project_cmap_dir, self.user_cmap_dir):
            if not d.exists():
                continue
            for p in sorted(list(d.glob('*.cpt')) + list(d.glob('*.ct'))):
                try:
                    self.cmap_registry[p.stem] = self._load_cpt(str(p))
                except Exception:
                    pass
        
        self._populate_cmap_combo(select=self.settings.value('cmap_name', 'Viridis'))
    
    def _populate_cmap_combo(self, select: str | None = None):
        self.cmb_cmap.blockSignals(True)
        self.cmb_cmap.clear()
        names = sorted(self.cmap_registry.keys())
        self.cmb_cmap.addItems(names)
        if select and select in self.cmap_registry:
            self.cmb_cmap.setCurrentText(select)
        else:
            self.cmb_cmap.setCurrentText('Viridis')
        self.cmb_cmap.blockSignals(False)
        
# ------------------------
# Entrypoint
# ------------------------
def main():
    # On some Linux desktops, force XCB to avoid wayland plugin warnings
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('PROJ_NETWORK', 'OFF')
    os.environ.setdefault('PYPROJ_NETWORK', 'OFF')
    
    app = QApplication(sys.argv)
    win = WRFViewer()
    #win.show()
    #win.showMaximized()
    #win.setWindowState(Qt.WindowActive | Qt.WindowMaximized)
    #win.show()
    win.showMaximized()
    sys.exit(app.exec())
    
if __name__ == '__main__':
    main()
    

"""
PyInstaller packing tips
-----------------------
1) Create a virtual environment and install deps:
pip install pyinstaller PySide6 matplotlib cartopy netCDF4 wrf-python metpy pint


2) Build:
pyinstaller --noconfirm --clean --onefile --name WRFViewer \
--hidden-import cartopy \
--hidden-import wrf \
wrf_viewer.py


(You may need "--collect-data cartopy --collect-data wrf --collect-data shapely" depending on platform.)


3) If Cartopy data/features missing at runtime, pre-fetch NaturalEarth to a known path and set CARTOPY_USER_BACKGROUNDS / CARTOPY_DATA_DIR.


4) On Windows, ensure PROJ/GEOS (via conda-forge) are present. Packaging Cartopy is easier from a conda env. As an alternative, drop the basemap features or switch to simple lat/lon grid without Cartopy for a lighter EXE.
"""