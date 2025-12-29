#!/usr/bin/env python3

'''
Created by Gabe Zago with the help of AI.
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
import traceback
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpatheffects
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
import numpy as np
import os
import shutil
import sys
import typing as T

from dataclasses import dataclass
from datetime import datetime, timedelta
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
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from wrf import to_np # getvar, latlon_coords, ALL_TIMES, interplevel
from sounding import SoundingWindow

from calc import (
    calc_height,
    calc_pressure,
    calc_relative_humidity,
    calc_temperature,
    calc_wind_gust_mph,
    dbz_to_rate_inhr,
    destagger,
    ensure_pressure_orientation,
    interp_to_pressure,
    PTYPE_INTENSITY_SPAN,
    PTYPE_MAX_RATE_INHR,
    ptype_rate_offset,
    calc_updraft_helicity,
    uh_minimum_for_spacing,
    snowfall_support,
    surface_based_cape,
    surface_based_cape_from_profile,
    slice_time_var,
)


# ------------------------
# Data Structure
# ------------------------
@dataclass
class WRFFrame:
    path: str
    time_index: int
    timestamp_str: str
    timestamp: datatime | None = None


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


@dataclass
class SurfaceWindData:
    scalar: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass
class ReflectivityUHOverlays:
    reflectivity: np.ndarray
    uh_max: np.ndarray
    uh_threshold: float


UPPER_AIR_SPECS: dict[str, UpperAirSpec] = {
    'HGT500': UpperAirSpec(
        canonical='HGT500',
        display_name='500 mb Height, Wind',
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
        display_name='700 mb Relative Humidity, Wind',
        level_hpa=700.0,
        shading_field='rh',
        colorbar_label='700 hPa Relative Humidity (%)',
        vmin=0.0,
        vmax=100.0,
        contour_field='height',
        contour_levels=np.arange(2300.0, 3500.1, 60.0),
        contour_color='black',
        contour_width=0.8,
        barb_stride=16,
        barb_length=5.5,
        barb_color='black',
        title='700 hPa Relative Humidity & Wind',
    ),
    'TEMP850': UpperAirSpec(
        canonical='TEMP850',
        display_name='850 mb Temperature, Wind',
        level_hpa=850.0,
        shading_field='temperature',
        colorbar_label='850 hPa Temperature (°C)',
        vmin=-51.1,
        vmax=48.9,
        contour_field='height',
        contour_levels=np.arange(1000.0, 1700.1, 30.0),
        contour_color='black',
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
        self._uh_threshold_cache: dict[str, float] = {}
    
    @staticmethod
    def _align_xy(*arrays: np.ndarray) -> list[np.ndarray]:
        finite_arrays = [arr for arr in arrays if arr is not None]
        if not finite_arrays:
            return list(arrays)
        
        min_y = min(arr.shape[-2] for arr in finite_arrays)
        min_x = min(arr.shape[-1] for arr in finite_arrays)
        
        aligned: list[np.ndarray] = []
        for arr in arrays:
            if arr is None:
                aligned.append(arr)
                continue
            slicer = [slice(None)] * arr.ndim
            slicer[-2] = slice(0, min_y)
            slicer[-1] = slice(0, min_x)
            aligned.append(np.asarray(arr[tuple(slicer)]))
        return aligned
    
    @staticmethod
    def _scalar_value(value: T.Any) -> T.Any:
        ''' Peel away nested sequences to return the first scalar-like value. '''
        
        seen: set[int] = set()
        cur = value
        while isinstance(cur, (list, tuple, np.ndarray, set)):
            if len(cur) == 0:
                return None
            # Avoid pathological self-referential containers.
            ident = id(cur)
            if ident in seen:
                break
            seen.add(ident)
            
            try:
                # Prefer deterministic ordering for sets by sorting stringified entries.
                if isinstance(cur, set):
                    cur = sorted(cur, key=str)[0]
                elif isinstance(cur, np.ndarray):
                    cur = np.asarray(cur).ravel()
                    if cur.size == 0:
                        return None
                    cur = cur[0]
                else:
                    cur = cur[0]
            except Exception:
                break
        
        return cur
    
    @staticmethod
    def _level_key(level_hpa: T.Optional[T.Any]) -> T.Optional[float]:
        if level_hpa is None:
            return None
        
        level_hpa = WRFLoader._scalar_value(level_hpa)
        hashable = WRFLoader._to_hashable(level_hpa)
        # If a sequence of levels was provided, prefer the first finite entry.
        if isinstance(hashable, tuple):
            for entry in hashable:
                try:
                    return float(entry)
                except Exception:
                    continue
            return None
        
        try:
            return float(hashable)
        except Exception:
            return None
    
    @staticmethod
    def _var_key(var: T.Any) -> str:
        ''' Coerce any variable selector into a stable, hashable string key. '''
        
        var = WRFLoader._scalar_value(var)
        hashable = WRFLoader._to_hashable(var)
        if isinstance(hashable, tuple):
            hashable = '|'.join(str(v) for v in hashable)
        return str(hashable).upper()
    
    @staticmethod
    def _to_hashable(value: T.Any) -> T.Any:
        ''' Convert any iterable input into an immutable, hashable structure.'''
        
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple(
                (WRFLoader._to_hashable(k), WRFLoader._to_hashable(v))
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(WRFLoader._to_hashable(v) for v in value)
        if isinstance(value, np.ndarray):
            return tuple(WRFLoader._to_hashable(v) for v in value.tolist())
        if hasattr(value, '__iter__'):
            try:
                return tuple(WRFLoader._to_hashable(v) for v in list(value))
            except Exception:
                pass
        return value
    
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
    
    @staticmethod
    def _parse_timestamp(ts: str) -> datetime | None:
        for fmt in ('%Y-%m-%d_%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(ts.strip(), fmt)
            except ValueError:
                continue
        return None
    
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
                    frames.append(
                        WRFFrame(
                            path=fp,
                            time_index=ti,
                            timestamp_str=formatted_ts,
                            timestamp=self._parse_timestamp(ts),
                        )
                    )
        frames.sort(key=lambda fr: (os.path.getmtime(fr.path), fr.time_index))
        self.frames = frames
        self._cache.clear()
        self._geo_cache.clear()
        self.preloaded.clear()
        self._pressure_orientation.clear()
        self._upper_base_cache.clear()
        self._uh_threshold_cache.clear()
        
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
    
    def _uh_threshold(self, frame_path: str) -> float:
        cached = self._uh_threshold_cache.get(frame_path)
        if cached is not None:
            return cached
        
        with Dataset(frame_path) as nc:
            dx = getattr(nc, 'DX', None)
            dy = getattr(nc, 'DY', None)
        threshold = uh_minimum_for_spacing(dx, dy)
        self._uh_threshold_cache[frame_path] = threshold
        return threshold
    
    def _get_upper_base_fields(self, frame: WRFFrame) -> dict[str, np.ndarray]:
        key = (frame.path, frame.time_index)
        cached = self._upper_base_cache.get(key)
        if cached is not None:
            self._upper_base_cache.move_to_end(key)
            return cached
            
        with Dataset(frame.path) as nc:
            pressure = calc_pressure(nc, frame.time_index).astype(float32)
            height = calc_height(nc, frame.time_index).astype(float32)
            u_stag = slice_time_var(nc.variables['U'], frame.time_index)
            v_stag = slice_time_var(nc.variables['V'], frame.time_index)
            u_mass = destagger(u_stag, axis=-1).astype(float32)
            v_mass = destagger(v_stag, axis=-2).astype(float32)
            temp_k = calc_temperature(nc, pressure, frame.time_index)
            temp_c = (temp_k - 273.15).astype(float32)
            qv = slice_time_var(nc.variables['QVAPOR'], frame.time_index)
            rh = calc_relative_humidity(temp_k, pressure, qv).astype(float32)
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
    
    def get_sounding_profile(
        self, frame: WRFFrame, latitude: float, longitude: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Return a single-column temperature/pressure/height profile nearest the lat/lon.
        
        Also returns dewpoint (C) computed from the column relative humidity.
        '''
        
        lat_grid, lon_grid = self.get_geo(frame)
        if lat_grid.size == 0 or lon_grid.size == 0:
            raise RuntimeError('Latitude/Longitude grid is empty; cannot extract sounding.')
        
        dist2 = np.square(lat_grid - latitude) + np.square(lon_grid - longitude)
        if not np.isfinite(dist2).any():
            raise RuntimeError('Latitude/Longitude grid has no finite coordinates.')
        
        flat_idx = np.nanargmin(dist2)
        y_idx, x_idx = np.unravel_index(flat_idx, lat_grid.shape)
        
        base_fields = self._get_upper_base_fields(frame)
        pressure_pa = base_fields['pressure'][:, y_idx, x_idx]
        temp_c = base_fields['temperature'][:, y_idx, x_idx]
        height_m = base_fields['height'][:, y_idx, x_idx]
        rh = base_fields['rh'][:, y_idx, x_idx]
        
        orient = ensure_pressure_orientation(frame.path, base_fields['pressure'], self._pressure_orientation)
        if orient == 'ascending':
            pressure_pa = pressure_pa[::-1]
            temp_c = temp_c[::-1]
            height_m = height_m[::-1]
            rh = rh[::-1]
        
        pressure_hpa = np.asarray(pressure_pa, dtype=float32) / 100.0
        temp_c = np.asarray(temp_c, dtype=float32)
        rh = np.asarray(rh, dtype=float32)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Magnus formula using relative humidity percent and temperature in Celcius.
            gamma = np.log(np.clip(rh, 1e-6, 100.0) * 0.01) + (17.67 * temp_c) / (temp_c + 243.5)
            dewpoint_c = (243.5 * gamma) / (17.67 - gamma)
            
        valid = (
            np.isfinite(pressure_hpa)
            & np.isfinite(temp_c)
            & np.isfinite(height_m)
            & np.isfinite(dewpoint_c)
        )
        if valid.sum() < 3:
            raise RuntimeError('Sounding column contains insufficient finite data to plot.')
        
        return (
            pressure_hpa[valid],
            temp_c[valid],
            height_m[valid],
            dewpoint_c[valid]
        )
    
    def _total_precip_inches(self, nc: Dataset, frame: WRFFrame) -> np.ndarray:
        accum: T.Optional[np.ndarray] = None
        for name in ('RAINNC', 'RAINC'):
            if name not in nc.variables:
                continue
            var = nc.variables[name]
            dims = tuple(getattr(var, 'dimensions', ()))
            if 'Time' in dims:
                arr = np.array(var[frame.time_index, :, :])
            else:
                arr = np.array(var[:, :])
            accum = arr if accum is None else accum + arr
        
        if accum is None:
            raise RuntimeError('Need RAINNC or RAINC to compute snowfall accumulation.')
        return np.asarray(accum, dtype=float32) / 25.4
    
    def _precip_type_field(self, frame: WRFFrame) -> np.ndarray:
        key = (frame.path, 'PTYPE', frame.time_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        base_fields = self._get_upper_base_fields(frame)
        pressure = base_fields['pressure']
        temp_c = base_fields['temperature']
        height = base_fields['height']
        
        #self._log_debug(f'PTYPE base shapes: pressure={pressure.shape}, temp={temp_c.shape}, height={height.shape}')
        
        orient = ensure_pressure_orientation(frame.path, pressure, self._pressure_orientation)
        surface_first = orient == 'descending'
        #self._log_debug(f'PTYPE orientation: {orient} (surface_first={surface_first})')
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
        
        #self._log_debug(f'PTYPE aligned shapes: pressure={pressure.shape}, temp={temp_c.shape}, height={height.shape}, nz={nz}, ny={ny}, nx={nx}')
        
        layer_thickness = np.diff(height, axis=0, append=height[::-1, :, :])
        layer_thickness = np.clip(layer_thickness, 0.0, None)
        
        if layer_thickness.shape != temp_c.shape:
            nz_energy = min(layer_thickness.shape[0], temp_c.shape[0])
            ny_energy = min(layer_thickness.shape[1], temp_c.shape[1])
            nx_energy = min(layer_thickness.shape[2], temp_c.shape[2])
            layer_thickness = layer_thickness[:nz_energy, :ny_energy, :nx_energy]
            temp_c = temp_c[:nz_energy, :ny_energy, :nx_energy]
            pressure = pressure[:nz_energy, :ny_energy, :nx_energy]
            #self._log_debug(f'PTYPE energy align: layer_thickness shape={layer_thickness.shape}, temp shape={temp_c.shape}, pressure shape={pressure.shape}')
        
        warm_energy = np.sum(np.clip(temp_c, 0.0, None) * layer_thickness, axis=0)
        cold_energy = np.sum(np.clip(-temp_c, 0.0, None) * layer_thickness, axis=0)
        surface_temp = temp_c[0, :, :]
        max_temp = temp_c.max(axis=0)
        
        #self._log_debug(f'PTYPE energies: warm_energy shape={warm_energy.shape}, cold_energy shape={cold_energy.shape}, surface_temp shape={surface_temp.shape}, max_temp shape={max_temp.shape}')
        
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
            rate_inhr = dbz_to_rate_inhr(mdbz)
            rate_inhr = np.clip(rate_inhr, 0.0, PTYPE_MAX_RATE_INHR)
            intensity = (rate_inhr / PTYPE_MAX_RATE_INHR).astype(float32) * PTYPE_INTENSITY_SPAN
            ptype = np.where(precip_mask, ptype.astype(float32) + intensity, np.nan)
        except Exception as exc:
            self._log_debug(f'PTYPE: Unable to apply precipitation mask ({exc})')
        
        self._cache[key] = ptype.astype(float32)
        return self._cache[key]
    
    def _snowfall_10_to_1(self, frame: WRFFrame) -> np.ndarray:
        try:
            target_idx = self.frames.index(frame)
        except ValueError:
            raise RuntimeError('Frame not found for snowfall calculation.')
        
        for idx in range(target_idx + 1):
            fr = self.frames[idx]
            snow_key = (fr.path, 'SNOW10', fr.time_index)
            if snow_key in self._cache:
                continue
            
            with Dataset(fr.path) as nc:
                precip_total = self._total_precip_inches(nc, fr)
            
            precip_key = (fr.path, 'PRECIP_TOTAL_IN', fr.time_index)
            
            if idx == 0:
                prev_accum = np.zeros_like(precip_total, dtype=float32)
                prev_precip = np.zeros_like(precip_total, dtype=float32)
            else:
                prev_frame = self.frames[idx - 1]
                prev_accum = np.asarray(
                    self._cache[(prev_frame.path, 'SNOW10', prev_frame.time_index)],
                    dtype=float32,
                )
                prev_precip = np.asarray(
                    self._cache[(prev_frame.path, 'PRECIP_TOTAL_IN', prev_frame.time_index)],
                    dtype=float32,
                )
            
            base_fields = self._get_upper_base_fields(fr)
            ptype = self._precip_type_field(fr)
            support = snowfall_support(base_fields['temperature'], ptype)
            
            precip_total, prev_accum, prev_precip, support = self._align_xy(
                precip_total, prev_accum, prev_precip, support
            )
            
            reset_mask = precip_total < prev_precip
            if np.any(reset_mask):
                prev_precip = np.where(reset_mask, 0.0, prev_precip)
            
            delta_liq = np.clip(precip_total - prev_precip, 0.0, None)
            snowfall_increment = delta_liq * 10.0 * support
            accum = np.clip(prev_accum + snowfall_increment, 0.0, None)
            
            self._cache[precip_key] = precip_total.astype(float32)
            self._cache[snow_key] = accum.astype(float32)
            
        return self._cache[(frame.path, 'SNOW10', frame.time_index)]
    
    def _uh_field(self, frame: WRFFrame) -> np.ndarray:
        key = (frame.path, 'UH2TO5', frame.time_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        with Dataset(frame.path) as nc:
            uh = calc_updraft_helicity(nc, frame.time_index)
        self._cache[key] = uh.astype(float32)
        return self._cache[key]
    
    def _uh_max_last_hour(self, target_idx: int) -> np.ndarray:
        frame = self.frames[target_idx]
        key = (frame.path, 'UHMAX1HR', frame.time_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        
        target_time = frame.timestamp
        lower_bound = target_time - timedelta(hours=1) if target_time else None
        
        uh_fields: list[np.ndarray] = []
        for idx in range(target_idx, -1, -1):
            fr = self.frames[idx]
            if lower_bound is not None and fr.timestamp is not None and fr.timestamp < lower_bound:
                break
            uh_fields.append(self._uh_field(fr))
            if lower_bound is None:
                break
        
        aligned = self._align_xy(*uh_fields)
        finite = [arr for arr in aligned if arr is not None]
        if not finite:
            result = np.array([], dtype=float32)
        else:
            stack = np.stack(finite, axis=0)
            with np.errstate(invalid='ignore'):
                result = np.nanmax(stack, axis=0)
        
        self._cache[key] = result.astype(float32, copy=False)
        return self._cache[key]
    
    def _reflectivity_with_uh(self, frame: WRFFrame) -> ReflectivityUHOverlays:
        try:
            target_idx = self.frames.index(frame)
        except ValueError:
            raise RuntimeError('Frame not found for UH overlay calculation.')
        
        mdbz = self.get2d(frame, 'MDBZ')
        uh_max = self._uh_max_last_hour(target_idx)
        threshold = self._uh_threshold(frame.path)
        
        aligned_refl, aligned_uh = self._align_xy(mdbz, uh_max)
        return ReflectivityUHOverlays(
            reflectivity=np.asarray(aligned_refl, dtype=float32),
            uh_max=np.asarray(aligned_uh, dtype=float32),
            uh_threshold=float(threshold),
        )
    
    def is_upper_air(self, var: str) -> bool:
        return self._var_key(var) in UPPER_AIR_SPECS
    
    def get_upper_air_data(self, frame: WRFFrame, var: str) -> UpperAirData:
        v = self._var_key(var)
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
        scalar2d = interp_to_pressure(scalar3d, pressure, spec.level_hpa, frame.path, self._pressure_orientation)
        
        contour2d = None
        if spec.contour_field:
            contour3d = base_fields.get(spec.contour_field)
            if contour3d is None:
                raise RuntimeError(f'Missing contour field "{spec.contour_field}" for {var}.')
            contour2d = interp_to_pressure(contour3d, pressure, spec.level_hpa, frame.path, self._pressure_orientation)
        
        u2d = interp_to_pressure(base_fields['u'], pressure, spec.level_hpa, frame.path, self._pressure_orientation)
        v2d = interp_to_pressure(base_fields['v'], pressure, spec.level_hpa, frame.path, self._pressure_orientation)
        
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
        key = (self._var_key(var), self._level_key(level_hpa))
        if key not in self.preloaded:
            self.preloaded[key] = [None] * len(self.frames)
            
    def set_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int, data: T.Any) -> None:
        key = (self._var_key(var), self._level_key(level_hpa))
        self.preloaded.setdefault(key, [None] * len(self.frames))
        self.preloaded[key][idx] = data
    
    def get_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int) -> T.Optional[T.Any]:
        key = (self._var_key(var), self._level_key(level_hpa))
        li = self.preloaded.get(key)
        return None if li is None else li[idx]
        
    def clear_preloaded(self):
        self.preloaded.clear()
    
    # --- Data Access ---
    def get2d(self, frame: WRFFrame, var: str, level_hpa: T.Optional[float] = None) -> np.ndarray:
        lvl_key = self._level_key(level_hpa)
        v = self._var_key(var)
        key = (frame.path, f'{v}@{lvl_key}', frame.time_index)
        if key in self._cache:
            return self._cache[key]
        
        with Dataset(frame.path) as nc:
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
            elif v == 'MDBZ_1HRUH':
                data2d = self._reflectivity_with_uh(frame)
            elif v == 'RAINNC':
                data2d = np.array(nc.variables['RAINNC'][frame.time_index, :, :])
            elif v == 'RAINC':
                data2d = np.array(nc.variables['RAINC'][frame.time_index, :, :])
            elif v == 'SNOW10':
                data2d = self._snowfall_10_to_1(frame)
            elif v == 'GUST':
                data2d = calc_wind_gust_mph(nc, frame.time_index)
            elif v == 'WSPD10':
                u10 = np.array(nc.variables['U10'][frame.time_index, :, :])
                v10 = np.array(nc.variables['V10'][frame.time_index, :, :])
                data2d = np.hypot(u10, v10)
            elif v == 'T2F':
                if 'T2' not in nc.variables:
                    raise RuntimeError('Variable "T2" not found; cannot compute 2 m temperature.')
                t2_var = nc.variables['T2']
                dims = tuple(getattr(t2_var, 'dimensions', ()))
                if 'Time' in dims:
                    t2k = np.array(t2_var[frame.time_index, :, :])
                else:
                    t2k = np.array(t2_varp[:, :])
                t2c = t2k - 273.15
                data2d = (t2c * 9.0 / 5.0) + 32.0
            elif v == 'TD2F':
                missing = [name for name in ('Q2', 'PSFC') if name not in nc.variables]
                if missing:
                    raise RuntimeError(f'Missing required variables for dewpoint: {", ".join(missing)}')
                    
                q2_var = nc.variables['Q2']
                psfc_var = nc.variables['PSFC']
                
                def _slice(v):
                    dims = tuple(getattr(v, 'dimensions', ()))
                    if 'Time' in dims:
                        return np.array(v[frame.time_index, :, :])
                    return np.array(v[:, :])
                
                q2 = _slice(q2_var)
                psfc = _slice(psfc_var) # Pa
                
                # Convert to dewpoint using vapor pressure from mixing ratio.
                # e = (q * p) / (0.622 + q)
                e_pa = (q2 * psfc) / (0.622 + q2)
                e_pa = np.clip(e_pa, 1e-6, None)
                log_ratio = np.log(e_pa / 611.2)
                td_c = (243.5 * log_ratio) / (17.67 - log_ratio)
                data2d = (td_c * 9.0 / 5.0) + 32.0
            elif v == 'RH2WIND10KT':
                required = ['T2', 'Q2', 'PSFC', 'U10', 'V10']
                missing = [name for name in required if name not in nc.variables]
                if missing:
                    raise RuntimeError(f'Missing required variables for 2 m RH / 10 m wind: {", ".join(missing)}')
                
                def _slice(vname: str) -> np.ndarray:
                    var = nc.variables[vname]
                    dims = tuple(getattr(var, 'dimensions', ()))
                    if 'Time' in dims:
                        return np.array(var[frame.time_index, :, :])
                    return np.array(var[:, :])
                
                t2_k = _slice('T2')
                q2 = _slice('Q2')
                psfc = _slice('PSFC')
                u10 = _slice('U10')
                v10 = _slice('V10')
                
                # Convert T2 (K) and mixing ratio to relative humidity (percent).
                e_pa = (q2 * psfc) / (0.622 + q2)
                e_pa = np.clip(e_pa, 1e-6, None)
                tc = t2_k - 273.15
                es_pa = 611.2 * np.exp((17.67 * tc) / (tc + 243.5))
                rh = np.clip((e_pa / es_pa) * 100.0, 0.0, 100.0)
                
                ms_to_kt = 1.9438444924406
                data2d = SurfaceWindData(
                    scalar=rh.astype(float32),
                    u=(u10 * ms_to_kt).astype(float32),
                    v=(v10 * ms_to_kt).astype(float32),
                )
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
            elif v == 'SBCAPE':
                base_fields = self._get_upper_base_fields(frame)
                orient = ensure_pressure_orientation(
                    frame.path, base_fields['pressure'], self._pressure_orientation
                )
                pressure = base_fields['pressure']
                temperature = base_fields['temperature']
                rh = base_fields['rh']
                height = base_fields['height']
                if orient == 'ascending':
                    pressure = pressure[::-1, :, :]
                    temperature = temperature[::-1, :, :]
                    rh = rh[::-1, :, :]
                    height = height[::-1, :, :]
                
                temp_k = temperature + 273.15
                data2d = surface_based_cape(pressure, temp_k, rh, height)
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
        self.var = loader._var_key(var)
        self.level_hpa = loader._level_key(level_hpa)
    
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
        self.resize(1400, 900)
        
        self.loader = WRFLoader()
        self.upper_air_specs = UPPER_AIR_SPECS
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100) # ms
        self.timer.timeout.connect(self._tick)
        self._stepping = False
        self._sounding_windows: list[SoundingWindow] = []
        self._uh_threshold_text: float | None = None
        
        # --- App Settings ---
        self.settings = QtCore.QSettings('WRFViewer1', 'WRFViewer')
        self.user_cmap_dir = Path.cwd() / 'Colortable'
        self.project_cmap_dir = self.user_cmap_dir
        for d in (self.user_cmap_dir, self.project_cmap_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.cmap_registry: dict[str, LinearSegmentedColormap] = {}
        self._var_cmap_lookup: dict[str, str] = {
            'MDBZ': 'Reflectivity',
            'MAXDBZ': 'Reflectivity',
            'MDBZ_1HRUH': 'Reflectivity',
            'REFL1KM': 'Reflectivity',
            'GUST': 'Wind_Gust',
            'WSPD10': 'Wind_Gust',
            'RH2WIND10KT': 'Relative-humidity',
            'RH700': 'Relative-humidity',
            'T2F': 'Temperature',
            'TEMP850': 'Temperature',
            'TD2F': 'Dewpoint',
            'SNOW10': 'Snowfall',
            'SBCAPE': 'Cape',
        }
        
        # --- Central ---
        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # --- Variable categories (accordion data) ---
        self.var_categories: dict[str, list[tuple[str, str]]] = {
            'Surface and Precipitation': [
                {'label': 'Surface', 'canonical': None, 'is_divider': True},
                {'label': '10 m AGL Wind Gusts', 'canonical': 'GUST'},
                {'label': '2 m AGL Temperature', 'canonical': 'T2F'},
                {'label': '2 m AGL Dewpoint', 'canonical': 'TD2F'},
                {'label': '2 m AGL Relative Humidity, Wind Barbs', 'canonical': 'RH2WIND10KT'},
                {'label': '10 m AGL Wind', 'canonical': 'WSPD10'},
                {'label': 'Precipitation Type', 'canonical': None, 'is_divider': True},
                {'label': 'Precipitation Type, Rate', 'canonical': 'PTYPE'},
                {'label': 'Quantitative Precipitation', 'canonical': None, 'is_divider': True},
                {'label': 'Total Rain Accumulation', 'canonical': 'RAINNC'},
                {'label': 'Radar Products', 'canonical': None, 'is_divider': True},
                {'label': 'Composite Reflectivity', 'canonical': 'MDBZ'},
            ],
            'Winter Weather': [
                {'label': 'Snowfall (10:1 Ratio)', 'canonical': None, 'is_divider': True},
                {'label': 'Total Snowfall (10:1)', 'canonical': 'SNOW10'},
            ],
            'Severe Weather': [
                {'label': 'Instability', 'canonical': None, 'is_divider': True},
                {'label': 'Surface-Based CAPE', 'canonical': 'SBCAPE'},
                {'label': 'Excplicit Convective Products', 'canonical': None, 'is_divider': True},
                {'label': 'Reflectivity, UH>75', 'canonical': 'MDBZ_1HRUH'},
                {'label': 'Composite Reflectivity', 'canonical': 'MDBZ'},
                {'label': 'Reflectivity 1km', 'canonical': 'REFL1KM'},
            ],
            'Upper Air: Height, Wind, Temperature': [
                {'label': 'Height and Wind', 'canonical': None, 'is_divider': True},
                {'label': UPPER_AIR_SPECS['HGT500'].display_name, 'canonical': 'HGT500'},
                {'label': 'Temperature and Wind', 'canonical': None, 'is_divider': True},
                {'label': UPPER_AIR_SPECS['TEMP850'].display_name, 'canonical': 'TEMP850'},
            ],
            'Upper Air: Moisture': [
                {'label': 'Relative Humidity and Wind', 'canonical': None, 'is_divider': True},
                {'label': UPPER_AIR_SPECS['RH700'].display_name, 'canonical': 'RH700'},
            ],
        }
        self._var_aliases: dict[str, str] = {
            entry['label'].upper(): entry['canonical'].upper()
            for items in self.var_categories.values()
            for entry in items
            if entry.get('canonical')
        }

        # ensure canonical names include themselves
        for canonical in {
            entry['canonical']
            for items in self.var_categories.values()
            for entry in items
            if entry.get('canonical')
        }:
            self._var_aliases.setdefault(canonical.upper(), canonical.upper())
        
        self._var_aliases.setdefault('SURFACE-BASED CAPE', 'SBCAPE')
        
        
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
            for entry in items:
                label = entry['label']
                canonical = entry.get('canonical')
                if canonical and label not in seen_labels:
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
        
        self.btn_generate_sounding = QPushButton('Generate Sounding')
        self.btn_generate_sounding.clicked.connect(self.on_generate_sounding)
        controls.addWidget(self.btn_generate_sounding)
        
        controls.addWidget(QLabel('Latitude: '))
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText('Click map or enter')
        self.lat_input.setFixedWidth(110)
        controls.addWidget(self.lat_input)
        
        controls.addWidget(QLabel('Longitude: '))
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText('Click map or enter')
        self.lon_input.setFixedWidth(110)
        controls.addWidget(self.lon_input)
        
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
        self._click_cid = self.canvas.mpl_connect('button_press_event', self.on_map_click)
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
            list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            divider_indices = [i for i, entry in enumerate(items) if entry.get('is_divider', False)]
            first_divider_idx = divider_indices[0] if divider_indices else None
            last_divider_idx = divider_indices[-1] if divider_indices else None
            for idx, entry in enumerate(items):
                label = entry['label']
                canonical = entry.get('canonical')
                is_divider = entry.get('is_divider', False)
                
                if (
                    is_divider
                    and divider_indices
                    and idx != first_divider_idx
                ):
                    spacer = QListWidgetItem('')
                    spacer.setFlags(Qt.NoItemFlags)
                    list_widget.addItem(spacer)
                
                item = QListWidgetItem(label)
                if canonical:
                    item.setData(Qt.UserRole, label)
                    item.setData(Qt.UserRole + 1, canonical)
                if is_divider:
                    font = item.font()
                    font.setBold(True)
                    font.setPointSize(font.pointSize() + 1)
                    item.setFont(font)
                    item.setForeground(QtGui.QBrush(Qt.black))
                    item.setFlags(Qt.ItemIsEnabled)
                elif not canonical:
                    item.setFlags(Qt.NoItemFlags)
                list_widget.addItem(item)
            list_widget.itemClicked.connect(self._on_category_var_selected)
            page_layout.addWidget(list_widget)
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
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setSizes([250, 1080])
        left_panel.setMinimumWidth(250)

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
        self._uh_artists: list = []
        
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
    def on_map_click(self, event):
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        self.lat_input.setText(f'{event.ydata:.4f}')
        self.lon_input.setText(f'{event.xdata:.4f}')
    
    def on_generate_sounding(self):
        lat_text = self.lat_input.text().strip()
        lon_text = self.lon_input.text().strip()
        
        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            QMessageBox.warning(
                self,
                'Invalid coordinates',
                'Enter numeric latitude and longitude values or click the map.',
            )
            return
        
        if not self.loader.frames:
            QMessageBox.warning(self, 'No data loaded', 'Open wrfout data before generating a sounding.')
            return
        
        idx = min(max(0, self.sld_time.value()), len(self.loader.frames) - 1)
        frame = self.loader.frames[idx]
        
        try:
            pressure_hpa, temp_c, height_m, dewpoint_c = self.loader.get_sounding_profile(frame, lat, lon)
        except Exception as exc:
            QMessageBox.critical(self, 'Sounding failed', str(exc))
            return
        
        sbcape = surface_based_cape_from_profile(pressure_hpa, temp_c, dewpoint_c, height_m)
        
        wnd = SoundingWindow(
            frame.timestamp_str,
            lat,
            lon,
            pressure_profile_hpa=pressure_hpa,
            temperature_profile_c=temp_c,
            height_profile_m=height_m,
            dewpoint_profile_c=dewpoint_c,
            sbcape_jkg=sbcape,
            parent=self,
        )
        wnd.show()
        wnd.showFullScreen()
        self._sounding_windows.append(wnd)
    
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
        if text is None:
            return
        while isinstance(text, (list, tuple, np.ndarray)):
            if len(text) == 0:
                return
            text = text[0]
        text = str(text)
        if not text:
            return
        self.current_var_label = text
        self._apply_var_colormap(self._canonical_var(text))
        self.loader.clear_preloaded()
        self._sync_category_selection(text)
        self.update_plot()

    def _on_category_var_selected(self, item: QListWidgetItem):
        if item is None:
            return
        canonical = item.data(Qt.UserRole + 1)
        if not canonical:
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

    def _canonical_var(self, var: str | T.Sequence[T.Any]) -> str:
        ''' Normalize a label or alias into a canonical variable string.'''
        
        if var is None:
            return ''
        
        var = WRFLoader._scalar_value(var)
        var = WRFLoader._to_hashable(var)
        if isinstance(var, tuple):
            if len(var) == 0:
                return ''
            var = var[0]
        
        var_key = str(var).upper()
        return self._var_aliases.get(var_key, var_key)
    
    def _apply_var_colormap(self, canonical_var: str) -> None:
        name = self._var_cmap_lookup.get(canonical_var.upper(), 'Jet')
        cmap = self.cmap_registry.get(name)
        if cmap is None:
            return
        
        self.cmb_cmap.blockSignals(True)
        self.cmb_cmap.setCurrentText(name)
        self.cmb_cmap.blockSignals(False)
        self.on_cmap_changed(name)
    
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
        path, _ = QFileDialog.getOpenFileName(self, 'Load CPT colormap', start_dir, 'CPT (*.ct *.cpt);;All files (*)')
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
        display_var: T.Any = self.current_var_label
        while isinstance(display_var, (list, tuple, np.ndarray)):
            if len(display_var) == 0:
                QMessageBox.critical(self, 'Plot error', 'No variable selected.')
                return
            display_var = display_var[0]
        display_var = str(display_var)
        self.current_var_label = display_var
        var = self.loader._var_key(self._canonical_var(display_var))

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
        var = self.loader._var_key(self._canonical_var(display_var))
        self._apply_var_colormap(var)
        spec = self.upper_air_specs.get(var)
        level = spec.level_hpa if spec else None
        vector_data: T.Optional[SurfaceWindData] = None
        overlay_obj: T.Optional[ReflectivityUHOverlays] = None

        data_obj = self.loader.get_preloaded(var, level, idx)
        if data_obj is None:
            try:
                if spec:
                    data_obj = self.loader.get_upper_air_data(frame, var)
                else:
                    data_obj = self.loader.get2d(frame, var, None)
            except Exception as e:
                traceback.print_exc()
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
            overlay_obj = data_obj if isinstance(data_obj, ReflectivityUHOverlays) else None
            vector_data = data_obj if isinstance(data_obj, SurfaceWindData) else None
            if overlay_obj is not None:
                data = np.asarray(overlay_obj.reflectivity)
            else:
                data = np.asarray(data_obj.scalar if isinstance(data_obj, SurfaceWindData) else data_obj)
            
            overlay_field = np.asarray(overlay_obj.uh_max) if overlay_obj is not None else None
        
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
            if overlay_field is not None and overlay_field.ndim >= 2:
                overlay_field = overlay_field[:ny, :nx]
        else:
            ny, nx = plot_lat.shape
            if overlay_field is not None and overlay_field.shape != (ny, nx):
                overlay_field = overlay_field[:ny, :nx]
        
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
            #self.loader._log_debug(f'PTYPE plot alignment: data shape={data.shape}, lat shape={plot_lat.shape}, lon shape={plot_lon.shape}, ticks={tick_values}')
        
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
        elif vector_data is not None:
            self._draw_surface_barbs(plot_lat, plot_lon, vector_data)
        else:
            self._clear_upper_air_artists()
        
        if overlay_obj is not None:
            self._draw_uh_overlay(plot_lat, plot_lon, overlay_field if overlay_field is not None else np.array([]), overlay_obj.uh_threshold)
        else:
            self._clear_uh_overlay()
        
        self._uh_threshold_text = overlay_obj.uh_threshold if overlay_obj is not None else None
        self.ax.set_title(self._title_text(display_var, var), loc='center', fontsize=12, fontweight='bold')
        self._draw_value_labels(plot_lat, plot_lon, data, var)
        self.canvas.draw_idle()
        
    def _default_range(self, var: str) -> tuple[T.Optional[float], T.Optional[float], str]:
        v = var.upper()
        if v in ('MDBZ', 'MAXDBZ'):
            return 5.0, 70.0, 'Reflectivity (dBZ)'
        if v == 'MDBZ_1HRUH':
            return 5.0, 70.0, 'Reflectivity (dBZ)'
        if v in ('RAINNC', 'RAINC'):
            return 0.0, None, 'Accumulated Precip (mm)'
        if v == 'GUST':
            return 15.0, 75.0, '10-m wind gust (mph)'
        if v in ('WSPD10', 'WIND10'):
            return 0.0, 40.0, '10-m wind speed (m s$^{-1}$)'
        if v == 'RH2WIND10KT':
            return 0.0, 100.0, '2-m Relative Humidity (%)'
        if v == 'T2F':
            return -60.0, 120.0, '2-m temperature (°F)'
        if v == 'TD2F':
            return -40.0, 90, '2-m dewpoint (°F)'
        if v == 'REFL1KM':
            return 5.0, 70.0, 'Reflectivity @ 1km AGL (dBZ)'
        if v == 'SNOW10':
            return 0.0, 60.0, 'Total Snowfall (in)'
        if v == 'PTYPE':
            # Keep the precipitation-type colorbar aligned to fixed 0-4 boundaries so
            # each category begins at an integer tick (0=rain, 1=snow, 2=mix, 3=sleet)
            # regardless of the intensity span we use inside each bucket.
            return -0.5, 4.0, 'Precipitation Type (in/hr rates)'
        if v == 'SBCAPE':
            return 0.0, 10000.0, 'Surface-Based CAPE (J kg$^{-1}$)'
        return None, None, var
    
    def _title_text(self, display_var: str, canonical_var: str) -> str:
        v = canonical_var.upper()
        if v in self.upper_air_specs:
            spec = self.upper_air_specs[v]
            return spec.title or display_var
        if v in ('MDBZ', 'MAXDBZ'):
            return 'MDBZ (Column Max dBZ)'
        if v == 'MDBZ_1HRUH':
            thr = getattr(self, '_uh_threshold_text', None)
            thr_txt = f'{thr:.0f}' if thr is not None and np.isfinite(thr) else '?'
            return f'Composite Reflectivity & 1-hr UH > {thr_txt} m$^2$ s$^{{-2}}$'
        if v == 'REFL1KM':
            return 'Reflectivity @ 1 km AGL (dBZ)'
        if v == 'PTYPE':
            return 'Precipitation Type'
        if v == 'SNOW10':
            return 'Total Snowfall (10:1)'
        if v == 'RH2WIND10KT':
            return '2 m Relative Humidity & 10 m Wind (kt)'
        if v == 'T2F':
            return '2 m Temperature (°F)'
        if v == 'TD2F':
            return '2 m Dewpoint (°F)'
        if v == 'SBCAPE':
            return 'Surface-Based CAPE (J kg$^{-1}$)'
        return display_var
    
    def _precip_type_style(self) -> tuple[ListedColormap, BoundaryNorm, list[int], list[str]]:
        if not hasattr(self, '_ptype_cmap'):
            steps_per_type = 96
            type_gradients = [
                ['#FFFFFF', '#BDE9BF', '#79B07B', '#498F53', '#498F53', '#52943C', '#BDD15D', '#FABF56', '#FF8B3C'], # Rain
                ['#BDE1F3', '#5BA1C7', '#0E4C66', '#B1258C', '#EDD1E8'], # Snow
                ['#F1CDDA', '#EB8880', '#ED4835', '#B43D37', '#7F3243'], # Mix / Freezing
                ['#DEC1EC', '#B56CD0', '#9B2EBF', '#72297F', '#491C31'], # Sleet
            ]
            
            def _ramp_list(colors: list[str], n: int) -> list[tuple[float, float, float]]:
                cmap = mcolors.LinearSegmentedColormap.from_list('ptype_tmp', colors, N=n)
                return [tuple(c[:5]) for c in cmap(np.linspace(0.0, 1.0, n))]
            
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
            return float(ptype_rate_offset(rate))
        
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
        if var.upper() not in {'T2F', 'TD2F', 'GUST', 'WSPD10', 'SNOW10'}:
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
        
        stride = 20
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
                    fontsize=12,
                    ha='center',
                    va='center',
                    color='black',
                )
                self._value_labels.append(txt)
    
    def _clear_uh_overlay(self) -> None:
        if not self._uh_artists:
            return
        for art in self._uh_artists:
            try:
                remover = getattr(art, 'remove', None)
                if callable(remover):
                    remover()
                    continue
            except Exception:
                pass
            try:
                for coll in getattr(art, 'collections', []) or []:
                    try:
                        coll.remove()
                    except Exception:
                        pass
            except Exception:
                pass
        self._uh_artists.clear()
    
    def _draw_uh_overlay(self, lat: np.ndarray, lon: np.ndarray, uh_field: np.ndarray, threshold: float) -> None:
        self._clear_uh_overlay()
        if uh_field.size == 0:
            return
        
        uh_arr = np.asarray(uh_field)
        mask = np.isfinite(uh_arr) & (uh_arr >= threshold)
        if not np.any(mask):
            return
        
        uh_masked = np.ma.masked_invalid(uh_arr)
        finite_vals = uh_masked.compressed()
        if finite_vals.size == 0:
            return
        
        max_uh = np.nanmax(finite_vals)
        if not np.isfinite(max_uh):
            return
        
        if max_uh <= threshold:
            max_uh = threshold + 1e-6
        
        fill = self.ax.contourf(
            lon,
            lat,
            uh_masked,
            levels=[threshold, max_uh],
            colors=['#8B4513'],
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=8,
        )
        '''halo = self.ax.contour(
            lon,
            lat,
            uh_masked,
            levels=[threshold],
            colors='white',
            linewidths=2,
            linestyles='solid',
            transform=ccrs.PlateCarree(),
            zorder=20
        )'''
        outline = self.ax.contour(
            lon,
            lat,
            uh_masked,
            levels=[threshold],
            colors='black',
            linewidths=1,
            linestyles='solid',
            transform=ccrs.PlateCarree(),
            zorder=21,
        )
        
        '''if outline is not None:
            collections = getattr(outline, 'collections', None)
            if collections:
                for coll in collections:
                    coll.set_path_effects(
                        [
                            mpatheffects.Stroke(linewidth=4.6, foreground='white'),
                            mpatheffects.Normal(),
                        ]
                    )
            elif hasattr(outline, 'set_path_effects'):
                outline.set_path_effects(
                    [
                        mpatheffects.Stroke(linewidth=4.6, foreground='white'),
                            mpatheffects.Normal(),
                    ]
                )'''
        
        '''if halo is not None:
            self._uh_artists.append(halo)'''
        if outline is not None:
            self._uh_artists.append(outline)
        if fill is not None:
            self._uh_artists.append(fill)
    
    def _reset_colorbar_ticks(self) -> None:
        if not self._cbar:
            return
        # Restore any tweaks appied by precipitation-type plots so other colorbars
        # use the standard right-hand tick placement without extra offsets.
        ax = self._cbar.ax
        ax.tick_params(axis='y', which='both', labelright=True, labelleft=False, pad=2)
        ax.yaxis.set_label_position('right')
        locator = mticker.AutoLocator()
        formatter = mticker.ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-6, 6))
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        self._cbar.update_ticks()
        
        # Matplotlib may refuse tick Text instances between colorbars. Reset the
        # transform for every tick so labels that were nudged for precipitation
        # type plots don't stay offset when we switch back to continuous scales
        # like snowfall totals.
        base_transform = ax.get_yaxis_text2_transform(0)[0]
        offset_transform = mtransforms.offset_copy(base_transform, fig=ax.figure, x=4, units='points')
        for lbl in ax.get_yticklabels():
            lbl.set_transform(offset_transform)
            lbl.set_horizontalalignment('left')
    
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
    
    def _draw_surface_barbs(self, lat: np.ndarray, lon: np.ndarray, data: SurfaceWindData) -> None:
        self._clear_upper_air_artists()
        
        stride = 12
        barb_length = 5.5
        barb_color = 'black'
        
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
        
        start_y = stride if lat.shape[0] > 1 else 0
        start_x = stride if lon.shape[1] > 1 else 0
        sl = (slice(start_y, None, stride), slice(start_x, None, stride))
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
                    length=barb_length,
                    color=barb_color,
                    linewidth=0.6,
                    transform=ccrs.PlateCarree(),
                )
            except Exception:
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