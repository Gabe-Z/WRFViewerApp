from __future__ import annotations

import numpy as np
from matplotlib.transforms import Affine2D
from netCDF4 import Dataset
from numpy import float32
from wrf import interplevel, rh

# Thermodynamic constants
RD = 287.05 # J/(kg*K)
CP = 1004.0 # J/(kg*K)
P0 = 100000.0 # Pa

PTYPE_INTENSITY_SPAN = 0.995
PTYPE_MAX_RATE_INHR = 0.5


def ptype_rate_offset(rate: np.ndarray | float) -> np.ndarray | float:
    '''Map precipitation rate (in/hr) to an intensity offset inside the band.'''
    
    rate_arr = np.asarray(rate, dtype=float32)
    break_rates = np.array([0.0, 0.01, 0.05, 0.25, PTYPE_MAX_RATE_INHR], dtype=float32)
    break_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float32) * PTYPE_INTENSITY_SPAN
    clamped = np.clip(rate_arr, break_rates[0], break_rates[-1])
    offset = np.interp(clamped, break_rates, break_positions)
    if np.isscalar(rate):
        return float(offset)
    return offset.astype(float32)


def slice_time_var(var_obj, time_index: int) -> np.ndarray:
    dims = tuple(getattr(var_obj, 'dimensions', ()))
    if 'Time' in dims:
        axis = dims.index('Time')
        slicer = [slice(None)] * var_obj.ndim
        slicer[axis] = time_index
        data = np.array(var_obj[tuple(slicer)])
    else:
        data = np.array(var_obj[:])
    return data


def destagger(arr: np.ndarray, axis: int) -> np.ndarray:
    slicer1 = [slice(None)] * arr.ndim
    slicer2 = [slice(None)] * arr.ndim
    slicer1[axis] = slice(0, -1)
    slicer2[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slicer1)] + arr[tuple(slicer2)])


def calc_wind_gust_mph(nc: Dataset, time_index: int) -> np.ndarray:
    ''' 10 m wind gust (mph) from WSPD10MAX (m/s). '''
    
    if 'WSPD10MAX' not in nc.variables:
        raise RuntimeError('Variable "WSPD10MAX" not found; cannot compute wind gusts.')
    
    gust_ms = slice_time_var(nc.variables['WSPD10MAX'], time_index)
    gust_ms = np.asarray(gust_ms, dtype=float32)
    
    ms_to_mph = 2.2369362920544
    return (gust_ms * ms_to_mph).astype(float32)


def calc_pressure(nc: Dataset, time_index: int) -> np.ndarray:
    ''' Mass-level pressure (Pa) from perturbation + base-state.'''
    
    p_pert = slice_time_var(nc.variables['P'], time_index)
    p_base = slice_time_var(nc.variables['PB'], time_index)
    return (p_pert + p_base).astype(float32)


def calc_height(nc: Dataset, time_index: int) -> np.ndarray:
    ''' Mass-level geometric height (m) via PH/PHB geopotential.'''
    
    ph = slice_time_var(nc.variables['PH'], time_index)
    phb = slice_time_var(nc.variables['PHB'], time_index)
    geo = ph + phb
    # PH/PHB are staggered only along the vertical (bottom_top_stag). After
    # slicing out the time dimension the vertical coordinate sits at axis 0, so
    # destagger along axis 0 to recover mass-level heights. Using axis=1 would
    # average adjacent rows in the y-direction and collapse the heights toward
    # zero, which corrupts upper-air products like 500 hPa geopotential height.
    geo_mass = destagger(geo, axis=0)
    return (geo_mass / 9.81).astype(float32)


def calc_temperature(nc: Dataset, pressure: np.ndarray, time_index: int) -> np.ndarray:
    ''' Absolute air temperature (K) from perturbation potential temperature.'''
    
    theta_pert = slice_time_var(nc.variables['T'], time_index)
    theta = theta_pert + 300.0
    with np.errstate(invalid='ignore'):
        temp_k = theta * np.power(pressure / P0, RD / CP, dtype=float32)
    return temp_k.astype(float32)


def calc_relative_humidity(temp_k: np.ndarray, pressure: np.ndarray, qv: ndarray) -> np.ndarray:
    ''' Relative humidity (%) using mixing ratio (kg/kg), temperature (K), and pressure (Pa).'''
    
    # Convert mixing ratio (r) to specific humidity (q) so vapor pressure follows the
    # standard thermodynamic relationships. This avoids ambiguity about wrf.rh input
    # units (C vs K, hPa vs Pa) and keeps the calculation fully explicit.
    r = np.asarray(qv, dtype=float32)
    temp_k = np.asarray(temp_k, dtype=float32)
    pressure = np.asarray(pressure, dtype=float32)
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Specific humidity q = r / (1 + r)
        q = r / (1.0 + r)
        
        # Saturation vapor pressure over liquid water (Pa) using Tetens formula.
        temp_c = temp_k - 273.15
        es = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5), dtype=float32)
        
        # Saturation mixing ratio (kg/kg) and RH.
        epsilon = 0.622
        rs = epsilon * es / np.clip(pressure - es, 1e-6, None)
        rh_pct = 100.0 * (r / rs)
        
    return np.clip(rh_pct, 0.0, 100.0, out=np.empty_like(rh_pct, dtype=float32))


def ensure_pressure_orientation(
    frame_path: str, pressure: np.ndarray, orientation_cache: dict[str, str]
) -> str:
    orient = orientation_cache.get(frame_path)
    if orient:
        return orient
    surf_med = np.nanmedian(pressure[0, :, :])
    top_med = np.nanmedian(pressure[-1, :, :])
    
    if np.isfinite(surf_med) and np.isfinite(top_med) and surf_med != top_med:
        orient = 'ascending' if surf_med <= top_med else 'descending'
    else:
        sample = pressure[:, pressure.shape[1] // 2, pressure.shape[2] // 2]
        sample = sample[np.isfinite(sample)]
        orient = 'descending'
        if sample.size >= 2:
            orient = 'ascending' if sample[0] <= sample[-1] else 'descending'
            
    orientation_cache[frame_path] = orient
    return orient


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


def interp_to_pressure(
    field: np.ndarray,
    pressure: np.ndarray,
    level_hpa: float,
    frame_path: str,
    orientation_cache: dict[str, str],
) -> np.ndarray:
    orient = ensure_pressure_orientation(frame_path, pressure, orientation_cache)
    if orient == 'ascending':
        # Normalize to surface-first descending order for interpolation.
        pressure = pressure[::-1, :, :]
        field = field[::-1, :, :]
        
    if field.shape != pressure.shape:
        min_dims = tuple(min(fs, ps) for fs, ps in zip(field.shape, pressure.shape))
        field = field[tuple(slice(0, m) for m in min_dims)]
        pressure = pressure[tuple(slice(0, m) for m in min_dims)]
    
    pressure_hpa = np.ascontiguousarray(pressure, dtype=float32) / 100.0
    field = np.ascontiguousarray(field, dtype=float32)
    
    with np.errstate(invalid='ignore'):
        interp = interplevel(field, pressure_hpa, level_hpa, meta=False)
    
    # Mask columns that never reach the requested pressure level (terrain above level
    # or truncated model tops). Use finite-aware extrema to avoid propagating NaNs.
    level_pa = level_hpa * 100.0
    col_max = np.nanmax(pressure, axis=0)
    col_min = np.nanmin(pressure, axis=0)
    valid = (col_max >= level_pa) & (col_min <= level_pa)
    interp = np.where(valid, interp, np.nan)
    
    return np.asarray(interp, dtype=float32)
    
    
def dbz_to_rate_inhr(dbz: np.ndarray) -> np.ndarray:
    '''Approximate precipitation rate (in/hr) from reflectivity (dBZ).'''
    
    dbz = np.asarray(dbz, dtype=float32)
    with np.errstate(over='ignore'):
        z_lin = np.power(10.0, dbz * 0.1)
    z_lin = np.clip(z_lin, 0.0, None)
    rain_rate_mmhr = np.power(z_lin / 200.0, 1.0 / 1.6, dtype=float32)
    return rain_rate_mmhr / 25.4


def snowfall_support(temp_c: np.ndarray, ptype_field: np.ndarray) -> np.ndarray:
    '''
    Estimate how much falling precipitation contributes to snowfall accumlation.
    
    Parameters
    ----------
    temp_c: np.ndarray
        3-D temperature profile (K or °C) on mass levels in Celsius.
    ptype_field: np.ndarray
        2-D precipitation type field where the integer portion follows the 
        categorical convention: 0=rain, 1=snow, 2=mix, 3=sleet.
    
    Returns
    -------
    np.ndarray
        Fraction [0-1] describing how much liquid precipitation accumlation as
        snow given the vertical thermal structure. Values account for a
        near-surface melting window (32-36 °F) so accumlation smoothly ramps
        down as temperature approach melting.
    '''
        
    temps = np.asarray(temp_c, dtype=float32)
    ptype = np.asarray(ptype_field, dtype=float32)
    
    # Align horizontal dimensions.
    surf_temp_f = (temps[0, :, :] * 9.0 / 5.0) + 32.0
    ny = min(surf_temp_f.shape[0], ptype.shape[0])
    nx = min(surf_temp_f.shape[1], ptype.shape[1])
    surf_temp_f = surf_temp_f[:ny, :nx]
    ptype = ptype[:ny, :nx]
    temps = temps[:ny, :nx]
    
    # Warm nose aloft can still erode flakes; weight by coldest column signal.
    max_temp_c = np.nanmax(temps, axis=0)
    melt_penalty = np.clip((1.5 - max_temp_c) / 1.5, 0.0, 1.0)
    
    # Surface temperature taper between 32-40 F.
    surface_weight = np.clip((40.0 - surf_temp_f) / 4.0, 0.0, 1.0)
    
    valid_ptype = np.isfinite(ptype)
    base_class = np.zeros_like(ptype, dtype=np.int8)
    base_class[valid_ptype] = np.floor(ptype[valid_ptype]).astype(np.int8)
    
    type_factor = np.zeros_like(surface_weight, dtype=float32)
    type_factor[base_class == 1] = 1.0 # Snow
    type_factor[base_class == 2] = 0.5 # Mix/Freezing Rain
    type_factor[base_class == 3] = 0.3 # Sleet/Ice pellets contribute a little depth
    
    support = surface_weight * type_factor * melt_penalty
    support[~np.isfinite(ptype)] = 0.0
    return support.astype(float32)


def sounding_temperature_ticks(min_c: float = -50.0, max_c: float = 50.0, step_c: float = 10.0) -> np.ndarray:
    ''' Evenly spaced temperature ticks for Skew-T backgrounds (°C). '''
    
    return np.arange(min_c, max_c + step_c, step_c, dtype=float32)


def sounding_isotherm_temperatures(
    min_c: float = -120.0, max_c: float = 50.0, step_c: float = 10.0
) -> np.ndarray:
    ''' Temperature values used to draw skewed isotherm guidelines (C). '''
    
    return np.arange(min_c, max_c + step_c, step_c, dtype=float32)


def sounding_temperature_bounds() -> tuple[float, float]:
    ''' X-axis limits for Skew-T temperature (°C). '''
    
    temps = sounding_temperature_ticks()
    return float(temps.min(initial=-50.0)), float(temps.max(initial=50.0))


def sounding_pressure_levels() -> np.ndarray:
    ''' Major pressure levels for Skew-T background (hPa). '''
    
    return np.array([1050.0, 1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0], dtype=float32)


def sounding_pressure_bounds() -> tuple[float, float]:
    ''' Y-axis limits for Skew-T pressure (hPa). '''
    
    levels = sounding_pressure_levels()
    return float(levels.max(initial=1050.0)), float(levels.min(initial=100.0))


def sounding_skewed_isotherm(
    temp_c: np.ndarray | float,
    pressures: np.ndarray,
    angle_deg: float = 45.0,
    aspect_correction: float = 1.0,
) -> np.ndarray:
    '''Return x-values for a skewed isotherm across the provided pressures.

    ``temp_c`` may be a scalar or an array matching ``pressures``. The returned
    values maintain the same shape so that profile points and background
    guidelines share the identical skew mapping.

    ``aspect_correction`` is a multiplicative factor to account for the current
    axes pixel aspect (height / width). This keeps the drawn isotherms visible
    even when the plot is wider than it is tall, preventing the warm lines from
    running off the chart.
    '''
    
    temp_min, temp_max = sounding_temperature_bounds()
    bottom, top = sounding_pressure_bounds()
    
    # When the y-axis uses logarithmic pressure, the apparent spacing between
    # levels follows log(p) rather than a linear delta in hPa. Base the skew on
    # the log-distance from the bottom pressure so the guideline remains a
    # straight line in plot space instead of bowing inward.
    bottom_log = np.log(bottom)
    top_log = np.log(top)
    span_log = bottom_log - top_log
    if span_log == 0:
        return np.full_like(pressures, temp_c, dtype=float32)
    
    skew_per_logp = np.tan(np.deg2rad(angle_deg)) * aspect_correction
    scale = (temp_max - temp_min) / span_log
    offsets = (bottom_log - np.log(pressures)) * skew_per_logp * scale
    return np.asarray(temp_c + offsets, dtype=float32)