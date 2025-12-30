from __future__ import annotations

import math
import numpy as np
from matplotlib.transforms import Affine2D
from netCDF4 import Dataset
from numpy import float32
from wrf import interplevel, rh, to_np

# Thermodynamic constants
RD = 287.05 # J/(kg*K)
CP = 1004.0 # J/(kg*K)
P0 = 100000.0 # Pa
G0 = 9.80665 # m/s^2
LV = 2.5e6 # J/kg
EPSILON = 0.622

PTYPE_INTENSITY_SPAN = 0.995
PTYPE_MAX_RATE_INHR = 5.0

PTYPE_RATE_BREAKS_INHR = np.array(
    [0.0, 0.01, 0.05, 0.25, 0.5, 1.0, 2.5, PTYPE_MAX_RATE_INHR], dtype=float32
)


def _saturation_water_pressure_pa(temp_c: float | np.ndarray) -> np.ndarray:
    ''' Saturation vapor pressure over liquid water (PA) via Tetens formula. '''
    
    temp_c = np.asarray(temp_c, dtype=float32)
    return 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5), dtype=float32)


def saturation_mixing_ratio(pressure_pa: float | np.ndarray, temp_k: float | np.ndarray) -> np.ndarray:
    ''' Saturation mixing ratio (kg/kg) given pressure (Pa) and temperature (K). '''
    
    pressure_pa = np.asarray(pressure_pa, dtype=float32)
    temp_k = np.asarray(temp_k, dtype=float32)
    temp_c = temp_k - 273.15
    es = _saturation_water_pressure_pa(temp_c)
    denom = np.clip(pressure_pa - es, 1e-6, None)
    return EPSILON * es / denom


def _mixing_ratio_from_dewpoint(pressure_hpa: float | np.ndarray, dewpoint_c: float | np.ndarray) -> np.ndarray:
    ''' Mixing ratio (kg/kg) from pressure (hPa) and dewpoint (C).'''
    
    pressure_pa = np.asarray(pressure_hpa, dtype=float32) * 100.0
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    es = _saturation_water_pressure_pa(dew_c)
    denom = np.clip(pressure_pa - es, 1e-6, None)
    return EPSILON * es / denom


def virtual_temperature(temp_k: float | np.ndarray, mixing_ratio: float | np.ndarray) -> np.ndarray:
    ''' Virtual temperature (K) from temperature (K) and mixing ratio (kg/kg). '''
    
    temp_k = np.asarray(temp_k, dtype=float32)
    r = np.asarray(mixing_ratio, dtype=float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        q = r / (1.0 + r)
        return temp_k * (1.0 + 0.61 * q)


def virtual_temperature_profile(
    pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray
) -> np.ndarray:
    ''' Virtual temperature profile (C) using the ambient dewpoint-derived mixing ratio. '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c)
    if valid.sum() < 2:
        return np.full_like(pres, np.nan, dtype=float32)
    
    mixing_ratio = _mixing_ratio_from_dewpoint(pres[valid], dew_c[valid])
    temp_k = temp_c[valid] + 273.15
    vt_k = virtual_temperature(temp_k, mixing_ratio)
    
    vt_c = np.full_like(pres, np.nan, dtype=float32)
    vt_c[valid] = vt_k - 273.15
    return vt_c


def _enforce_monotonic_height(height_m: np.ndarray) -> np.ndarray:
    '''
    Return a version of ``height_m`` that is monotonically increasing along the first axis.
    
    Small non-physical inversions occasionally appear near the surface over
    complex terrain and can collapse layer depths to ~0 m, which zeros out the
    buoyancy integral. This helper nudges each level upward by the smallest
    amount needed to keep the profile strictly increasing without materially
    altering its shape.
    '''
    
    hgt = np.asarray(height_m, dtype=float32)
    if hgt.ndim == 1:
        if hgt.ndim == 0:
            return hgt
        fixed = np.array(hgt, dtype=float32)
        for idx in range(1, fixed.size):
            if fixed[idx] <= fixed[idx - 1]:
                fixed[idx] = fixed[idx - 1] + 1e-3
        return fixed
    
    fixed = np.array(hgt, dtype=float32)
    nz = fixed.shape[0]
    for lvl in range(1, nz):
        prev = fixed[lvl - 1]
        cur = fixed[lvl]
        mask = cur <= prev
        if np.any(mask):
            fixed[lvl, mask] = prev[mask] + 1e-3
    return fixed


def _integrate_positive_buoyancy(height_m: np.ndarray, buoyancy: np.ndarray) -> float:
    '''
    Integrate positive buoyancy (J/kg) using zero-crossing interpolation.
    
    The integration assumes the inputs are 1-D profiles with matching shapes.
    '''
    
    hgt = np.asarray(height_m, dtype=float32)
    b = np.asarray(buoyancy, dtype=float32)
    valid = np.isfinite(hgt) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    
    hgt = hgt[valid]
    b = b[valid]
    
    order = np.argsort(hgt)
    hgt = _enforce_monotonic_height(hgt[order])
    b = b[order]
    
    energy = 0.0
    for idx in range(b.size - 1):
        b0 = b[idx]
        b1 = b[idx + 1]
        h0 = hgt[idx]
        h1 = hgt[idx + 1]
        
        if b0 >= 0.0 and b1 >= 0.0:
            energy += 0.5 * (b0 + b1) * (h1 - h0)
        elif b0 >= 0.0 > b1:
            frac = b0 / np.clip(b0 - b1, 1e-6, None)
            h_cross = h0 + frac * (h1 - h0)
            energy += 0.5 * b0 * (h_cross - h0)
        elif b0 < 0.0 <= b1:
            frac = b1 / np.clip(b1 - b0, 1e-6, None)
            h_cross = h0 + frac * (h1 - h0)
            energy += 0.5 * b1 * (h1 - h_cross)
    
    return float(np.clip(energy, 0.0, None))

def _surface_based_cape_profile(
    pressure_hpa: np.ndarray,
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
    height_m: np.ndarray,
    *,
    presorted: bool = False,
) -> float:
    '''
    Surface-based CAPE (J/kg) for a single column using virtual temperature buoyancy.
    
    The ``presorted`` flag allows callers that already enforce a surface-to-top
    ordering to bypass an otherwise expensive per-column argsort.
    '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    hgt = np.asarray(height_m, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c) & np.isfinite(hgt)
    if valid.sum() < 3:
        return np.nan
    
    pres = pres[valid]
    temp_c = temp_c[valid]
    dew_c = dew_c[valid]
    hgt = hgt[valid]
    
    temp = temp_c
    dew = dew_c
    
    if not presorted:
        order = np.argsort(pres)[::-1]
        pres = pres[order]
        temp_c = temp_c[order]
        dew_c = dew_c[order]
        hgt = hgt[order]
    elif pres[0] < pres[-1]:
        # Ensure we integrate from the surface upward even if a caller claimed the
        # data were presorted but provided a top-down profile.
        pres = pres[::-1]
        temp_c = temp_c[::-1]
        dew_c = dew_c[::-1]
        hgt = hgt[::-1]
    
    vt_env_c = virtual_temperature_profile(pres, temp_c, dew_c)
    vt_parcel_c = parcel_trace_temperature_profile(pres, temp_c, dew_c)
    
    valid_vt = np.isfinite(vt_env_c) & np.isfinite(vt_parcel_c) & np.isfinite(hgt)
    if valid_vt.sum() < 2:
        return np.nan
    
    pres = pres[valid_vt]
    vt_env_k = vt_env_c[valid_vt] + 273.15
    vt_parcel_k = vt_parcel_c[valid_vt] + 273.15
    hgt = _enforce_monotonic_height(hgt[valid_vt])
    
    sort_h = np.argsort(hgt)
    hgt = hgt[sort_h]
    vt_env_k = vt_env_k[sort_h]
    vt_parcel_k = vt_parcel_k[sort_h]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        buoyancy = G0 * (vt_parcel_k - vt_env_k) / np.clip(vt_env_k, 1e-6, None)
    return _integrate_positive_buoyancy(hgt, buoyancy)


def surface_based_cape_from_profile(
    pressure_hpa: np.ndarray,
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
    height_m: np.ndarray,
) -> float:
    '''
    Convenience wrapper around the single-column SBCAPE calculator.
    '''
    
    return _surface_based_cape_profile(pressure_hpa, temperature_c, dewpoint_c, height_m)


def surface_based_cape(
    pressure_pa: np.ndarray,
    temperature_k: np.ndarray,
    rh_percent: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    '''
    Compute surface_based CAPE (J/kg) for every gridpoint.
    '''
    
    pressure = np.asarray(pressure_pa, dtype=float32)
    temp_k = np.asarray(temperature_k, dtype=float32)
    rh = np.asarray(rh_percent, dtype=float32)
    height = np.asarray(height_m, dtype=float32)
    
    nz = min(pressure.shape[0], temp_k.shape[0], rh.shape[0], height.shape[0])
    ny = min(pressure.shape[1], temp_k.shape[1], rh.shape[1], height.shape[1])
    nx = min(pressure.shape[2], temp_k.shape[2], rh.shape[2], height.shape[2])
    if nz < 3 or ny == 0 or nx == 0:
        return np.full((ny, nx), np.nan, dtype=float32)
    
    slicer = (slice(0, nz), slice(0, ny), slice(0, nx))
    pressure = pressure[slicer]
    temp_k = temp_k[slicer]
    rh = rh[slicer]
    height = height[slicer]
    
    temp_c = temp_k - 273.15
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = np.log(np.clip(rh, 1e-6, 100.0) * 0.01) + (17.67 * temp_c) / (temp_c + 243.5)
        dewpoint_c = (243.5 * gamma) / np.clip(17.67 - gamma, 1e-6, None)
    
    # Normalize orientation to surface-first ordering so the vectorized ascent
    # sweep runs monotonically upward.
    surf_med = np.nanmedian(pressure[0, :, :])
    top_med = np.nanmedian(pressure[-1, :, :])
    if np.isfinite(surf_med) and np.isfinite(top_med) and surf_med < top_med:
        pressure = pressure[::-1, :, :]
        temp_k = temp_k[::-1, :, :]
        temp_c = temp_c[::-1, :, :]
        dewpoint_c = dewpoint_c[::-1, :, :]
        height = height[::-1, :, :]
    
    pressure_hpa = pressure / 100.0
    
    # Flatten horizontal dimensions so the moist-adiabatic ascent loops only
    # over vertical levels instead of every grid cell.
    ncol = ny * nx
    pres_flat = pressure_hpa.reshape(nz, ncol)
    temp_flat = temp_k.reshape(nz, ncol)
    dew_flat = dewpoint_c.reshape(nz, ncol)
    hgt_flat = height.reshape(nz, ncol)
    
    cape_flat = _surface_based_cape_profiles_vectorized(pres_flat, temp_flat, dew_flat, hgt_flat)
    return cape_flat.reshape(ny, nx)


def lcl_temperature_pressure(
    pressure_hpa: float, temperature_c: float, dewpoint_c: float
) -> tuple[float, float]:
    ''' Return LCL temperature (C) and pressure (hPa) via Bolton (1980).'''
    
    temp_k = np.asarray(temperature_c, dtype=float32) + 273.15
    dewpoint_k = np.asarray(dewpoint_c, dtype=float32) + 273.15
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        tlcl_k = 1.0 / ((1.0 / (dewpoint_k - 56.0)) + (np.log(temp_k / dewpoint_k) / 800.0)) + 56.0
    tlcl_k = np.clip(tlcl_k, 0.0, None)
    plcl_hpa = np.asarray(pressure_hpa, dtype=float32) * np.power(tlcl_k / temp_k, CP / RD)
    return tlcl_k - 273.15, plcl_hpa


def _moist_ascent_step(prev_p_hpa: np.ndarray, prev_temp_k: np.ndarray, target_p_hpa: np.ndarray) -> np.ndarray:
    '''
    Single hydrostatic moist-adiabatic step from ``prev_p_hpa`` to ``target_p_hpa``.

    The calculation is vectorized across columns so a single call updates all
    active profiles at the current level instead of iterating over gridpoints.
    '''
    
    p_mid = 0.5 * (prev_p_hpa + target_p_hpa)
    temp_mid = prev_temp_k
    r_s = saturation_mixing_ratio(p_mid * 100.0, temp_mid)
    
    gamma_m = G0 * (1.0 + (LV * r_s) / (RD * temp_mid))
    gamma_m /= CP + ((LV ** 2) * r_s * EPSILON) / (RD * (temp_mid ** 2))
    
    dp_pa = (target_p_hpa - prev_p_hpa) * 100.0
    dz = -RD * temp_mid * dp_pa / (p_mid * 100.0 * G0)
    return temp_mid - gamma_m * dz


def parcel_trace_temperature_profile(
    pressure_hpa: np.ndarray,
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
    *,
    start_pressure_hpa: float | None = None,
    start_temperature_c: float | None = None,
    start_dewpoint_c: float | None = None,
) -> np.ndarray:
    ''' Parcel virtual temperature (C) along the provided pressure levels. '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c)
    if valid.sum() < 2:
        return np.full_like(pres, np.nan, dtype=float32)
    
    order = np.argsort(pres)[::-1]
    pres_sorted = pres[order]
    temp_sorted = temp_c[order]
    dew_sorted = dew_c[order]
    n_levels = pres_sorted.size
    
    start_idx = 0 if start_pressure_hpa is None else int(np.nanargmin(np.abs(pres_sorted - start_pressure_hpa)))
    start_p = float(start_pressure_hpa) if start_pressure_hpa is not None else float(pres_sorted[start_idx])
    start_temp_c = float(start_temperature_c) if start_temperature_c is not None else float(temp_sorted[start_idx])
    start_dew_c = float(start_dewpoint_c) if start_dewpoint_c is not None else float(dew_sorted[start_idx])
    
    pres_start = pres_sorted[start_idx:]
    
    tlcl_c, plcl_hpa = lcl_temperature_pressure(start_p, start_temp_c, start_dew_c)
    start_temp_k = start_temp_c + 273.15
    theta = start_temp_k * np.power(1000.0 / start_p, RD / CP)
    
    # Parcel total water content remains constant below the LCL. Above the LCL
    # the parcel follows a saturated mixing ratio determined by its temperature
    # and pressure.
    surf_es = _saturation_water_pressure_pa(start_dew_c)
    surf_r = EPSILON * surf_es / np.clip(start_p * 100.0 - surf_es, 1e-6, None)
    
    parcel_temps_k = np.full_like(pres_start, np.nan, dtype=float32)
    dry_mask = pres_start >= plcl_hpa
    
    # Dry-adiabatic ascent from the start level to the LCL.
    if dry_mask.any():
        parcel_temps_k[dry_mask] = theta * np.power(pres_start[dry_mask] / 1000.0, RD / CP)
    
    # Moist-adiabatic ascent above the LCL, stepping sequentially so curvature is preserved.
    if (~dry_mask).any():
        lcl_temp_k = theta * np.power(plcl_hpa / 1000.0, RD / CP)
        prev_p = plcl_hpa
        prev_temp_k = lcl_temp_k
        for idx in np.where(~dry_mask)[0]:
            p_level = float(pres_start[idx])
            temp_k = _moist_ascent_step(np.array(prev_p, dtype=float32), np.array(prev_temp_k, dtype=float32), np.array(p_level, dtype=float32))
            parcel_temps_k[idx] = temp_k
            prev_p = p_level
            prev_temp_k = float(temp_k)
    
    # Convert parcel temperature to virtual temperature using the appropriate
    # mixing ratio profile.
    mixing_ratio = np.full_like(parcel_temps_k, np.nan, dtype=float32)
    
    if dry_mask.any():
        mixing_ratio[dry_mask] = surf_r
    
    if (~dry_mask).any():
        for idx in np.where(~dry_mask)[0]:
            p_level = float(pres_start[idx])
            temp_k = float(parcel_temps_k[idx])
            mixing_ratio[idx] = saturation_mixing_ratio(p_level * 100.0, temp_k)
    
    parcel_virtual_k = virtual_temperature(parcel_temps_k, mixing_ratio)
    parcel_virtual_c = parcel_virtual_k - 273.15
    full_profile = np.full(n_levels, np.nan, dtype=float32)
    full_profile[start_idx:] = parcel_virtual_c
    
    inv_order = np.argsort(order)
    return full_profile[inv_order]


def _surface_based_cape_profiles_vectorized(
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    dewpoint_c: np.ndarray,
    height_m: np.ndarray,
) -> np.ndarray:
    '''
    Vectorized SBCAPE across all grid columns using a single vertical sweep.

    Parameters expect ``pressure_hpa``, ``temperature_k``, ``dewpoint_c``, and
    ``height_m`` with shape (nz, ncol) ordered surface-to-top.
    '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_k = np.asarray(temperature_k, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    hgt = np.asarray(height_m, dtype=float32)
    
    nz, ncol = pres.shape
    column_valid = np.isfinite(pres[0]) & np.isfinite(temp_k[0]) & np.isfinite(dew_c[0]) & np.isfinite(hgt[0])
    sufficient_levels = np.isfinite(pres) & np.isfinite(temp_k) & np.isfinite(dew_c) & np.isfinite(hgt)
    column_valid &= np.sum(sufficient_levels, axis=0) >= 3
    if not column_valid.any():
        return np.full(ncol, np.nan, dtype=float32)
    
    surf_p = pres[0, column_valid]
    surf_temp_c = temp_k[0, column_valid] - 273.15
    surf_dew_c = dew_c[0, column_valid]
    
    _, plcl_hpa = lcl_temperature_pressure(surf_p, surf_temp_c, surf_dew_c)
    theta = (temp_k[0, column_valid]) * np.power(1000.0 / surf_p, RD / CP)
    surf_es = _saturation_water_pressure_pa(surf_dew_c)
    surf_r = EPSILON * surf_es / np.clip(surf_p * 100.0 - surf_es, 1e-6, None)
    
    dry_profile = theta[None, :] * np.power(pres[:, column_valid] / 1000.0, RD / CP)
    dry_mask = pres[:, column_valid] >= plcl_hpa
    
    parcel_temps_k = np.full_like(pres[:, column_valid], np.nan, dtype=float32)
    parcel_temps_k[dry_mask] = dry_profile[dry_mask]
    
    lcl_temp_k = theta * np.power(plcl_hpa / 1000.0, RD / CP)
    prev_temp = lcl_temp_k.astype(float32)
    prev_p = plcl_hpa.astype(float32)
    
    for level in range(nz):
        p_level = pres[level, column_valid]
        active = p_level < plcl_hpa
        if not np.any(active):
            continue
        next_temp = _moist_ascent_step(prev_p[active], prev_temp[active], p_level[active])
        parcel_temps_k[level, active] = next_temp
        prev_temp[active] = next_temp
        prev_p[active] = p_level[active]
    
    mixing_ratio = np.full_like(parcel_temps_k, np.nan, dtype=float32)
    if dry_mask.any():
        mixing_ratio[dry_mask] = np.broadcast_to(surf_r, dry_mask.shape)[dry_mask]
    
    sat_mr = saturation_mixing_ratio(pres[:, column_valid] * 100.0, parcel_temps_k)
    moist_mask = ~dry_mask & np.isfinite(parcel_temps_k)
    if moist_mask.any():
        mixing_ratio[moist_mask] = sat_mr[moist_mask]
    
    parcel_virtual_k = virtual_temperature(parcel_temps_k, mixing_ratio)
    env_mixing_ratio = _mixing_ratio_from_dewpoint(pres[:, column_valid], dew_c[:, column_valid])
    env_virtual_k = virtual_temperature(temp_k[:, column_valid], env_mixing_ratio)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        buoyancy = G0 * (parcel_virtual_k - env_virtual_k) / np.clip(env_virtual_k, 1e-6, None)
    
    hgt_col = _enforce_monotonic_height(hgt[:, column_valid])
    # Fast vectorized integration of positive buoyancy. Using trapz on clipped
    # buoyancy avoids a Python loop over every grid column and dramatically
    # reduces UI latency when the user requests SBCAPE fields.
    cape = np.trapz(np.clip(buoyancy, 0.0, None), hgt_col, axis=0)
    cape = np.clip(cape, 0.0, None)
    
    result = np.full(ncol, np.nan, dtype=float32)
    result[column_valid] = cape
    return result


def _dewpoint_from_mixing_ratio(mixing_ratio: float, pressure_hpa: float) -> float:
    ''' Dewpoint (°C) from mixing ratio (kg/kg) and pressure (hPa). '''
    
    pressure_pa = pressure_hpa * 100.0
    es = (mixing_ratio * pressure_pa) / np.clip(EPSILON + mixing_ratio, 1e-6, None)
    es = np.clip(es, 1e-6, None)
    log_term = np.log(es / 611.2)
    return float((243.5 * log_term) / np.clip(17.67 - log_term, 1e-6, None))


def _equivalent_potential_temperature(
    pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray
) -> np.ndarray:
    temp_k = np.asarray(temperature_c, dtype=float32) + 273.15
    pres = np.asarray(pressure_hpa, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    r = _mixing_ratio_from_dewpoint(pres, dew_c)
    tlcl_c, _ = lcl_temperature_pressure(pres, temperature_c, dew_c)
    tlcl_k = tlcl_c + 273.15
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        theta_l = temp_k * np.power(1000.0 / pres, (RD / CP) * (1.0 - 0.28 * r))
        exp_term = np.exp(((3036.0 / tlcl_k) - 1.78) * r * (1.0 + 0.448 * r))
        return theta_l * exp_term


def most_unstable_parcel_source(
    pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray, depth_hpa: float = 300.0
) -> tuple[float, float, float]:
    '''
    Return (pressure, temperature, dewpoint) for the most unstable parcel within ``depth_hpa`` of the surface.
    '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c)
    if valid.sum() < 1:
        return np.nan, np.nan, np.nan
    
    order = np.argsort(pres)[::-1]
    pres_sorted = pres[order]
    temp_sorted = temp_c[order]
    dew_sorted = dew_c[order]
    
    surface_p = pres_sorted[0]
    layer_mask = pres_sorted >= (surface_p - depth_hpa)
    theta_e = _equivalent_potential_temperature(
        pres_sorted[layer_mask], temp_sorted[layer_mask], dew_sorted[layer_mask]
    )
    if not np.isfinite(theta_e).any():
        return np.nan, np.nan, np.nan
    
    idx = int(np.nanargmax(theta_e))
    start_p = float(pres_sorted[layer_mask][idx])
    start_temp = float(temp_sorted[layer_mask][idx])
    start_dew = float(dew_sorted[layer_mask][idx])
    return start_p, start_temp, start_dew


def mixed_layer_parcel_source(
    pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray, depth_hpa: float = 100.0
) -> tuple[float, float, float]:
    '''
    Return (pressure, temperature, dewpoint) for a mixed-layer parcel averaged over ``depth_hpa``.
    '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c)
    if valid.sum() < 1:
        return np.nan, np.nan, np.nan
    
    order = np.argsort(pres)[::-1]
    pres_sorted = pres[order]
    temp_sorted = temp_c[order]
    dew_sorted = dew_c[order]
    
    surface_p = pres_sorted[0]
    layer_mask = pres_sorted >= (surface_p - depth_hpa)
    if not layer_mask.any():
        return np.nan, np.nan, np.nan
    
    mean_temp = float(np.nanmean(temp_sorted[layer_mask]))
    mean_pres = float(np.nanmean(pres_sorted[layer_mask]))
    layer_mr = _mixing_ratio_from_dewpoint(pres_sorted[layer_mask], dew_sorted[layer_mask])
    mean_mr = float(np.nanmean(layer_mr))
    mean_dew = _dewpoint_from_mixing_ratio(mean_mr, mean_pres)
    return mean_pres, mean_temp, mean_dew


def parcel_thermo_indices_from_profile(
    pressure_hpa: np.ndarray,
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
    height_m: np.ndarray,
    *,
    start_pressure_hpa: float | None = None,
    start_temperature_c: float | None = None,
    start_dewpoint_c: float | None = None,
    presorted: bool = False,
) -> tuple[float, float, float, float, float, float]:
    '''
    Thermodynamic indices for a parcel starting at the specified thermodynamic point.
    
    Returns at tuple of (CAPE, CINH, LCL_height_m, LFC_height_m, EL_height_m, LiftedIndex_C).
    Heights are above ground level and the Lifted Index is the 500-hPa ambient
    temperature minus the parcel temperature at 500 hPa.
    '''
    
    pres = np.asarray(pressure_hpa, dtype=float32)
    temp_c = np.asarray(temperature_c, dtype=float32)
    dew_c = np.asarray(dewpoint_c, dtype=float32)
    hgt = np.asarray(height_m, dtype=float32)
    
    valid = np.isfinite(pres) & np.isfinite(temp_c) & np.isfinite(dew_c) & np.isfinite(hgt)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    pres = pres[valid]
    temp_c = temp_c[valid]
    dew_c = dew_c[valid]
    hgt = hgt[valid]
    
    if not presorted:
        order_start = np.argsort(pres)[::-1]
        pres = pres[order_start]
        temp = temp_c[order_start]
        dew = dew_c[order_start]
        hgt = hgt[order_start]
    elif pres[0] < pres[-1]:
        pres = pres[::-1]
        temp = temp_c[::-1]
        dew = dew_c[::-1]
        hgt = hgt[::-1]
    else:
        order_start = np.argsort(pres)[::-1]
        pres = pres[order_start]
        temp = temp_c[order_start]
        dew = dew_c[order_start]
        hgt = hgt[order_start]
    
    start_idx = 0 if start_pressure_hpa is None else int(np.nanargmin(np.abs(pres - start_pressure_hpa)))
    start_p = float(start_pressure_hpa) if start_pressure_hpa is not None else float(pres[start_idx])
    start_temp_c = float(start_temperature_c) if start_temperature_c is not None else float(temp[start_idx])
    start_dew_c = float(start_dewpoint_c) if start_dewpoint_c is not None else float(dew[start_idx])
    _, plcl_hpa = lcl_temperature_pressure(start_p, start_temp_c, start_dew_c)
    if np.isfinite(plcl_hpa):
        # The parcel cannot condense at a pressure higher than its start level.
        # Clamp to the starting pressure to keep LCL interpolation within the
        # observed profile instead of falling outside the pressure range and
        # returning NaN heights for buoyant parcels.
        plcl_hpa = float(np.clip(plcl_hpa, pres.min(), start_p))
    
    vt_env_c = virtual_temperature_profile(pres, temp, dew)
    vt_parcel_c = parcel_trace_temperature_profile(
        pres,
        temp,
        dew,
        start_pressure_hpa=start_pressure_hpa,
        start_temperature_c=start_temperature_c,
        start_dewpoint_c=start_dewpoint_c,
    )
    
    # Normalize profiles to surface-first order before rebuilding heights so
    # parcels starting aloft keep the correct AGL reference instead of being
    # re-anchored to 0 m at their start level.
    order = np.argsort(pres)[::-1]
    pres = pres[order]
    temp = temp[order]
    dew = dew[order]
    hgt = hgt[order]
    vt_env_c = vt_env_c[order]
    vt_parcel_c = vt_parcel_c[order]
    
    vt_env_k = vt_env_c + 273.15
    vt_parcel_k = vt_parcel_c + 273.15
    
    # Rebuild a smooth, monotonic AGL height profile directly from the
    # hypsometric equation so equilibrium-level searches use physically
    # consistent layer depths even when raw model heights contain noise or
    # terrain artifacts. Anchoring at 0 m avoids negative heights while still
    # preserving realistic depth between pressure levels.
    hgt_agl = np.zeros_like(pres, dtype=float32)
    for idx in range(1, pres.size):
        tv_bar = 0.5 * (vt_env_k[idx - 1] + vt_env_k[idx])
        dp_ratio = np.log(np.clip(pres[idx - 1] / pres[idx], 1e-6, None))
        hgt_agl[idx] = hgt_agl[idx - 1] + (RD / G0) * tv_bar * dp_ratio
    
    hgt_agl = _enforce_monotonic_height(hgt_agl)
    
    valid_vt = np.isfinite(vt_env_k) & np.isfinite(vt_parcel_k) & np.isfinite(hgt_agl)
    if valid_vt.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    pres = pres[valid_vt]
    vt_env_k = vt_env_k[valid_vt]
    vt_parcel_k = vt_parcel_k[valid_vt]
    vt_parcel_c = vt_parcel_c[valid_vt]
    hgt = hgt_agl[valid_vt]
    temp = temp[valid_vt]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        buoyancy = G0 * (vt_parcel_k - vt_env_k) / np.clip(vt_env_k, 1e-6, None)
    
    buoyancy = np.asarray(buoyancy, dtype=float32)
    if not np.isfinite(buoyancy).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    def _zero_cross_height(h0: float, h1: float, b0: float, b1: float) -> float:
        denom = b1 - b0
        if abs(denom) < 1e-6:
            denom = math.copysign(1e-6, denom if denom != 0 else 1.0)
        frac = -b0 / denom
        return h0 + frac * (h1 - h0)
    
    # Estimate heights for key parcel levels using pressure-to-height interpolation.
    lcl_height = np.nan
    if np.isfinite(plcl_hpa) and pres.min() <= plcl_hpa <= pres.max():
        lcl_height = float(np.interp(plcl_hpa, pres[::-1], hgt[::-1]))
    
    lfc_search_idx = 0
    if np.isfinite(lcl_height):
        lfc_search_idx = int(np.searchsorted(hgt, lcl_height, side='left'))
    
    # Locate the first level of positive buoyancy (LFC) to bound CINH to the
    # layer the parcel must lift through. If no positive buoyancy exists, CAPE
    # is zero and CINH is simply the integrated negative buoyancy of the
    # profile.
    positive_idx = np.where((np.arange(buoyancy.size) >= lfc_search_idx) & (buoyancy > 0.0))[0]
    if positive_idx.size == 0:
        # No positive buoyancy means the parcel never becomes unstable. Treat the
        # profile as zero CAPE/zero CINH instead of letting deep cold layers
        # integrate large negative buoyancy.
        return 0.0, 0.0, np.nan, np.nan, np.nan, np.nan
    
    lfc_idx = int(positive_idx[0])
    lfc_height = hgt[lfc_idx]
    lfc_buoy = buoyancy[lfc_idx]
    if lfc_idx > 0 and buoyancy[lfc_idx - 1] <= 0.0:
        lfc_height = _zero_cross_height(
            hgt[lfc_idx - 1], hgt[lfc_idx], buoyancy[lfc_idx - 1], buoyancy[lfc_idx]
        )
        lfc_buoy = 0.0
    
    if np.isfinite(lcl_height):
        if lcl_height > lfc_height:
            lfc_buoy = 0.0
        lfc_height = max(lfc_height, lcl_height)
    
    # The equilibrium level is the level above the last positive buoyancy where
    # the parcel returns to neutral or negative and does not become unstable
    # again. This guards against choosing a shallow negative blip as the EL when
    # the parcel quickly re-enters a buoyant layer aloft.
    pos_after_lfc = np.where((np.arange(buoyancy.size) >= lfc_idx) & (buoyancy > 0.0))[0]
    last_pos_idx = int(pos_after_lfc[-1])
    
    el_idx = None
    el_height = np.nan
    el_buoy = np.nan
    
    # Prefer the first zero-crossing *after the final* positive layer, which
    # better matches the conceptual "top" of the final buoyant plume when the
    # profile briefly dips negative and recovers aloft. Starting the search at
    # ``last_pos_idx`` avoids prematurely selecting a lower negative excursion
    # when buoyancy becomes positive again above it.
    for idx in range(last_pos_idx, buoyancy.size - 1):
        b0 = buoyancy[idx]
        b1 = buoyancy[idx + 1]
        if b0 > 0.0 and b1 <= 0.0:
            el_height = _zero_cross_height(hgt[idx], hgt[idx + 1], b0, b1)
            el_buoy = 0.0
            el_idx = idx + 1
            break
    
    if el_idx is None:
        el_candidates = np.where((np.arange(buoyancy.size) > last_pos_idx) & (buoyancy <= 0.0))[0]
        if el_candidates.size:
            el_idx = int(el_candidates[0])
            el_height = hgt[el_idx]
            el_buoy = buoyancy[el_idx]
        else:
            el_idx = buoyancy.size - 1
            el_height = hgt[-1]
            el_buoy = buoyancy[-1]
    
    cinh_h = np.concatenate([hgt[:lfc_idx], [lfc_height]])
    cinh_b = np.concatenate([buoyancy[:lfc_idx], [0.0]])
    cinh = np.trapz(np.clip(cinh_b, None, 0.0), cinh_h)
    
    cape_h = np.concatenate([[lfc_height], hgt[lfc_idx + 1:el_idx], [el_height]])
    cape_b = np.concatenate([[lfc_buoy], buoyancy[lfc_idx + 1:el_idx], [el_buoy]])
    cape = _integrate_positive_buoyancy(cape_h, cape_b)
    
    cape = float(np.clip(cape, 0.0, None))
    cinh = float(cinh)
    
    # Lifted index at 500 hPa (ambient temp minus parcel temp).
    li = np.nan
    target_pres = 500.0
    if pres.min() <= target_pres <= pres.max():
        env_temp_500 = float(np.interp(target_pres, pres[::-1], temp[::-1]))
        parcel_temp_profile_c = vt_parcel_k - 273.15
        parcel_temp_500 = float(np.interp(target_pres, pres[::-1], parcel_temp_profile_c[::-1]))
        li = env_temp_500 - parcel_temp_500
    
    lfc_height_final = lfc_height if np.isfinite(lfc_height) else np.nan
    el_height_final = el_height if np.isfinite(el_height) else np.nan
    
    if np.isfinite(el_height_final) and el_height_final < 0.0:
        el_height_final = np.nan
    
    if cape <= 1e-3:
        # When CAPE is effectively zero (e.g., very cold/stable profiles),
        # suppress CINH and hide heights so the display does not accumulate
        # misleading values for stable parcels.
        cape = 0.0
        cinh = 0.0
        return cape, cinh, np.nan, np.nan, np.nan, np.nan
    
    return cape, cinh, lcl_height, lfc_height_final, el_height_final, li
    

def parcel_cape_cinh_from_profile(
    pressure_hpa: np.ndarray,
    temperature_c: np.ndarray,
    dewpoint_c: np.ndarray,
    height_m: np.ndarray,
    *,
    start_pressure_hpa: float | None = None,
    start_temperature_c: float | None = None,
    start_dewpoint_c: float | None = None,
    presorted: bool = False,
) -> tuple[float, float]:
    '''
    CAPE and CINH (J/kg) for a parcel starting at the specified thermodynamic point.
    '''
    
    cape, cin, *_ = parcel_thermo_indices_from_profile(
        pressure_hpa,
        temperature_c,
        dewpoint_c,
        height_m,
        start_pressure_hpa=start_pressure_hpa,
        start_temperature_c=start_temperature_c,
        start_dewpoint_c=start_dewpoint_c,
        presorted=presorted,
    )
    return cape, cin


def ptype_rate_offset(rate: np.ndarray | float) -> np.ndarray | float:
    '''Map precipitation rate (in/hr) to an intensity offset inside the band.'''
    
    rate_arr = np.asarray(rate, dtype=float32)
    break_rates = PTYPE_RATE_BREAKS_INHR
    break_positions = np.linspace(0.0, 1.0, break_rates.size, dtype=float32) * PTYPE_INTENSITY_SPAN
    clamped = np.clip(rate_arr, break_rates[0], break_rates[-1])
    offset = np.interp(clamped, break_rates, break_positions)
    if np.isscalar(rate):
        return float(offset)
    return offset.astype(float32)


def ptype_rate_from_offset(offset: np.ndarray | float) -> np.ndarray | float:
    '''Invert ``ptype_rate_offset`` back to an approximate precipitation rate.'''
    
    offset_arr = np.asarray(offset, dtype=float32)
    break_rates = PTYPE_RATE_BREAKS_INHR
    break_positions = np.linspace(0.0, 1.0, break_rates.size, dtype=float32) * PTYPE_INTENSITY_SPAN
    clamped = np.clip(offset_arr, break_rates[0], break_rates[-1])
    rate = np.interp(clamped, break_positions, break_rates)
    if np.isscalar(offset):
        return float(rate)
    return rate.astype(float32)


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


def uh_minimum_for_spacing(dx_m: float | None, dy_m: float | None) -> float:
    '''
    Resolution-aware minimum updraft helicity threshold (m^s s^-2).
    
    A 3 km grid uses 75 m^s s^-2 while a 9 km grid uses 25 m^s s^-2). The
    threshold scales inversely with grid spacing so coarser domains use a lower
    cutoff. Values default to 75 when grid spacing metadata is unavailable.
    '''
    
    dx_km = np.nan
    if dx_m is not None:
       dx_km = float(dx_m) / 1000.0
    dy_km = np.nan
    if dy_m is not None:
       dy_km = float(dy_m) / 1000.0
    
    spacing_km = np.nanmean([v for v in (dx_km, dy_km) if np.isfinite(v)])
    if not np.isfinite(spacing_km) or spacing_km <= 0.0:
        return 75.0
    
    return float(np.clip(75.0 / spacing_km, 1.0, None))


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


def calc_updraft_helicity(
    nc: Dataset,
    time_index: int,
    *,
    bottom_m: float = 2000.0,
    top_m: float = 5000.0,
) -> np.ndarray:
    '''
    Updraft helicity (m^s s^-2) between ``bottom_m`` amd ``top_m`` AGL.
    
    Uses destaggered U/V/W winds on mass levels with geometric height derived
    from PH/PHB and surface height from HGT. Returns ``Nan`` where inputs are
    insufficient for the intergration.
    '''
    
    required = ['U', 'V', 'W', 'PH', 'PHB', 'HGT']
    missing = [name for name in required if name not in nc.variables]
    if missing:
        raise RuntimeError(f'Missing variables for updraft helicity: {", ".join(missing)}')
    
    u_stag = slice_time_var(nc.variables['U'], time_index)
    v_stag = slice_time_var(nc.variables['V'], time_index)
    w_stag = slice_time_var(nc.variables['W'], time_index)
    
    u = destagger(u_stag, axis=-1).astype(float32)
    v = destagger(v_stag, axis=-2).astype(float32)
    w = destagger(w_stag, axis=0).astype(float32)
    
    height = calc_height(nc, time_index)
    
    hgt_var  = nc.variables['HGT']
    h_dims = tuple(getattr(hgt_var, 'dimensions', ()))
    if 'Time' in h_dims:
        hgt = np.array(hgt_var[time_index, :, :], dtype=float32)
    else:
        hgt = np.array(hgt_var[:, :], dtype=float32)
    
    z_agl = height - hgt[None, :, :]
    
    vert_depths = [u.shape[0], v.shape[0], w.shape[0], z_agl.shape[0]]
    common_levels = min(vert_depths)
    if common_levels < 2:
        return np.full(z_agl[0], np.nan, dtype=float32)
    
    u = u[:common_levels]
    v = v[:common_levels]
    w = w[:common_levels]
    z_agl = z_agl[:common_levels]
    
    dx = getattr(nc, 'DX', np.nan)
    dy = getattr(nc, 'DY', np.nan)
    spacing_dx = float(dx) if np.isfinite(dx) else np.nan
    spacing_dy = float(dy) if np.isfinite(dy) else np.nan
    
    if not np.isfinite(spacing_dx) or spacing_dx <= 0.0:
        spacing_dx = np.nanmean(np.diff(nc.variables['XLONG'][0, 0, :]))
        spacing_dx = spacing_dx * 111320.0 * np.cos(np.deg2rad(np.nanmean(nc.variables['XLAT'][0, :, :]))) if np.isfinite(spacing_dx) else np.nan
    
    if not np.isfinite(spacing_dy) or spacing_dy <= 0.0:
        spacing_dy = np.nanmean(np.diff(nc.variables['XLAT'][0, :, 0]))
        spacing_dy = spacing_dy * 111320.0 if np.isfinite(spacing_dy) else np.nan
    
    if not np.isfinite(spacing_dx) or spacing_dx <= 0.0:
        spacing_dx = 3000.0
    if not np.isfinite(spacing_dy) or spacing_dy <= 0.0:
        spacing_dy = 3000.0
    
    # Relative vertical vorticity on mass levels.
    dvdx = np.gradient(v, spacing_dx, axis=-1)
    dudy = np.gradient(u, spacing_dy, axis=-2)
    rel_vort = dvdx - dudy
    
    # Restrict to the requested layer and intergrate w * zeta through depth.
    layer_mask = (z_agl >= bottom_m) & (z_agl <= top_m)
    integrand = np.where(layer_mask, rel_vort * w, np.nan)
    layer_heights = np.where(layer_mask, z_agl, np.nan)
    
    valid_pairs = np.isfinite(integrand[1:]) & np.isfinite(integrand[:-1])
    valid_pairs &= np.isfinite(layer_heights[1:]) & np.isfinite(layer_mask[:-1])
    
    delta_h = np.where(valid_pairs, layer_heights[1:] - layer_heights[:-1], 0.0)
    avg_integrand = np.where(valid_pairs, 0.5 * (integrand[1:] + integrand[:-1]), 0.0)
    
    uh = np.sum(avg_integrand * delta_h, axis=0)
    uh = np.where(valid_pairs.any(axis=0), uh, np.nan)
    uh = np.asarray(uh, dtype=float32)
    
    return uh


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
    
    # Warm layers aloft erode flakes but allow a modest cushion so light warm noses
    # do not zero out accumulation support too aggressively.
    max_temp_c = np.nanmax(temps, axis=0)
    warm_fraction = np.nanmean(temps > 0.0, axis=0)
    temp_penalty = np.clip((3.0 - np.clip(max_temp_c, 0.0, None)) / 3.0, 0.0, 1.0)
    melt_penalty = np.clip(temp_penalty * (1.0 - 0.5 * warm_fraction), 0.2, 1.0)
    
    # Surface temperature taper between 32-40 F.
    surface_weight = np.clip((41.0 - surf_temp_f) / 5.0, 0.0, 1.0)
    
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