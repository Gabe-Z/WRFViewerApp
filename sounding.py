from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from matplotlib import transforms
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QMainWindow, QVBoxLayout, QWidget

from calc import (
    parcel_trace_temperature_profile,
    sounding_isotherm_temperatures,
    sounding_pressure_bounds,
    sounding_pressure_levels,
    sounding_skewed_isotherm,
    sounding_temperature_bounds,
    sounding_temperature_ticks,
    virtual_temperature_profile,
)


def _standard_atmosphere_pressure(height_m: float) -> float:
    pressure0 = 1013.25 # hPa
    if height_m < 0:
        return pressure0
    
    # Troposhere (up to ~11 km): gradient temperature layer.
    if height_m <= 11000.0:
        return pressure0 * np.power(1.0 - 2.25577e-5 * height_m, 5.25588)
    
    # Lower stratosphere (11-20 km): isothermal layer.
    pl1 = pressure0 * np.power(1.0 - 2.25577e-5 * 11000.0, 5.25588)
    tl1 = 216.65 # K
    g0 = 9.80665 # m/s^2
    r = 287.053 # J/(kg*K)
    return pl1 * np.exp(-g0 * (height_m - 11000.0) / (r * tl1))


class SoundingWindow(QMainWindow):
    def __init__(
        self,
        timestamp: str,
        latitude: float,
        longitude: float,
        pressure_profile_hpa: np.ndarray | None = None,
        temperature_profile_c: np.ndarray  | None = None,
        height_profile_m: np.ndarray | None = None,
        dewpoint_profile_c: np.ndarray | None = None,
        sbcape_jkg: float | None = None,
        parent=None,
    ):
        super().__init__(parent)
        
        self.setWindowTitle(
            f'WRF Gabe Zago Sounding - {timestamp} - {latitude:.4f}, {longitude:.4f}'
        )
        self.setWindowState(Qt.WindowFullScreen)
        
        central = QWidget(self)
        central.setStyleSheet('background-color: black;')
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(central)
        
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 12)
        title_label = QLabel(
            f'WRF Gabe Zago Sounding - {timestamp} - {latitude:.4f}, {longitude:.4f}'
        )
        title_label.setStyleSheet('color: white; font-size: 18px; font-weight: 600;')
        header.addWidget(title_label)
        if sbcape_jkg is not None and np.isfinite(sbcape_jkg):
            cape_label = QLabel(f'SBCAPE: {sbcape_jkg:.0f} J/kg')
            cape_label.setStyleSheet('color: white; font-size: 14px; padding-left: 18px;')
            header.addWidget(cape_label)
        header.addStretch(1)
        
        btn_exit = QPushButton('Exit')
        btn_exit.setStyleSheet(
            'color: black; background-color: white; border: 1px solid white; padding: 6px 14px;'
        )
        btn_exit.clicked.connect(self.close)
        header.addWidget(btn_exit)
        layout.addLayout(header)
        
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        self.figure.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet('background-color: black;')
        self.canvas.setMinimumSize(1000, 800)
        self.canvas.setMaximumSize(1200, 950)
        layout.addWidget(self.canvas, alignment=Qt.AlignLeft | Qt.AlignTop)
        
        self._pressure_profile_hpa = pressure_profile_hpa
        self._temperature_profile_c = temperature_profile_c
        self._height_profile_m = height_profile_m
        self._dewpoint_profile_c = dewpoint_profile_c
        self._sbcape_jkg = sbcape_jkg
        
        self._draw_background()
        
        if pressure_profile_hpa is not None and temperature_profile_c is not None:
            self._plot_temperature_profile(pressure_profile_hpa, temperature_profile_c)
        if (
            pressure_profile_hpa is not None
            and temperature_profile_c is not None
            and dewpoint_profile_c is not None
        ):
            self._plot_virtual_temperature_profile(
                pressure_profile_hpa, temperature_profile_c, dewpoint_profile_c
            )
        if pressure_profile_hpa is not None and dewpoint_profile_c is not None:
            self._plot_dewpoint_profile(pressure_profile_hpa, dewpoint_profile_c)
        if (
            pressure_profile_hpa is not None
            and temperature_profile_c is not None
            and dewpoint_profile_c is not None
        ):
            self._plot_parcel_trace(
                pressure_profile_hpa, temperature_profile_c, dewpoint_profile_c
            )
        
    def _draw_background(self) -> None:
        temp_min, temp_max = sounding_temperature_bounds()
        pressure_bottom, pressure_top = sounding_pressure_bounds()
        pressure_levels = sounding_pressure_levels()
        temp_ticks = sounding_temperature_ticks()
        isotherm_temps = sounding_isotherm_temperatures()
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.tick_params(color='white')
        
        self.figure.canvas.draw()
        bbox = self.ax.get_window_extent().transformed(
            self.figure.dpi_scale_trans.inverted()
        )
        width, height = bbox.width, bbox.height
        self._aspect_correction = height / width if width > 0 else 1.0
        
        self.ax.set_xlim(temp_min, temp_max)
        self.ax.set_yscale('log')
        self.ax.set_ylim(pressure_bottom, pressure_top)
        self.ax.set_yticks(pressure_levels)
        self.ax.set_xticks(temp_ticks)
        
        self.ax.set_xticklabels([f'{temp:.0f}' for temp in temp_ticks], color='white')
        self.ax.set_yticklabels([f'{level:.0f}' for level in pressure_levels], color='white')
        
        for level in pressure_levels:
            self.ax.axhline(level, color='white', linewidth=1.0)
        
        pressures = np.geomspace(pressure_bottom, pressure_top, num=64, dtype=float)
        for temp in isotherm_temps:
            skewed_temps = sounding_skewed_isotherm(
                temp, pressures, aspect_correction=self._aspect_correction
            )
            self.ax.plot(
                skewed_temps,
                pressures,
                color='#8a8a8a',
                linestyle='--',
                linewidth=1.0,
                alpha=0.9,
                zorder=1,
            )
        
        self.ax.grid(False)
        self.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.ax.set_xlabel('Temperature (°C)', color='white', labelpad=12)
        self.ax.set_ylabel('Pressure (hPa)', color='white', labelpad=12)
        
        self.figure.tight_layout(rect=[0.04, 0.02, 0.98, 0.98])
        
        self._add_height_markers()
    
    def _add_height_markers(self) -> None:
        height_km_levels = [0, 1, 3, 6, 9, 12, 15]
        transform = transforms.blended_transform_factory(
            self.ax.transAxes, self.ax.transData
        )
        
        pressure_profile = getattr(self, '_pressure_profile_hpa', None)
        height_profile = getattr(self, '_height_profile_m', None)
        pressure_from_profile = None
        
        if pressure_profile is not None and height_profile is not None:
            pres = np.asarray(pressure_profile, dtype=float)
            hgt = np.asarray(height_profile, dtype=float)
            
            if hgt.ndim > 1:
                if hgt.shape[0] == pres.shape[0]:
                    # Collapse horizontal dimensions to a 1-D column when the
                    # leading (vertical) dimension matches the pressure count.
                    hgt = np.nanmean(hgt, axis=tuple(range(1, hgt.ndim)))
                elif hgt.size == pres.size:
                    # Last resort: reshape to match the pressure profile
                    # length when the total number of points aligns.
                    hgt = hgt.reshape(pres.shape)
                else:
                    hgt = np.asarray([], dtype=float)
            
            valid = np.isfinite(pres) & np.isfinite(hgt)
            if valid.sum() >= 2:
                pres = pres[valid]
                hgt = hgt[valid]
                surface_height = np.nanmin(hgt)
                hgt_agl = hgt - surface_height
                order = np.argsort(hgt_agl)
                hgt_agl_sorted = hgt_agl[order]
                pres_sorted = pres[order]
                
                def pressure_from_profile(height_m: float) -> float:
                    if height_m < hgt_agl_sorted[0] or height_m > hgt_agl_sorted[-1]:
                        return
                    return float(
                        np.interp(
                            height_m,
                            hgt_agl_sorted,
                            pres_sorted,
                            left=np.nan,
                            right=np.nan,
                        )
                    )
        
        for height_km in height_km_levels:
            height_m = height_km * 1000.0
            pressure_hpa = (
                pressure_from_profile(height_m)
                if pressure_from_profile is not None
                else _standard_atmosphere_pressure(height_m)
            )
            if not np.isfinite(pressure_hpa):
                continue
            
            self.ax.plot(
                [0.001, 0.02],
                [pressure_hpa, pressure_hpa],
                transform=transform,
                color='red',
                linewidth=1.4,
                clip_on=False,
                zorder=5,
            )
            self.ax.text(
                0.025,
                pressure_hpa,
                f'{height_km} km',
                transform=transform,
                color='red',
                fontsize=9,
                va='center',
                ha='left',
                bbox={
                'facecolor': 'black',
                'edgecolor': 'none',
                'alpha': 0.5,
                'pad': 2.5,
                },
                clip_on=False,
                zorder=5,
            )
    
    def _plot_temperature_profile(
        self, pressure_hpa: np.ndarray, temperature_c: np.ndarray
    ) -> None:
        valid = np.isfinite(pressure_hpa) & np.isfinite(temperature_c)
        if valid.sum() < 2:
            return
        
        skewed_temps = sounding_skewed_isotherm(
            temperature_c[valid],
            pressure_hpa[valid],
            aspect_correction=getattr(self, '_aspect_correction', 1.0),
        )
        
        self.ax.plot(
            skewed_temps,
            pressure_hpa[valid],
            color='red',
            linewidth=2.2,
            zorder=6,
        )
        
        # Show the surface temperature in Fahrenheit inline near the
        # lowest-level pressure to quickly verify the plotted value.
        surface_idx = int(np.nanargmax(pressure_hpa[valid]))
        surface_temp_c = float(temperature_c[valid][surface_idx])
        surface_temp_f = surface_temp_c * 9.0 / 5.0 + 32.0
        surface_pressure = float(pressure_hpa[valid][surface_idx])
        surface_x = float(skewed_temps[surface_idx])
        label_offset = 0.8
        self.ax.text(
            surface_x + label_offset,
            surface_pressure,
            f'{surface_temp_f:.1f}°F',
            color='red',
            fontsize=10,
            va='center',
            ha='left',
            bbox={
                'facecolor': 'black',
                'edgecolor': 'none',
                'alpha': 0.5,
                'pad': 2.5,
            },
            zorder=7,
        )
        self.figure.canvas.draw_idle()
    
    def _plot_virtual_temperature_profile(
        self, pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray
    ) -> None:
        vt_c = virtual_temperature_profile(pressure_hpa, temperature_c, dewpoint_c)
        
        valid = np.isfinite(pressure_hpa) & np.isfinite(vt_c)
        if valid.sum() < 2:
            return
        
        skewed_temps = sounding_skewed_isotherm(
            vt_c[valid],
            pressure_hpa[valid],
            aspect_correction=getattr(self, '_aspect_correction', 1.0),
        )
        
        self.ax.plot(
            skewed_temps,
            pressure_hpa[valid],
            color='red',
            linestyle='--',
            linewidth=2.0,
            zorder=5,
        )
        self.figure.canvas.draw_idle()
    
    def _plot_dewpoint_profile(self, pressure_hpa: np.ndarray, dewpoint_c: np.ndarray) -> None:
        valid = np.isfinite(pressure_hpa) & np.isfinite(dewpoint_c)
        if valid.sum() < 2:
            return
        
        skewed_temps = sounding_skewed_isotherm(
            dewpoint_c[valid],
            pressure_hpa[valid],
            aspect_correction=getattr(self, '_aspect_correction', 1.0),
        )
        
        self.ax.plot(
            skewed_temps,
            pressure_hpa[valid],
            color='lime',
            linewidth=2.0,
            zorder=6,
        )
        
        surface_idx = int(np.nanargmax(pressure_hpa[valid]))
        surface_dewpoint_c = float(dewpoint_c[valid][surface_idx])
        surface_dewpoint_c_f = surface_dewpoint_c * 9.0 / 5.0 + 32.0
        surface_pressure = float(pressure_hpa[valid][surface_idx])
        surface_x = float(skewed_temps[surface_idx])
        label_offset = 0.8
        self.ax.text(
            surface_x - label_offset,
            surface_pressure,
            f'{surface_dewpoint_c_f:.1f}°F',
            color='lime',
            fontsize=10,
            va='center',
            ha='right',
            bbox={
                'facecolor': 'black',
                'edgecolor': 'none',
                'alpha': 0.5,
                'pad': 2.5,
            },
            zorder=7,
        )
        self.figure.canvas.draw_idle()
    
    def _plot_parcel_trace(
        self, pressure_hpa: np.ndarray, temperature_c: np.ndarray, dewpoint_c: np.ndarray
    ) -> None:
        parcel_temp_c = parcel_trace_temperature_profile(
            pressure_hpa, temperature_c, dewpoint_c
        )
        
        valid = np.isfinite(pressure_hpa) & np.isfinite(parcel_temp_c)
        if valid.sum() < 2:
            return
        
        skewed_temps = sounding_skewed_isotherm(
            parcel_temp_c[valid],
            pressure_hpa[valid],
            aspect_correction=getattr(self, '_aspect_correction', 1.0),
        )
        
        self.ax.plot(
            skewed_temps,
            pressure_hpa[valid],
            color='#a0a0a0',
            linestyle='--',
            linewidth=1.8,
            zorder=5,
        )
        self.figure.canvas.draw_idle()