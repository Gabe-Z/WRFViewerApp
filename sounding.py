from __future__ import annotations

import datetime
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from matplotlib import transforms
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
    QWidget
)

from calc import (
    parcel_thermo_indices_from_profile,
    mixed_layer_parcel_source,
    parcel_trace_temperature_profile,
    most_unstable_parcel_source,
    bunkers_storm_motion,
    effective_inflow_layer,
    pressure_weighted_mean_wind_components,
    sounding_isotherm_temperatures,
    sounding_pressure_bounds,
    sounding_pressure_levels,
    sounding_skewed_isotherm,
    sounding_temperature_bounds,
    sounding_temperature_ticks,
    virtual_temperature_profile,
    storm_relative_wind_components,
    wind_components_from_direction_speed,
    wind_components_at_height,
    shear_vector,
)


def _app_root() -> Path:
    if getattr(sys, 'frozen', False):
        return(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


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
        wind_direction_deg: np.ndarray | None = None,
        wind_speed_ms: np.ndarray | None = None,
        frame_timestamp: datetime.datetime | None = None,
        sbcape_jkg: float | None = None,
        parent=None,
    ):
        super().__init__(parent)
        
        self._timestamp = timestamp
        self._latitude = latitude
        self._longitude = longitude
        self._frame_timestamp = frame_timestamp
        self._window_title_text = (
            f'WRF Gabe Zago Sounding - {timestamp} - {latitude:.4f}, {longitude:.4f}'
        )
        self.setWindowTitle(self._window_title_text)
        self.setWindowState(Qt.WindowFullScreen)
        
        central = QWidget(self)
        central.setStyleSheet('background-color: black;')
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(central)
        
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 12)
        parcel_label = QLabel('Parcel:')
        parcel_label.setStyleSheet('color: white; font-size: 15px; font-weight: 600;')
        header.addWidget(parcel_label)
        
        self._parcel_combo = QComboBox()
        self._parcel_combo.addItems(
            ['Surface-Based', 'Most Unstable', 'Mixed-Layer']
        )
        self._parcel_combo.setMinimumContentsLength(15)
        self._parcel_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._parcel_combo.setStyleSheet(
            'color: black; background-color: white; border: 1px solid white; padding: 4px 6px;'
        )
        self._parcel_combo.currentTextChanged.connect(self._on_parcel_selection_changed)
        header.addWidget(self._parcel_combo)
        
        hodograph_label = QLabel('Hodograph:')
        hodograph_label.setStyleSheet('color: white; font-size: 15px; font-weight: 600;')
        header.addWidget(hodograph_label)
        
        self._hodograph_mode_combo = QComboBox()
        self._hodograph_mode_combo.addItems(['Ground-Relative', 'Storm-Relative'])
        self._hodograph_mode_combo.setMinimumContentsLength(16)
        self._hodograph_mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._hodograph_mode_combo.setStyleSheet(
            'color: black; background-color: white; border: 1px solid white; padding: 4px 6px;'
        )
        self._hodograph_mode_combo.currentTextChanged.connect(self._on_hodograph_mode_changed)
        header.addWidget(self._hodograph_mode_combo)
        
        title_label = QLabel(self._window_title_text)
        title_label.setStyleSheet('color: white; font-size: 18px; font-weight: 600;')
        header.addWidget(title_label)
        header.addStretch(1)
        
        btn_export = QPushButton('Export')
        btn_export.setStyleSheet(
            'color: black; background-color: white; border: 1px solid white; padding: 6px 14px;'
        )
        btn_export.clicked.connect(self._export_sounding)
        header.addWidget(btn_export)
        
        btn_exit = QPushButton('Exit')
        btn_exit.setStyleSheet(
            'color: black; background-color: white; border: 1px solid white; padding: 6px 14px;'
        )
        btn_exit.clicked.connect(self.close)
        header.addWidget(btn_exit)
        layout.addLayout(header)
        
        self.figure = plt.figure(figsize=(12, 8))
        self.figure.patch.set_facecolor('black')
        
        grid = self.figure.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[4.5, 3.5],
            left=0.06,
            right=0.98,
            top=0.96,
            bottom=0.08,
            wspace=0.08,
        )
        
        self.ax = self.figure.add_subplot(grid[0, 0])
        self.ax.set_facecolor('black')
        self.ax.set_anchor('NW')
        self.ax.set_box_aspect(1.0)
        
        self._hodograph_ax = self.figure.add_subplot(grid[0, 1])
        self._hodograph_ax.set_facecolor('black')
        self._hodograph_ax.set_anchor('NW')
        self._resize_hodograph_axis()

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet('background-color: black;')
        self.canvas.setMinimumSize(1020, 600)
        layout.addWidget(self.canvas)
        
        self._pressure_profile_hpa = pressure_profile_hpa
        self._temperature_profile_c = temperature_profile_c
        self._height_profile_m = height_profile_m
        self._dewpoint_profile_c = dewpoint_profile_c
        self._wind_direction_deg = wind_direction_deg
        self._wind_speed_ms = wind_speed_ms
        self._sbcape_jkg = sbcape_jkg
        self._parcel_label_widgets: dict[str, QLabel] = {}
        self._parcel_artists = []
        self._hodograph_artist: list[object] = []
        self._hodograph_range_ms = 60.0
        self._hodograph_mode = 'Ground-Relative'
        self._hodograph_profile: dict[str, np.ndarray] = {}
        self._storm_relative_profile: dict[str, np.ndarray] = {}
        self._storm_motion: dict[str, tuple[float, float]] = {}
        self._effective_inflow_layer: tuple[float, float] = (np.nan, np.nan)
        self._low_level_shear: tuple[float, float] = (np.nan, np.nan)
        self._parcel_data: dict[str, dict[str, float | None]] = {}
        
        self._draw_background()
        self._init_hodograph_axes()
        
        if (
            pressure_profile_hpa is not None
            and temperature_profile_c is not None
            and dewpoint_profile_c is not None
            and height_profile_m is not None
        ):
            self._parcel_data = self._compute_parcel_data(
                pressure_profile_hpa, temperature_profile_c, dewpoint_profile_c, height_profile_m
            )
        
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
            height_profile_m is not None
            and wind_direction_deg is not None
            and wind_speed_ms is not None
        ):
            self._compute_hodograph_diagnostics(
                height_profile_m,
                wind_direction_deg,
                wind_speed_ms,
                pressure_profile_hpa,
                temperature_profile_c,
                dewpoint_profile_c,
            )
            self._plot_hodograph()
        else:
            self._hodograph_mode_combo.setEnabled(False)
        if (
            pressure_profile_hpa is not None
            and temperature_profile_c is not None
            and dewpoint_profile_c is not None
            and height_profile_m is not None
        ):
            self._add_parcel_indices_section(
                pressure_profile_hpa, temperature_profile_c, dewpoint_profile_c, height_profile_m
            )
        
        self._on_parcel_selection_changed(self._parcel_combo.currentText())
    
    def _export_sounding(self) -> None:
        '''try:'''
        if self._pressure_profile_hpa is None:
            raise ValueError('Pressure profile is unavailable.')
        if self._height_profile_m is None:
            raise ValueError('Height profile is unavailable.')
        if self._temperature_profile_c is None:
            raise ValueError('Temperature profile is unavailable.')
        if self._dewpoint_profile_c is None:
            raise ValueError('Dewpoint profile is unavailable.')
        if self._wind_direction_deg is None or self._wind_speed_ms is None:
            raise ValueError('Wind profile is unavailable.')
        
        computer_name = (platform.node() or 'computer').replace(' ', '_')
        export_time = self._frame_timestamp or datetime.datetime.now()
        yymmdd_hhmm = export_time.strftime('%y%m%d/%H%M')
        yymmdd_hh = export_time.strftime('%y%m%d/%H')
        timestamp = export_time.strftime('%Y-%m-%d_%H_%M_%S')
        export_dir = _app_root() / 'Sounding Export'
        export_dir.mkdir(parents=True, exist_ok=True)
        filename = export_dir / (
            f'WRF_{computer_name}_{self._latitude:.4f}_{self._longitude:.4f}_{timestamp}.txt'
        )
        title_name = (
            f'{computer_name}-WRF|-|{yymmdd_hhmm}|{self._latitude:.4f},{self._longitude:.4f}| {yymmdd_hh}'
        )
        
        lines = [
            '%TITLE%',
            title_name,
            '	LEVEL	HGHT	TEMP	DWPT	WDIR	WSPD',
            '-------------------------------------------------',
            '%RAW%',
        ]
        
        row_count = min(
            len(self._pressure_profile_hpa),
            len(self._height_profile_m),
            len(self._temperature_profile_c),
            len(self._dewpoint_profile_c),
            len(self._wind_direction_deg),
            len(self._wind_speed_ms),
        )
        
        for idx in range(row_count):
            pressure = float(self._pressure_profile_hpa[idx])
            height = float(self._height_profile_m[idx])
            temp = float(self._temperature_profile_c[idx])
            dew = float(self._dewpoint_profile_c[idx])
            wdir = float(self._wind_direction_deg[idx])
            wspd = float(self._wind_speed_ms[idx]) * 1.94384 # Convert m/s to knots
            lines.append(
                f'  {pressure:.1f}, {height:.1f},   {temp:.1f}, {dew:.1f},  {wdir:.1f}, {wspd:.1f}'
            )
        
        lines.append('%END%')
        
        with open(filename, 'w', encoding='utf-8') as export_file:
            export_file.write('\n'.join(lines))
        
        QMessageBox.information(
            self,
            'Export Successful',
            f'Sounding exported to {filename}',
        )
        '''except Exception as exc: # pylint: disable=broad-except
            QMessageBox.critical(
                self,
                'Export Failed',
                f'Could not export sounding: {exc}',
            )'''
    
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
        self.ax.set_title('Skew-T Sounding', color='white', fontsize=12, pad=10)
        
        self._add_height_markers()
    
    def _resize_hodograph_axis(self, scale: float = 0.75) -> None:
        if self._hodograph_ax is None:
            return
        
        bbox = self._hodograph_ax.get_position()
        scaled_width = bbox.width * scale
        scaled_height = bbox.height * scale
        new_x0 = bbox.x1 - scaled_width
        new_y0 = bbox.y1 - scaled_height
        self._hodograph_ax.set_position([new_x0, new_y0, scaled_width, scaled_height])
    
    def _init_hodograph_axes(self) -> None:
        self._hodograph_ax.set_facecolor('black')
        for spine in self._hodograph_ax.spines.values():
            spine.set_color('white')
        self._hodograph_ax.tick_params(color='white', labelcolor='white')
        self._hodograph_ax.set_aspect('equal', adjustable='box')
        self._hodograph_ax.set_xlim(-self._hodograph_range_ms, self._hodograph_range_ms)
        self._hodograph_ax.set_ylim(-self._hodograph_range_ms, self._hodograph_range_ms)
        self._hodograph_ax.set_xlabel('u (kt)', color='white')
        self._hodograph_ax.set_ylabel('v (kt)', color='white')
        self._hodograph_ax.set_title('Hodograph', color='white', fontsize=12, pad=10)
        self._hodograph_ax.grid(True, color='#444444', linestyle='--', linewidth=0.8, alpha=0.7)
        self._hodograph_ax.axhline(0.0, color='#666666', linewidth=1.0, zorder=1)
        self._hodograph_ax.axvline(0.0, color='#666666', linewidth=1.0, zorder=1)
        
        rings = np.arange(10.0, self._hodograph_range_ms + 0.1, 10.0)
        for radius in rings:
            circle = plt.Circle(
                (0.0, 0.0),
                radius,
                color='#5a5a5a',
                fill=False,
                linestyle=':',
                linewidth=0.8,
                alpha=0.8,
                zorder=1,
            )
            self._hodograph_ax.add_artist(circle)
            self._hodograph_ax.text(
                radius - 3.5,
                0.5,
                f'{radius:.0f}',
                color='white',
                fontsize=8,
                ha='left',
                va='bottom',
                alpha=0.9,
                zorder=2,
            )
            self._hodograph_ax.text(
                -radius,
                0.5,
                f'{radius:.0f}',
                color='white',
                fontsize=8,
                ha='left',
                va='bottom',
                alpha=0.9,
                zorder=2,
            )
            self._hodograph_ax.text(
                0.5,
                radius - 3.5,
                f'{radius:.0f}',
                color='white',
                fontsize=8,
                ha='left',
                va='bottom',
                alpha=0.9,
                zorder=2,
            )
            self._hodograph_ax.text(
                0.5,
                -radius,
                f'{radius:.0f}',
                color='white',
                fontsize=8,
                ha='left',
                va='bottom',
                alpha=0.9,
                zorder=2,
            )
    
    def _cape_color(self, value: float) -> str:
        if not np.isfinite(value):
            return '#a0a0a0'
        if value >= 4000.0:
            return 'purple'
        if value >= 3000.0:
            return 'red'
        if value >= 2000.0:
            return 'yellow'
        return 'white'
    
    def _cinh_color(self, value: float) -> str:
        if not np.isfinite(value):
            return '#a0a0a0'
        if value <= -100.0:
            return 'maroon'
        if value <= -50.0:
            return 'brown'
        if value <= 0.0:
            return 'green'
        return 'white'
    
    def _format_parcel_value(self, value: float) -> str:
        if not np.isfinite(value):
            return 'N/A'
        return f'{value:.0f} J/kg'
    
    def _format_height_value(self, unstable: bool, value: float) -> str:
        if not unstable or not np.isfinite(value):
            return 'N/A'
        return f'{value:.0f}'
    
    def _format_lifted_index(self, unstable: bool, value: float) -> str:
        if not unstable or not np.isfinite(value):
            return 'N/A'
        return f'{value:.1f}'
    
    def _selected_parcel_key(self) -> str:
        combo = getattr(self, '_parcel_combo', None)
        if combo is None:
            return 'Surface-Based'
        return combo.currentText() or 'Surface-Based'
    
    def _parcel_is_buoyant(self, parcel: dict[str, float | None]) -> bool:
        return np.isfinite(parcel.get('cape', np.nan)) and parcel.get('cape', 0.0) > 10.0
    
    def _clear_parcel_artists(self) -> None:
        for artist in self._parcel_artists:
            try:
                artist.remove()
            except ValueError:
                continue
        self._parcel_artists = []
        self.figure.canvas.draw_idle()
    
    def _update_parcel_row_styles(self) -> None:
        selected_key = self._selected_parcel_key()
        for key, widget in self._parcel_label_widgets.items():
            color = 'yellow' if key == selected_key else 'white'
            widget.setStyleSheet(f'color: {color}; font-size: 13px; font-weight: 600;')
    
    def _compute_parcel_data(
        self,
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        height_m: np.ndarray | None,
    ) -> dict[str, dict[str, float | None]]:
        if height_m is None:
            return {}
        
        sb_cape, sb_cinh, sb_lcl, sb_lfc, sb_el, sb_li = parcel_thermo_indices_from_profile(
            pressure_hpa, temperature_c, dewpoint_c, height_m
        )
        
        mu_p, mu_temp, mu_dew = most_unstable_parcel_source(
            pressure_hpa, temperature_c, dewpoint_c
        )
        mu_cape, mu_cinh, mu_lcl, mu_lfc, mu_el, mu_li = parcel_thermo_indices_from_profile(
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            height_m,
            start_pressure_hpa=mu_p if np.isfinite(mu_p) else None,
            start_temperature_c=mu_temp if np.isfinite(mu_temp) else None,
            start_dewpoint_c=mu_dew if np.isfinite(mu_dew) else None,
        )
        
        ml_p, ml_temp, ml_dew = mixed_layer_parcel_source(
            pressure_hpa, temperature_c, dewpoint_c
        )
        ml_cape, ml_cinh, ml_lcl, ml_lfc, ml_el, ml_li = parcel_thermo_indices_from_profile(
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            height_m,
            start_pressure_hpa=ml_p if np.isfinite(ml_p) else None,
            start_temperature_c=ml_temp if np.isfinite(ml_temp) else None,
            start_dewpoint_c=ml_dew if np.isfinite(ml_dew) else None,
        )
        
        return {
            'Surface-Based': {
                'label': 'SFC',
                'cape': sb_cape,
                'cinh': sb_cinh,
                'lcl': sb_lcl,
                'lfc': sb_lfc,
                'el': sb_el,
                'li': sb_li,
                'start_pressure': None,
                'start_temperature': None,
                'start_dewpoint': None,
            },
            'Most Unstable': {
                'label': 'MU',
                'cape': mu_cape,
                'cinh': mu_cinh,
                'lcl': mu_lcl,
                'lfc': mu_lfc,
                'el': mu_el,
                'li': mu_li,
                'start_pressure': mu_p if np.isfinite(mu_p) else None,
                'start_temperature': mu_temp if np.isfinite(mu_temp) else None,
                'start_dewpoint': mu_dew if np.isfinite(mu_dew) else None,
            },
            'Mixed-Layer': {
                'label': 'ML',
                'cape': ml_cape,
                'cinh': ml_cinh,
                'lcl': ml_lcl,
                'lfc': ml_lfc,
                'el': ml_el,
                'li': ml_li,
                'start_pressure': ml_p if np.isfinite(ml_p) else None,
                'start_temperature': ml_temp if np.isfinite(ml_temp) else None,
                'start_dewpoint': ml_dew if np.isfinite(ml_dew) else None,
            }
        }
    
    def _add_parcel_row(
        self,
        grid: QVBoxLayout | QGridLayout,
        row_index: int,
        label: str,
        cape: float,
        cinh: float,
        lcl: float,
        lfc: float,
        li: float,
        el: float,
        unstable: bool,
        *,
        highlight: bool,
        key: str,
    ) -> None:
        row_unstable = unstable and np.isfinite(cape) and cape > 0.0
        label_widget = QLabel(label)
        label_color = 'yellow' if highlight else 'white'
        label_widget.setStyleSheet(f'color: {label_color}; font-size: 13px; font-weight: 600;')
        label_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        self._parcel_label_widgets[key] = label_widget
        
        cape_widget = QLabel(self._format_parcel_value(cape))
        cape_widget.setStyleSheet(f'color: {self._cape_color(cape)}; font-size: 13px;')
        cape_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        cinh_widget = QLabel(self._format_parcel_value(cinh))
        cinh_widget.setStyleSheet(f'color: {self._cinh_color(cinh)}; font-size: 13px;')
        cinh_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        lcl_widget = QLabel(self._format_height_value(row_unstable, lcl))
        lcl_widget.setStyleSheet(f'color: white; font-size: 13px;')
        lcl_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        lfc_widget = QLabel(self._format_height_value(row_unstable, lfc))
        lfc_widget.setStyleSheet(f'color: white; font-size: 13px;')
        lfc_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        li_widget = QLabel(self._format_lifted_index(row_unstable, li))
        li_widget.setStyleSheet(f'color: white; font-size: 13px;')
        li_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        el_widget = QLabel(self._format_height_value(row_unstable, el))
        el_widget.setStyleSheet(f'color: white; font-size: 13px;')
        el_widget.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        
        grid.addWidget(label_widget, row_index, 0)
        grid.addWidget(cape_widget, row_index, 1)
        grid.addWidget(cinh_widget, row_index, 2)
        grid.addWidget(lcl_widget, row_index, 3)
        grid.addWidget(lfc_widget, row_index, 4)
        grid.addWidget(li_widget, row_index, 5)
        grid.addWidget(el_widget, row_index, 6)
    
    def _add_parcel_indices_section(
        self,
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        height_m: np.ndarray,
    ) -> None:
        container = QWidget(self)
        container.setStyleSheet('background-color: #1b1b1b; border: 1px solid #3a3a3a;')
        section = QGridLayout(container)
        section.setContentsMargins(12, 12, 12, 12)
        section.setHorizontalSpacing(18)
        section.setVerticalSpacing(6)
        section.setColumnStretch(0, 2)
        section.setColumnStretch(1, 1)
        section.setColumnStretch(2, 1)
        section.setColumnStretch(3, 1)
        section.setColumnStretch(4, 1)
        section.setColumnStretch(5, 1)
        section.setColumnStretch(6, 1)
        
        for col, text in enumerate(('Parcel', 'CAPE', 'CINH', 'LCL', 'LFC', 'LI', 'EL')):
            lbl = QLabel(text)
            lbl.setStyleSheet('color: white; font-size: 13px; font-weight: 700;')
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
            section.addWidget(lbl, 0, col)
        
        if not self._parcel_data:
            self._parcel_data = self._compute_parcel_data(
                pressure_hpa, temperature_c, dewpoint_c, height_m
            )
        
        parcels = self._parcel_data
        sb = parcels.get('Surface-Based', {})
        mu = parcels.get('Most Unstable', {})
        ml = parcels.get('Mixed-Layer', {})
        
        unstable = any(self._parcel_is_buoyant(parcel) for parcel in parcels.values())
        
        selected_key = self._selected_parcel_key()
        
        self._add_parcel_row(
            section,
            1,
            sb.get('label', 'SFC'),
            sb.get('cape', np.nan),
            sb.get('cinh', np.nan),
            sb.get('lcl', np.nan),
            sb.get('lfc', np.nan),
            sb.get('li', np.nan),
            sb.get('el', np.nan),
            unstable,
            highlight=selected_key == 'Surface-Based',
            key='Surface-Based',
        )
        self._add_parcel_row(
            section,
            2,
            mu.get('label', 'MU'),
            mu.get('cape', np.nan),
            mu.get('cinh', np.nan),
            mu.get('lcl', np.nan),
            mu.get('lfc', np.nan),
            mu.get('li', np.nan),
            mu.get('el', np.nan),
            unstable,
            highlight=selected_key == 'Most Unstable',
            key='Most Unstable',
        )
        self._add_parcel_row(
            section,
            3,
            ml.get('label', 'ML'),
            ml.get('cape', np.nan),
            ml.get('cinh', np.nan),
            ml.get('lcl', np.nan),
            ml.get('lfc', np.nan),
            ml.get('li', np.nan),
            ml.get('el', np.nan),
            unstable,
            highlight=selected_key == 'Mixed-Layer',
            key='Mixed-Layer',
        )
        
        # Align the parcel indices with the Skew-T's y-axis by offsetting the
        # widget by the axes' left position within the canvas.
        self.canvas.draw()
        axes_pos = self.ax.get_position()
        axis_left_px = axes_pos.x0 * self.figure.get_figwidth() * self.figure.dpi
        left_pad = max(int(axis_left_px), 0)
        
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addSpacing(left_pad)
        row.addWidget(container, alignment=Qt.AlignLeft | Qt.AlignTop)
        row.addStretch(1)
        
        self.centralWidget().layout().addLayout(row)
        
        self._update_parcel_row_styles()
    
    def _pressure_from_height(self, height_m: float) -> float:
        converter = getattr(self, '_pressure_from_profile', None)
        if converter is not None:
            pressure = converter(height_m)
            if pressure is not None and np.isfinite(pressure):
                return pressure
        return _standard_atmosphere_pressure(height_m)
    
    def _add_parcel_level_markers(
        self, lcl_height_m: float, lfc_height_m: float, el_height_m: float
    ) -> list:
        level_info = (
            ('LCL', lcl_height_m, 'green', 'top'),
            ('LFC', lfc_height_m, 'yellow', 'bottom'),
            ('EL', el_height_m, 'purple', 'bottom'),
        )
        
        x_center = 40.0
        half_width = 0.8
        offset_px = 6.0 / self.figure.dpi
        
        artists = []
        
        for label, height_m, color, valign in level_info:
            if not np.isfinite(height_m):
                continue
            pressure_hpa = self._pressure_from_height(height_m)
            if not np.isfinite(pressure_hpa):
                continue
            y_offset = offset_px if valign == 'bottom' else -offset_px
            text_transform = self.ax.transData + transforms.ScaledTranslation(
                0.0, y_offset, self.figure.dpi_scale_trans
            )
            line = self.ax.plot(
                [x_center - half_width, x_center + half_width],
                [pressure_hpa, pressure_hpa],
                color=color,
                linewidth=2.0,
                solid_capstyle='butt',
                zorder=8,
            )[0]
            text = self.ax.text(
                x_center,
                pressure_hpa,
                label,
                color=color,
                fontsize=10,
                va='bottom' if valign == 'bottom' else 'top',
                ha='center',
                transform=text_transform,
                bbox={
                    'facecolor': 'black',
                    'edgecolor': 'none',
                    'alpha': 0.6,
                    'pad': 2.5,
                },
                zorder=8,
            )
            artists.extend((line, text))
        
        self.figure.canvas.draw_idle()
        return artists
    
    def _on_parcel_selection_changed(self, _: str) -> None:
        if (
            not self._parcel_data
            or self._pressure_profile_hpa is None
            or self._temperature_profile_c is None
            or self._dewpoint_profile_c is None
        ):
            return
        
        self._update_parcel_row_styles()
        self._clear_parcel_artists()
        
        parcel = self._parcel_data.get(self._selected_parcel_key())
        if not parcel:
            return
        
        if not self._parcel_is_buoyant(parcel):
            return
        
        line = self._plot_parcel_trace(
            self._pressure_profile_hpa,
            self._temperature_profile_c,
            self._dewpoint_profile_c,
            start_pressure_hpa=parcel.get('start_pressure'),
            start_temperature_c=parcel.get('start_temperature'),
            start_dewpoint_c=parcel.get('start_dewpoint'),
            color='white',
            zorder=6,
        )
        if line is not None:
            self._parcel_artists.append(line)
        
        markers = self._add_parcel_level_markers(
            parcel.get('lcl', np.nan),
            parcel.get('lfc', np.nan),
            parcel.get('el', np.nan),
        )
        self._parcel_artists.extend(markers)
        
        if (
            self._height_profile_m is not None
            and self._wind_direction_deg is not None
            and self._wind_speed_ms is not None
        ):
            self._compute_hodograph_diagnostics(
                self._height_profile_m,
                self._wind_direction_deg,
                self._wind_speed_ms,
                self._pressure_profile_hpa,
                self._temperature_profile_c,
                self._dewpoint_profile_c,
            )
            self._plot_hodograph()
        
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
                
                self._pressure_from_profile = pressure_from_profile
        
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
                clip_on=True,
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
                clip_on=True,
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
        self,
        pressure_hpa: np.ndarray,
        temperature_c: np.ndarray,
        dewpoint_c: np.ndarray,
        *,
        start_pressure_hpa: float | None = None,
        start_temperature_c: float | None = None,
        start_dewpoint_c: float | None = None,
        color: str = '#a0a0a0',
        zorder: int = 5,
    ) -> object | None:
        parcel_temp_c = parcel_trace_temperature_profile(
            pressure_hpa,
            temperature_c,
            dewpoint_c,
            start_pressure_hpa=start_pressure_hpa,
            start_temperature_c=start_temperature_c,
            start_dewpoint_c=start_dewpoint_c,
        )
        
        valid = np.isfinite(pressure_hpa) & np.isfinite(parcel_temp_c)
        if valid.sum() < 2:
            return
        
        skewed_temps = sounding_skewed_isotherm(
            parcel_temp_c[valid],
            pressure_hpa[valid],
            aspect_correction=getattr(self, '_aspect_correction', 1.0),
        )
        
        line = self.ax.plot(
            skewed_temps,
            pressure_hpa[valid],
            color=color,
            linestyle='--',
            linewidth=1.8,
            zorder=zorder,
        )[0]
        self.figure.canvas.draw_idle()
        return line
    
    def _compute_hodograph_diagnostics(
        self,
        height_m: np.ndarray,
        wind_direction_deg: np.ndarray,
        wind_speed_ms: np.ndarray,
        pressure_hpa: np.ndarray | None,
        temperature_c: np.ndarray | None,
        dewpoint_c: np.ndarray | None,
    ) -> None:
        self._hodograph_profile = {}
        self._storm_relative_profile = {}
        self._storm_motion = {}
        self._effective_inflow_layer = (np.nan, np.nan)
        self._low_level_shear = (np.nan, np.nan)
        
        hgt = np.asarray(height_m, dtype=float)
        wdir = np.asarray(wind_direction_deg, dtype=float)
        wspd = np.asarray(wind_speed_ms, dtype=float)
        
        valid = np.isfinite(hgt) & np.isfinite(wdir) & np.isfinite(wspd) & (hgt <= 12000.0)
        if valid.sum() < 2:
            self._hodograph_mode_combo.setEnabled(False)
            return
        
        surface_height = float(np.nanmin(hgt[valid]))
        hgt_agl_full = hgt - surface_height
        
        hgt = hgt_agl_full[valid]
        wdir = wdir[valid]
        wspd = wspd[valid]
        
        order = np.argsort(hgt)
        hgt = hgt[order]
        wdir = wdir[order]
        wspd = wspd[order]
        
        u_ms, v_ms = wind_components_from_direction_speed(wdir, wspd)
        ms_to_kt = 1.94384
        self._hodograph_profile = {
            'height_m': hgt,
            'u_ms': u_ms,
            'v_ms': v_ms,
            'u_kt': u_ms * ms_to_kt,
            'v_kt': v_ms * ms_to_kt,
        }
        
        self._storm_motion = bunkers_storm_motion(hgt, wdir, wspd)
        parcel_layer = self._parcel_data.get(self._selected_parcel_key(), {}) if self._parcel_data else {}
        lcl_height = parcel_layer.get('lcl', np.nan) if isinstance(parcel_layer, dict) else np.nan
        el_height = parcel_layer.get('el', np.nan) if isinstance(parcel_layer, dict) else np.nan
        
        if (
            pressure_hpa is not None
            and np.isfinite(lcl_height)
            and np.isfinite(el_height)
            and el_height > lcl_height
        ):
            mw_u, mw_v = pressure_weighted_mean_wind_components(
                hgt_agl_full, pressure_hpa, wind_direction_deg, wind_speed_ms, lcl_height, el_height
            )
            if np.isfinite(mw_u) and np.isfinite(mw_v):
                self._storm_motion['MW'] = (mw_u, mw_v)
        
        rm_u, rm_v = self._storm_motion.get('RM', (np.nan, np.nan))
        if np.isfinite(rm_u) and np.isfinite(rm_v):
            sr_u_ms, sr_v_ms = storm_relative_wind_components(u_ms, v_ms, rm_u, rm_v)
            self._storm_relative_profile = {
                'height_m': hgt,
                'u_ms': sr_u_ms,
                'v_ms': sr_v_ms,
                'u_kt': sr_u_ms * ms_to_kt,
                'v_kt': sr_v_ms * ms_to_kt,
            }
        
        if (
            pressure_hpa is not None
            and temperature_c is not None
            and dewpoint_c is not None
        ):
            self._effective_inflow_layer = effective_inflow_layer(
                pressure_hpa, temperature_c, dewpoint_c, hgt_agl_full
            )
        
        base_height, _ = self._effective_inflow_layer
        if np.isfinite(base_height) and base_height <= 50.0:
            shear_u, shear_v = shear_vector(
                hgt_agl_full, wind_direction_deg, wind_speed_ms, base_height, 500.0
            )
            if np.isfinite(shear_u) and np.isfinite(shear_v):
                self._low_level_shear = (
                    shear_u,
                    shear_v,
                )
        
        self._hodograph_mode_combo.setEnabled(True)
    
    def _components_at_height(self, target_height_m: float, storm_relative: bool = False) -> tuple[float, float]:
        profile = self._storm_relative_profile if (storm_relative and self._storm_relative_profile) else self._hodograph_profile
        if not self._hodograph_profile:
            return np.nan, np.nan
        
        heights = profile.get('height_m')
        u_ms = profile.get('u_ms')
        v_ms = profile.get('v_ms')
        if heights is None or u_ms is None or v_ms is None:
            return np.nan, np.nan
            
        if target_height_m < heights.min() or target_height_m > heights.max():
            return np.nan, np.nan
        
        u = float(np.interp(target_height_m, heights, u_ms))
        v = float(np.interp(target_height_m, heights, v_ms))
        return u, v
    
    def _hodograph_layer_color(self, height_m: float) -> str:
        height_km = height_m / 1000.0
        if height_km < 1.0:
            return 'purple'
        if height_km < 3.0:
            return 'red'
        if height_km < 6.0:
            return 'green'
        if height_km < 9.0:
            return 'yellow'
        if height_km < 12.0:
            return 'cyan'
        return '#a0a0a0'
    
    def _plot_hodograph(self) -> None:
        self._hodograph_ax.cla()
        self._hodograph_artist = []
        self._init_hodograph_axes()
        
        use_storm_relative = (
            self._hodograph_mode == 'Storm-Relative' and bool(self._storm_relative_profile)
        )
        profile = self._storm_relative_profile if use_storm_relative else self._hodograph_profile
        
        heights = profile.get('height_m') if profile else None
        u_comp = profile.get('u_kt') if profile else None
        v_comp = profile.get('v_kt') if profile else None
        
        if heights is None or u_comp is None or v_comp is None or heights.size < 2:
            self._hodograph_ax.text(
                0.0,
                0.0,
                'No hodograph data',
                color='white',
                fontsize=10,
                ha='center',
                va='center',
            )
            self.figure.canvas.draw_idle()
            return
        
        if self._hodograph_mode == 'Storm-Relative' and not use_storm_relative:
            self._hodograph_ax.text(
                0.0,
                0.0,
                'Storm motion unavailable\nShowing ground-relative winds',
                color='white',
                fontsize=9,
                ha='center',
                va='center',
            )
        
        for idx in range(u_comp.size - 1):
            layer_height = 0.5 * (heights[idx] + heights[idx + 1])
            color = self._hodograph_layer_color(layer_height)
            segment = self._hodograph_ax.plot(
                u_comp[idx:idx + 2],
                v_comp[idx:idx + 2],
                color=color,
                linewidth=2.4,
                zorder=3,
            )[0]
            self._hodograph_artist.append(segment)
        
        self._plot_storm_motion_markers(use_storm_relative)
        self._plot_effective_inflow_layer(use_storm_relative)
        self._plot_low_level_shear()
        
        title_mode = 'Storm-Relative' if self._hodograph_mode == 'Storm-Relative' else 'Ground-Relative'
        self._hodograph_ax.set_title(
            f'Hodograph ({title_mode})',
            color='white',
            fontsize=12,
            pad=10,
        )
        self._hodograph_ax.set_xlim(-self._hodograph_range_ms, self._hodograph_range_ms)
        self._hodograph_ax.set_ylim(-self._hodograph_range_ms, self._hodograph_range_ms)
        
        self.figure.canvas.draw_idle()
    
    def _plot_storm_motion_markers(self, use_storm_relative: bool) -> None:
        if not self._storm_motion:
            return
        
        rm_u, rm_v = self._storm_motion.get('RM', (np.nan, np.nan))
        reference_u = rm_u if use_storm_relative else 0.0
        reference_v = rm_v if use_storm_relative else 0.0
        ms_to_kt = 1.94384
        
        colors = {'RM': 'white', 'MW': 'orange', 'LM': 'white'}
        for key, color in colors.items():
            motion_u, motion_v = self._storm_motion.get(key, (np.nan, np.nan))
            if not (np.isfinite(motion_u) and np.isfinite(motion_v)):
                continue
            plot_u = (motion_u - reference_u) * ms_to_kt 
            plot_v = (motion_v - reference_v) * ms_to_kt
            marker = self._hodograph_ax.scatter(
                plot_u, plot_v, color=color, edgecolor='black', s=55, zorder=5
            )
            label = self._hodograph_ax.text(
                plot_u + 2.0,
                plot_v,
                key,
                color=color,
                fontsize=9,
                va='center',
                ha='left',
                bbox={
                    'facecolor': 'black',
                    'edgecolor': 'none',
                    'alpha': 0.1,
                    'pad': 1.5,
                },
                zorder=6,
            )
            self._hodograph_artist.extend([marker, label])
    
    def _plot_effective_inflow_layer(self, use_storm_relative: bool) -> None:
        base_height, top_height = self._effective_inflow_layer
        rm_u, rm_v = self._storm_motion.get('RM', (np.nan, np.nan))
        if not (
            np.isfinite(base_height)
            and np.isfinite(top_height)
            and np.isfinite(rm_u)
            and np.isfinite(rm_v)
        ):
            return
        
        base_u, base_v = self._components_at_height(base_height)
        top_u, top_v = self._components_at_height(top_height)
        if not np.all(np.isfinite([base_u, base_v, top_u, top_v])):
            return
        
        ref_u = rm_u if use_storm_relative else 0.0
        ref_v = rm_v if use_storm_relative else 0.0
        ms_to_kt = 1.94384
        points_u = [
            (base_u - ref_u) * ms_to_kt,
            (rm_u - ref_u) * ms_to_kt,
            (top_u - ref_u) * ms_to_kt,
        ]
        points_v = [
            (base_v - ref_v) * ms_to_kt,
            (rm_v - ref_v) * ms_to_kt,
            (top_v - ref_v) * ms_to_kt,
        ]
        line = self._hodograph_ax.plot(points_u, points_v, color='cyan', linewidth=1.5, zorder=4)[0]
        self._hodograph_artist.append(line)
    
    def _plot_low_level_shear(self) -> None:
        shear_u_ms, shear_v_ms = self._low_level_shear
        base_height, _ = self._effective_inflow_layer
        
        if not (np.isfinite(base_height) and np.isfinite(shear_u_ms) and np.isfinite(shear_v_ms)):
            return
        
        use_storm_relative = (
            self._hodograph_mode == 'Storm-Relative' and bool(self._storm_relative_profile)
        )
        base_u_ms, base_v_ms = self._components_at_height(base_height, storm_relative=use_storm_relative)
        if not np.all(np.isfinite([base_u_ms, base_v_ms])):
            return
        
        ms_to_kt = 1.94384
        start_u = base_u_ms * ms_to_kt
        start_v = base_v_ms * ms_to_kt
        end_u = start_u + shear_u_ms * ms_to_kt
        end_v = start_v + shear_v_ms * ms_to_kt
        
        line = self._hodograph_ax.plot(
            [start_u, end_u], [start_v, end_v], color='blue', linewidth=0.5, zorder=4
        )[0]
        self._hodograph_artist.append(line)
    
    def _on_hodograph_mode_changed(self, mode: str) -> None:
        self._hodograph_mode = mode
        if self._hodograph_profile:
            self._plot_hodograph()