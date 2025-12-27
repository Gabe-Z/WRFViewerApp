from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MultipleLocator
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QMainWindow, QVBoxLayout, QWidget

from calc import (
    sounding_isotherm_temperatures,
    sounding_pressure_bounds,
    sounding_pressure_levels,
    sounding_skewed_isotherm,
    sounding_temperature_bounds,
    sounding_temperature_ticks,
)


class SoundingWindow(QMainWindow):
    def __init__(self, timestamp: str, latitude: float, longitude: float, parent=None):
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
        
        self._draw_background()
    
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
        aspect_correction = height / width if width > 0 else 1.0
        
        self.ax.set_xlim(temp_min, temp_max)
        self.ax.set_ylim(pressure_bottom, pressure_top)
        self.ax.set_yticks(pressure_levels)
        self.ax.set_xticks(temp_ticks)
        
        self.ax.set_xticklabels([f'{temp:.0f}' for temp in temp_ticks], color='white')
        self.ax.set_yticklabels([f'{level:.0f}' for level in pressure_levels], color='white')
        
        for level in pressure_levels:
            self.ax.axhline(level, color='white', linewidth=1.0)
        
        pressures = np.linspace(pressure_bottom, pressure_top, num=64, dtype=float)
        for temp in isotherm_temps:
            skewed_temps = sounding_skewed_isotherm(
                temp, pressures, aspect_correction=aspect_correction
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