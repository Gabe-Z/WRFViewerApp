#!/usr/bin/env python3

'''
WRF Viewer - Single file desktop app (PySide6 + Matplotlib + Cartopy)

Features (MVP)
    - Open one or many wrfout_* NetCDF files (concatenated in time order).
    - Variables: MDBZ, RAINNC, RAINC, WSPD10, REFL1KM.
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
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import typing as T

from dataclasses import dataclass
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.colors import LinearSegmentedColormap
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
    
class WRFLoader(QtCore.QObject):
    '''
    Loads wrfout files, make a timeline of frames, and provide 2D fields.
    Supports optional full preloading for smooth scrubbing.
    '''
    def __init__(self):
        super().__init__()
        self.files: list[str] = []
        self.frames: list[WRFFrame] = []
        self._cache: dict[tuple[str, str, int], np.ndarray] = {}
        self._geo_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Preload store: {(VAR, level_or_None): [np.ndarray | None per frame]}
        self.preloaded: dict[tuple[str, T.Optional[float]], list[T.Optional[np.ndarray]]] = {}
    
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
                    frames.append(WRFFrame(path=fp, time_index=ti, timestamp_str=ts))
        frames.sort(key=lambda fr: (os.path.getmtime(fr.path), fr.time_index))
        self.frames = frames
        self._cache.clear()
        self._geo_cache.clear()
        self.preloaded.clear()
        
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
    
    # --- Preload API (helpers) ---
    def allocate_preload(self, var: str, level_hpa: T.Optional[float]) -> None:
        key = (var.upper(), float(level_hpa) if level_hpa is not None else None)
        if key not in self.preloaded:
            self.preloaded[key] = [None] * len(self.frames)
            
    def set_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int, data: np.ndarray) -> None:
        key = (var.upper(), float(level_hpa) if level_hpa is not None else None)
        self.preloaded.setdefault(key, [None] * len(self.frames))
        self.preloaded[key][idx] = data
    
    def get_preloaded(self, var: str, level_hpa: T.Optional[float], idx: int) -> T.Optional[np.ndarray]:
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
                ('MDBZ', 'MDBZ'),
                ('RAINNC', 'RAINNC'),
                ('RAINC', 'RAINC'),
                ('WIND10', 'WSPD10'),
            ],
            'Severe': [
                ('MDBZ', 'MDBZ'),
                ('REFL1KM', 'REFL1KM'),
            ],
            'Upper Air': [
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
        
        controls.addWidget(QLabel('Variable:'))
        self.cmb_var = QComboBox()
        seen_labels: set[str] = set()
        self._variable_labels: list[str] = []
        for items in self.var_categories.values():
            for label, _ in items:
                if label not in seen_labels:
                    seen_labels.add(label)
                    self._variable_labels.append(label)
                    self.cmb_var.addItem(label)
        self.cmb_var.currentTextChanged.connect(self.on_var_changed)
        controls.addWidget(self.cmb_var)
        
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
        
        # Basemap Features
        self.ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        self.ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4)
        self.ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.6)

        # Sync default selection with accordion
        current_label = self.cmb_var.currentText()
        if current_label:
            self._sync_category_selection(current_label)
    
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
        current = self.cmb_var.currentText()
        if current == label:
            # Ensure downstream actions happen even if combo box doesn't emit
            self.on_var_changed(label)
            return
        idx = self.cmb_var.findText(label)
        if idx == -1:
            self.cmb_var.addItem(label)
            idx = self.cmb_var.count() - 1
        self.cmb_var.setCurrentIndex(idx)

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
        display_var = self.cmb_var.currentText()
        var = self._canonical_var(display_var)

        # show inline progress bar
        self.pb.setVisible(True)
        self.pb.setValue(0)
        self.status.showMessage(f'Preloading {display_var}')

        self.worker_thread = QtCore.QThread(self)
        self.worker = PreloadWorker(self.loader, var, None)
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
        
        display_var = self.cmb_var.currentText()
        var = self._canonical_var(display_var)
        level = None

        data = self.loader.get_preloaded(var, None, idx)
        if data is None:
            try:
                data = self.loader.get2d(frame, var, None)
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
        
        data = np.squeeze(data)
        if data.shape != lat.shape:
            ny, nx = lat.shape
            data = data[:ny, :nx]
        
        vmin, vmax, label = self._default_range(var)
        
        # Create once /then resure for speed
        if self._img_art is None or self._img_shape != data.shape:
            if self._img_art is not None:
                try:
                    self._img_art.remove()
                except Exception:
                    pass
            
            self._img_art = self.ax.pcolormesh(
                lon, lat, data,
                transform=ccrs.PlateCarree(),
                cmap=self.current_cmap,
                shading='nearest',
                vmin=vmin, vmax=vmax,
                antialiased=False,
                rasterized=True,
            )
            self._img_shape = data.shape
            
            if self._cbar is None:
                self._cbar = self.fig.colorbar(self._img_art, ax=self.ax, orientation='vertical', shrink=0.8, pad=0.02)
            self._cbar.set_label(label)
        else:
            # Update only face colors and limits
            # pcolormesh set_array expects one value per face; averaging corners is okay for speed.
            self._img_art.set_array(np.asarray(data).ravel())
            self._img_art.set_cmap(self.current_cmap)
            if vmin is not None and vmax is not None:
                self._img_art.set_clim(vmin, vmax)
            self._cbar.update_normal(self._img_art)
            self._cbar.set_label(label)
            
        self.ax.set_title(self._title_text(display_var, var), loc='center', fontsize=12, fontweight='bold')
        self.canvas.draw_idle()
        
    def _default_range(self, var: str) -> tuple[T.Optional[float], T.Optional[float], str]:
        v = var.upper()
        if v in ('MDBZ', 'MAXDBZ'):
            return 0.0, 70.0, 'Reflectivity (dBZ)'
        if v in ('RAINNC', 'RAINC'):
            return 0.0, None, 'Accumulated Precip (mm)'
        if v in ('WSPD10', 'WIND10'):
            return 0.0, 40.0, '10-m wind speed (m s$^{-1}$)'
        if v == 'REFL1KM':
            return 0.0, 70.0, 'Reflectivity @ 1km AGL (dBZ)'
        return None, None, var
    
    def _title_text(self, display_var: str, canonical_var: str) -> str:
        v = canonical_var.upper()
        if v in ('MDBZ', 'MAXDBZ'):
            return 'MDBZ (Column Max dBZ)'
        if v == 'REFL1KM':
            return 'Reflectivity @ 1 km AGL (dBZ)'
        return display_var
        
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