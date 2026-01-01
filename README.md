# WRF Viewer App

Currently a work in progress and not intended for public uses. You will need to install the runtime libraries the scripts requires before running them.

For easier way to download without compiling:
	Go to Download Folder
	WARNING: The file size is 238 mb due to all of the libraries and packages being included with it.
	Also include Colortable folder in the same path as WRF_Viewer_App.exe.

## Required Python packages
The scripts import the following packages:
`numpy`
`netCDF4`
`wrf-python`
`matplotlib`
`cartopy`
`PySide6`
NEW: `imageio`
NEW: `imageio[ffmpeg]`

## QT Error
If you see this error: `Could not find the Qt platform plugin "xcb"`:

Try (Depending which system you use):
```bash
export QT_QPA_PLATFORM_PLUGIN_PATH="{Environment Path}/lib/qt6/plugins"
or
setenv QT_QPA_PLATFORM_PLUGIN_PATH="{Environment Path}/lib/qt6/plugins"
or
set QT_QPA_PLATFORM=windows
```