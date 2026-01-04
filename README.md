# WRF Viewer App

## License & Usage Terms (2026)

This project is **Source Available**. By accessing this repository, you agree to the following terms:

*   ✅ **Forking:** You may fork this repository on GitHub as permitted by GitHub 's Terms of Service for personal study or private experimentation.
*   ✅ **Private Modification:** You are permitted to modify the source code for your own **private, personal use** only.
*   ❌ **No Commercial Use:** You are strictly prohibited from using this software, its source code, or any derivatives for any commercial purpose, profit-seeking activity, or business use.
*   ❌ **No Redistribution:** You may **not** re-upload, publish, or distribute the original or modified source code to any public platform, app store, or website outside of your private GitHub fork.

For the full legal text, see the [LICENSE](LICENSE.md) file.

# README

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