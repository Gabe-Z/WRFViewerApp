# To-Do:
	- [ ] (High Priority) Add severe weather composite such as stp and etc.
	- [ ] Add Surface Vorticity
	- [ ] More Upper Air plots
	- [ ] (Low Priority) Fix mercator projection resulting in calculating UH threshold to be off.
	- [ ] Add PW, DCAPE, 3CAPE to the Thermodyanmic Indices.
	- [ ] Plot effective inflow layer onto the Skew-T chart.
	- [ ] And much more.

# Version: 0.2.1 Alpha
## Added:
	- [x] Added more upper air products (Toaster is happy!)
	- [x] Added Bulk Shear (0-1 km, 0-3 km, & 0-6 km)
	- [x] Added SRH (0-1 km & 0-3 km)
	- [x] Added Updraft Helicity 0-3 km AGL (Run Max)
	- [x] Added Updraft Helicity 2-5 km AGL (1 h Max, 3 h Max, Run Max)

## Fixes:
	- [x] Fixed wind data in upper air products.
	
## Version: 0.2.0 Alpha
### Added:
	- [x] Most Unstable CAPE variable.
	- [x] Mixed-Layer CAPE variable.
	- [x] 0-3 km AGL CAPE variable.
	- [x] County lines and primary/secondary networks in US.
	- [x] OLR variable.
	- [x] IR simulated satellite.

### Changes:
	- [x] 500 mb Height, Wind now shows the Wind Speed plot instead of Geopotential Height plot.
	- [x] Improve Image Export
	- [x] Changes the inline value label/barbs strides to 100 to prevent them lagging too much.
	- [x] Added domain resolution info to the info/time label.
	- [x] Change the geopotential height from meter to decameter.

### Fixes:
	- [x] Fixed the plot/colorbar positions.
	- [x] Fixed the figure displacement on Export Image.
	- [x] Change info/time label lower to prevent from label overlapping the plot.

## Version: 0.1.1 Alpha
### Added:

### Changes:
	- [x] Change info/time label to center below the plot rather than lower-left.
### Fixes:
	- [x] Fix Video Export: [WinError 2] The system cannot find the file specified.
	- [x] When opening wrfout files, make it able to include network like wsl files. Since using Window system (.exe) cause network files to disappear.



## Version: 0.1.0 Alpha
### Added:
	- [x] Export Video.
	- [x] Plot Hodograph with Storm Motions Vectors.

### Changes:

### Fixes:
	- [x] Fix overlapping uh tracks and hover annotations.