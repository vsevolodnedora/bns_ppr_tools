# bns_ppr_tools

Set of tools and methods to postprocess standard output of the 
`WhiskyTHC` GRHD code for binary neturon star mergers.

## Overview: 
The follwing modules are available:
- `module_ejecta` contains scripts to analyze data extracted by `HYDRO` thorn of `Whsiky` at a given extraction radius.
All methods are semi-independent and require only `ouflowed_det_N.dat` ascii files for each output of the simulation.
- `module_gw` contains methods to analyze `mp_Psi4_lN_mM_rXXX.asc` multipolar modes. Using `scidata` routines 
it computes the `strain` and `waveforms/` as well as Grav.Wave energy and angular momentum losses.  
- `module_preanalysis` used to create `ittime.h5` files that collects information about what data is available 
for the simulation: which ascii files, 2D .h5 files and profiles, and for what timesteps and iterations. 
This data is then used by other modules. It can also `collate` ascii files.
- `module_profile` contains methods to analyze 3D profiles of the simulation. It uses `scidata` to reconstract 
the cartesian grid of the simulation.
- `module_slices` contains methods to analyze `variable_xy.h5` 2D slices of the simulation. Also uses `scidata` to
reconstract the cartesian grid.
- `plotting` contains methods for plotting
- `standalone` contains scripts that do not require `ittime.h5` can can be run independently. 
- `pipeline` contains a set of wrappers for above mentioned methods that allow to launch them one after another to 
perform a complete analysis of the available data.



## Dependencies:
`python 2.7.xx` (incl. `scipy`, `numpy`, `itertools`, `h5py`, `csv`, `matplotlib`, `re`, `argparse`)
`scidata` for `python 2.7.xx` that can be found at [scidata](https://bitbucket.org/dradice/scidata/src/default/)



# Suggested Simulation Setup

## Suggested Setup for `pipeline`:

**Example**: 
Consider a simulation named `Sly4_M125125_M0_SR/`. Inside, there are 
`WhiskyTHC` `output-xxxx/data/` directories with ascii files and .h5 files.
Profiles are stored in `profiles/3d/` and named according to iterations, e.g.,
`12345.h5`.

In the file `config.py` default paths can be set.  
The path to this directory is given to variable `default_data_dir`, 
e.g., default_data_dir = '/home/MySimulations/Sly4_M125125_M0_SR/'.  
The path to where to put the results in given as `default_ppr_dir` variable, e.g, 
default_ppr_dir = '/home/MyPostProcessing/Sly4_M125125_M0_SR/'



## Limiting the postprocessing with maximum time.

In case postprocessing for a given simulation has to be limited to a given time, 
a user can create file `maxtime.txt` insider the `default_data_dir` with a single float: time in **ms**. 

In order to limit the analysis of ejecta, the `outflowed.py` from the `pipeline` runs the script 
`outflowed.py` with option `-t reshape` to create new `outflow_surface_dens.h5` file that has data limited to that time. 
Other methods, e.g., `preanalysis/` with `-t collate` and `profile` will see the `maxtime.txt` and 
adjust the workload automatically



## Running pipeline with bash script

First. Set all paths in `config.py`.

then run the main script is `pipeline/analyse.sh Sly4_M125125_M0_SR`, where the passed argument is the 
name of the simulation directory that should be present in `default_data_dir/` and in `default_ppr_dir/`.

The following sequence of analysis will being:
1. `preanalysis.py -s $1 -t update_status print_status`  that will create `ittime.h5` for a given simulation ($1) 
with info in available data.
2. `python preanalysis.py -s $1 -t collate --overwrite yes` that will collate ascii files (necessary for GW) analysis and ejecta
3. `python gw.py -t strain tmergtcoll -s $1 --overwrite yes` that will compute strain and waveforms from the collated data
4. `python ejecta.py -s $1 -t reshape -d 0 1 -p 8 --overwrite yes` that will create `.h5` files for the 
set of ascii files that HYDRO torn outputs into every `output-xxxx` (The .h5 files are more compact and more versitile)
5. `python ejecta.py -s $1 -t all -d 0 1 -m all --v_n all --overwrite yes` will perform the complete postprocessing of 
   the ejecta using these `.h5` files as an input.
6. `python profile.py -s $1 -t all --it all --mask disk remnant --plane all --v_n all --overwrite yes` will 
perform the analysis of all profiles (named `12345.h5`) assinging them timesteps according to data from `ittime.h5`.


## Detailed option list (Warning! Might not be up-to-date)

### preprocessing.py
purpose: check and show the available data and timespans. Create an `ittime.h5` file that contains the information about timestaps and iterations of different data types, such as ascii files, .xy.h5 files and parfile.h5 files.  
The `ittime.h5` file is **essential** for all other methods, as they do not have to scan for available data every time.  
Options for this script:  
`-s` simulation_dir_name
`-i` /path_to_this_dir/  
`-o` /path_to_output/  
`-t` task to perform, such as: `update_status` or `print_status`, `collate`,  
`--overwrite` replaces the old results if set to `yes` and does not if set to `no`
`--usemaxtime` used only for the task `-t collate`. Essentially it limits the time up to which to collate data.
where the last task allows to collate certain ascii files, removing the repetitions.


### outflowed.py

Requirements:  
`ittime.h5` file, created by `preanalysis.py` (see above)  

Purpose and usage: 
1) parse set of `outflow_surface_det_0_fluxdens.asc` files from every output-xxxx into a singular .h5 file with the same data, just reshaped onto a spherical grid with `n_theta` and `n_phi` parameters of the grid.  
To do that for data from detector (-d), run  
`python outflowed.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ --eos /path/to/hydro_eos_file.h5 -t reshape -d 0`  

**Note** Running the `-t reshape` (the longest part of the outflow analysis) on a multiprocessor 
system can be done in parallel. For that specify `-p 4` option, setting the number to a number of processors to use. Here
the user can limit, up to what time to postprecess the data. If the maxtime attribute is present in `ittime.h5` by setting 
`--usemaxtime auto` it will be picked up and the analysis will be limited. Another option is to set the value manually here as 
`--usemaxtime 20` where 20 is the 20 ms time, up to which to postprocess. If no limits needed, the option can me set to 
`--usemaxtime no` (which is the default)  

2) do the comprehensive, easily extendeble analysis of this data. Available standart methods (-t otion). For that mask option (-m) has to set. For example: `-t geo` would stand for geodesic criteria to unbound material.  

*Available masks:*  
`-m geo` apply only geodesic criterion: einf >= set_min_eninf, which is 0 by default  
`-m geo_entropy_below_10` is a `-m geo` with additional criterion for entropy to be below 10  
`-m geo_entropy_above_10` is a `-m geo` with additional criterion for entropy to be above 10  
`-m bern` apply bernoulli critrion ((enthalpy * (einf + 1) - 1) > set_min_eninf) & (enthalpy >= set_min_enthalpy), where set_min_eninf is by default 0. and set_min_enthalpy is a minimum enthalpy of the atmosphere. 1.0022 by default.  
`-m bern_geoend` is a `-m bern` with an additional criterion for total cumulative ejected mass (using `-m geo`)  to reach 98% of its maximum value. Basically, it would compute ejecta after dynamical one has saturated.  
`-m Y_e04_geoend` is a `-m bern_geoend` with and additional criterion on electron fraction to be above 0.4.  
`-m theta60_geoend` is a `-m bern_geoend` with and additional criterion on the angle from binary plane to be above 60 degrees.    

*Available tasks:*  
`-t all` to do all the below mention tasks one after another  
`-t hist` creates and plot histograms of variables that speified with option -v, like --v_n Y_e theta vel_inf  
`-t timecorr` creates a correlation from set of histograms for different timesteps. Usefull to see the evolution  
`-t corr` creates and plots correlations (2D histograms) of pairs of variables that speified with option -v, like -v Y_e theta vel_inf theta  
`-t totflux` creates and plots total flux of the ejecta  
`-t massave` creates the mass averaged quantities  
`-t summary` creates the summary.txt files with total fluxes and mass-average quantities of the ejecta  
`-t ejtau` computes the 3D histogram, with Y_e, entropy and expansion timescale as axis  
`-t yeilds` computes and plots nuclesynthetic yeilds  
`-t mknprof` computes and plots angular profile of mass, Y_e and vel_inf for macrokilonova bayes code  

*Additional options*  
`-p` only valid for task `-t reshape`, for parallel processing of the input ourflowed files. Speeds up the analysis.
`--maxtime` only valid for task `-t reshape` with parallel option `-p n`, where n is a integer, number 
of processors to use. It allows to limit, up to what time to postprocess the data. If the maxtime attribute is present in `ittime.h5` by setting 
`--usemaxtime auto` it will be picked up and the analysis will be limited. Another option is to set the value manually here as 
`--usemaxtime 20` where 20 is the 20 ms time, up to which to postprocess. If no limits needed, the option can me set to 
`--usemaxtime no` (which is the default)



### slice.py

Requirements:  
`ittime.h5` file, created by `preanalysis.py` (see above)  

Purpose and usage:
1) To load and easily plot `var_name.xy.h5` and `var_name.xz.h5` data. as a 2 attached plots, with `xz` ont top and `xy` square plot in the bottom.  
For that use task: `-t plot`.   
Exaple:  
`python slices.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ -t plot --v_n rho --rl 3 --time 20 --overwrite yes`  
This will plot rest mass density (--v_n rho) for reflevel (--rl 3) and time (--time 20) millisecond, overwriting existing plot (--overwrite yes)  
The plot will be located in:  
`/path_to_output/simulation_name/slice/plot/rho/rl_3/00000.png`  
with the name of the plot, corresponding to iteration for the plotted timestep.  
Data to use can be specified with either moments in time `--time` or iterations `--it`  

2) To make 2D movie of the variable evolution.
For that use task `-t movie`
Example:  
`python slices.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ -t movie --v_n rho --rl 3 --time all --overwrite yes`  
This will plot rest mass density (--v_n rho) for reflevel (--rl 3) for all the timesteps available (--time all) overwriting existing plots (--overwrite yes)  
After all plots are made, movies will be created, using `ffmpeg`.  
These plots and the final movie will be located in:  
`/path_to_output/simulation_name/slice/movie/rho/rl_3/rho_rl3.mp4`  
To redo the movie without recomputing all the plots, -- remove the `rho_rl3.mp4` and relaunch the script with `--overwrite no` flag.  

3) Other:  
`-t` addm0 adds m0 quantities (such as Q_eff, R_eff) if available to the `profile.xy/xz.h5` (see profile.py for descriptions) 
the profile.xy/xz.h5 are expected to be in `/path_to_sim_dir/profiles/123456/`, where 12345 is the directory name corresponding to the profile iteration  

As movie creation takes a considerable time (for long simulations) this is not a part of a pipeline. To be run separately.  

### makeprofile.py 

Purpose and usage:
This is a stand alone tool for converting 3D .h5  data (that is usually 
saved as `var_name.file_0.h5`, where number stands for a processor, 
on from which this file was dumped) into a single profile.h5 file 
that contains several variable data that has been mapped onto a unique 
grid for every reflevel using `scidata`.  
Two types of profiles can be created as of now. 
`-t` prof    (task for a hydro profile)
`-t` nuprof  (task for a neutrino M0 profile)  

First a directory `profiles/` will be created inside the output directory (-o)  
Then, the data for every variable will be loaded and saved as a unique `variable_name.h5` file. This is done to
avoind overloading `h5py`. Then these files would be loaded and a unique `parfile.h5` will be created with 
all the data and grid parameters.  
**Note** that to evaluate internal energy and pressure, the EOS is used, provided Ye, rho and temperature from simulation. However,
if one of these quantities is out of limits for EOS the closes EOS value is used. No extrapolation is done.    

For neutrino M0 profile, only the M0 variables are used from the 3D output. 
**Note** that M0 data is not evolved on the same grid as hydrodynamic variables. There is only one refinemnet level and the grid 
is spherical with `nrad` `nphi` and `ntheta` being its parameters. This grid is saved in `profilenu.h5` as well.  

Example:  
`python makeprofile.py -i path_to_inside_of_simulation_dir --eos path_to_hydro_eos.h5 -o same -t prof nuprof -m times --time 90`  
This will create `parfile/` directory inside of the one given in (-o), 
Here the output dir (-o) is set to be the same as input (-i). The mode is (-m times) is set, so the data will be extracted for given timesteps.
One by one `variable.h5` will be saved and then both (-t prof nuprof) will be saved as `123456.h5` and `123456nu.h5`
where 123456 would be the iteration, closest to the required time (--time 90) ms. 

**Note** that overall, this is a lengthy procedure and henceforth is not a part of a pipeline.  
Which profiles to extract and analyze is up to the user.  

### profile.py

Requirements: 
1) ittime.h5 file, created by preanalysis.py (see above)
2) Extracted profiles, located in `/path/to/simulation_dir/profiles/3d/`

Purpose and usage:  
To do a comprehensive analysis of the 3D data. It allows to compute: 
1) Compute quntities such as `dens_unb_bern` or `ang_mom_flux` with methods specified in the class `FORMULAS`  
2) 1D histograms, 2D correlations (histograms), total mass, (applying user specified masks)
3) Plot xz-xy snapshots of the data, initially available as well as computed.

By default the mask is applied to 3D data, to ignore the atmosphere (density cut) and remnant (density cut) and black hole  
(lapse cut). The exact parameters of the mask are set in the class 'MASK_STORE'. There, additional masks are avaialble, such as
'disk' -- for disk analysis and 'remnant' for remnant, NS, analysis. Additional masks can be specified there, and the entire
postprocessing re-done for selected data [this is not implemented at present].

Parameters:  
`-t` tasklist to do.  
`-i` path to the simulation dir (for example `/home/myname/mysimulations/`)  
`-s` simulation dir (for example: `LS220_M130130_SR`)  
`--v_n` list of variable names or their combinations (for correlation task)
`--rl` list of refinement levels to use  
`--time` list of timesteps to use  
`--it` list of iterations to use  
`--overwrite` flag to overwrite data if already exists.  
`--usemaxtime` flag/value to limit the usage of data (see description in `outflowed.py`)

Example:  
`python profile.py -s LS220_130130 -i /home/my_simulations/ -o /home/my_postprocessing/ 
-t all --it all`  
This would perform the complete analysis for every `profile.h5`. For every profile, it would create a separate output subdirectory in the root postrpocessing directory (-o), named with the iteration of this profile
1) `-t densmode` compute density modes for default 1-8 modes accounting for center of mass drift, saving output in the root as `density_modes_lap15`.
2) `-t slice` computes additional variables and saves their xy and xz slices in `profile.xy.h5` and `profile.xz.h5` in the in the aforementioned subdirectories.
3) `-t corr` computes correlations for all available variables, saving the `corr_v_n1_v_n2.h5` files in the aforementioned subdirectories.  
4) `-t mass` computes mass of the disk using the present mask for the disk, also computes the remnant baryonic mass, assuming that everything that has higher density than 1e13 is a remnant.  
5) `-t hist` computes histograms for some variables, saving the `hist_v_n.dat` files in the aforementioned subdirectories.  

**Note** for `-t corr` and `-t hist`, the 'disk' mask for all data is used. Default is lapse>0.15 and 6e4<rho<1e13 (cgs)

There are also analysis methods that rely on interpolated data implemented:  
Using 'cylindrical grid', set up in a class 'CYLINDRICAL_GRID', the following tasks are performed:
1) `-t mjenclosed` computes baryonic, angular momentum and moment of inertia enclosed in cylindrical shells.

Using the 'cartesian grid', set up in class 'CARTESIAN_GRID', the following tasks are performed
1) `-t vtk` computes a .vtr file for visit visualisation. The data for given `v_n` and `it` (or `time`) is first inteprolated onto a cartesian grid and then parsed into `gridToVTK()` function from PyEVTK library. Requires preinstallation of `https://bitbucket.org/pauloh/pyevtk`. 


6) `-t plotslice` loads `profile.xy.h5`, `profile.xz.h5` and for every reflevel and variable plots xz-xy 2D slice, saving in `/slices/`
7) `-t plotcorr` loads computed  `corr_v_n1_v_n2.h5` and plots data, saving in `/corr_plots/`
8) `-t plothist` loads computed  `hist_v_n.dat` and plots data, saving in `/hist_plots/`
9) `-t plotdensmode` loads computed  `density_modes_lap15` and plots data, saving in root. 

Any of these tasks can be peroformed for one or a list of:  
1) iterations by specifying `--it` option **or** timesteps by specifying `--time` option
2) reflevels by specifying `--rl` option (will not affect `corr`, `mass`, `hist` tasks, as they use the entire simulation domain by default.
3) variable names by specifying `--v_n` option. 

Known issues:
- making many plots, e.g. setting `--it all` and `--v_n all` for a task `-t plotcorr`, might cause some images to be emtpy.
Cause: overload of matplotlib.pyplot cash. Required plot can be redone with the code separately. This corrects the problem. 

### gw.py

Requirements: 
1) ittime.h5 file, created by `preanalysis.py` (see above)  
2) collated data, created by `preanalysis.py` (see above)

**Note**  
This part of the data analysis is incomplete as I am not a specialist in this area. 

Purpose and usage:  
1) to do a rudimentary, zero-order analysis of the Psi4 data and to obtain a waveform, wfrom which 
the time of the merger can be deduced. Similarly, time of the collapse to a BH can be obtained but this 
is not as reliable, as the magnitude of the strain can go to almost zero in between remnant osciallation. 

Example:  
`python gw.py -s LS220_130130 -i /home/my_simulations/ -o /home/my_postprocessing/ -t all`  
this would create a `/waveforms/` subdirectory inside the `/home/my_postprocessing/LS220_130130/` 
and put there the following:  
`-t strain` computes the strain and some basing properties of the radiation reaction.   
`-t tmergtcoll` plots the waveform alongside the collated density and makes an estimate of the time of the 
merger and the time of the collapse (if occures).  
**Note** that user inspection of the produced summory plot is required to determine if the time of the collapse 
was estimated properly. 