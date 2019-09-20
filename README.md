# bns_ppr_tools
Set of scripts and methods for WhyskyTHC output postprocessing 

## Dependencies:
`python 2.7.xx` with `scipy`, `numpy`, `cPickle`, `itertools`, `h5py`, `csv`, `mpl_toolkits`, `matplotlib`, `statsmodels`, `pandas`, `math`, `click`, `re`, `argparse`  
`scidata` that can be found at https://bitbucket.org/dradice/scidata/src/default/


# Running pipeline:  
cd `/bns_ppr_tools/`   
`./analyze.sh simulation_dir_name /path_to_this_dir/ /path_to_output/`  
example:  
`./analyze.sh SLy4_M13641364_M0_SR /home/myname/simulations/ /home/myname/postprocessing/`    
<mark>Note: inside the output directory a directory with the simulation name will be created and results will be put into it</mark>  
  
## preprocessing.py
purpose: check and show the available data and timespans. Create an `ittime.h5` file that contains the information about timestaps and iterations of different data types, such as ascii files, .xy.h5 files and parfile.h5 files.  
The ittime.h5 is essential for all other methods, as they do not have to scan for available data every time.  
Options for this script:  
-s simulation_dir_name
-i /path_to_this_dir/  
-o /path_to_output/  
-t task to perform, such as: `update_status` or `print_status`  

## outflowed.py
purpose: 
1) parse set of `outflow_surface_det_0_fluxdens.asc` files from every output-xxxx into a singular .h5 file with the same data, just reshaped onto a spherical grid with `n_theta` and `n_phi` parameters of the grid.  
To do that for data from detector (-d), run  
`python outflowed.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ --eos /path/to/hydro_eos_file.h5 -t reshape -d 0`  
2) do the comprehensive, easily extendeble analysis of this data. Available standart methods (-t otion). For that mask option (-m) has to set. For example: `-t geo` would stand for geodesic criteria to unbound material.  
-t all (to do all the below mention tasks one after another)
-t hist (creates and plot histograms of variables that speified with option -v, like -v Y_e theta vel_inf )
-t corr (creates and plots correlations (2D histograms) of pairs of variables that speified with option -v, like -v Y_e theta vel_inf theta)  
-t totflux (creates and plots total flux of the ejecta  
-t massave (creates the mass averaged quantities)  
-t ejtau (computes the 3D histogram, with Y_e, entropy and expansion timescale as axis)  
-t yeilds (computes and plots nuclesynthetic yeilds)  
-t mknprof (computes and plots angular profile of mass, Y_e and vel_inf for macrokilonova bayes code)  
Example:
`python outflowed.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ --eos /path/to/hydro_eos_file.h5 -t all -m geo -d 0 --overwrite yes`  
would perform all (-t all) the default analysis methods, for geodeiscally unbound materai (-m geo) for detector 0 (-d 0) and if the results are already present, it will overwrite them (--overwrite yes)   

# slice.py

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

2) To make 2D movie of the variable evolution.
For that use task `-t movie`
Example:  
`python slices.py -s simulation_name -i /path_to_this_dir/ -o /path_to_output/ -t movie --v_n rho --rl 3 --time all --overwrite yes`  
This will plot rest mass density (--v_n rho) for reflevel (--rl 3) for all the timesteps available (--time all) overwriting existing plots (--overwrite yes)  
After all plots are made, movies will be created, using `ffmpeg`.  
These plots and the final movie will be located in:  
`/path_to_output/simulation_name/slice/movie/rho/rl_3/rho_rl3.mp4`  
To redo the movie without recomputing all the plots, -- remove the `rho_rl3.mp4` and relaunch the script with `--overwrite no` flag.  

As movie creation takes a considerable time (for long simulations) this is not a part of a pipeline. To be run separately.  

    
