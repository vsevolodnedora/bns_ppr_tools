# bns_ppr_tools
Set of scripts and methods for WhyskyTHC output postprocessing 

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
2) do the comprehensive, easily extendeble analysis of this data. Available standart methods (-t otion):
-t all (to do all the below mention tasks one after another)
-t hist (creates and plot histograms of variables that speified with option -v, like -v Y_e theta vel_inf )
-t corr (creates and plots correlations (2D histograms) of pairs of variables that speified with option -v, like -v Y_e theta vel_inf theta)  
-t totflux (creates and plots total flux of the ejecta  
-t massave (creates the mass averaged quantities)  
-t ejtau (computes the 3D histogram, with Y_e, entropy and expansion timescale as axis)  
-t yeilds (computes and plots nuclesynthetic yeilds)  
-t mknprof (computes and plots angular profile of mass, Y_e and vel_inf for macrokilonova bayes code)  
