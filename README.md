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
-t task to perform, such as: update_status or print_status
--eos 
