#!/usr/bin/env bash

function call {
    echo $*
    eval $*
}

ANALYSIS_HOME=$(cd $(dirname $0); pwd)

if [[ (-z "$1") || (-z "$2") || (-z "$3") ]]; then
    echo "Usage: $0 sim_dir_name /path/to/simulation/ /path/to/output/"
    echo "For example $0 MYSIM /hdd/data/MYSIM/ /hdd/data/MYSIM/ppr_output/"
    exit 1
fi
sim=$1    # e.g. LS220_M13641364_LR
target=$2 # e.g. home/LS220_M13641364_LR/
output=$3 # e.g. home/LS220_M13641364_LR/ppr_output/




# creates ittime.h5 file that maps iterations and timesteps of available data
call python preanalysis/preanalysis.py -s $1 -i $2 -o $3 -t update_status print_status || exit 1

# collalte ascii files
call python preanalysis/preanalysis.py -s $1 -i $2 -o $3 -t collate --overwrite yes --usemaxtime auto || exit 1

# analyze the strain and produce GW waveforms
call python gw/gw.py -t all -i $3/collated/ --overwrite yes || exit 1

# convert outflow ascii files into a single .h5 file for a given detector (-d)
hydroEOS="/media/vsevolod/data/EOS/LS220/LS_220_hydro_27-Sep-2014.h5"
call python ejecta/ejecta.py -s $1 -i $2 -o $3 -t reshape -d 0 -p 8 --overwrite yes --usemaxtime auto --eos $hydroEOS || exit 1

# perform complete analysis of the module_ejecta for given detector and masks
skynetDIR="/media/vsevolod/data/skynet/"
call python ejecta/ejecta.py -s $1 -i $2 -o $3 -t all \
    -m geo bern_geoend geo_entropy_above_10 geo_entropy_below_10 theta60_geoend Y_e04_geoend \
    -d 0 --overwrite yes \
    --eos $hydroEOS \
    --skynet $skynetDIR || exit 1

# to make a 2D movie use
call python slices/slices.py -s $1 -i $2 -o $3 -t movie --v_n rho --it all --plane xy  --rl 3  || exit 1



# creates ittime.h5 file that maps iterations and timesteps of available data
#call python module_preanalysis/old_preanalysis.py -s $sim -i $2 -o $3 -t update_status print_status || exit 1

# call python module_preanalysis/old_preanalysis.py -s $sim -i $2 -o $3 -t collate --overwrite yes --usemaxtime auto || exit 1

# cmpute strain andwaveform. Attmpt to get tmerg and tcoll
# call python module_gw/module_gw.py -t all -i $3$1/collated/ -o $3$1/waveforms/ --overwrite yes || exit 1

# creates .h5 file with remaped outflow data for a spherical grid
# call python old_outflowed.py -s $sim -i $2 -o $3 -t reshape -d 0 -p 8 --overwrite yes --usemaxtime auto || exit 1

# do outflwoed analysis for detector -d 0  and for mask -m  geo (geodesic)
# call python old_outflowed.py -s $sim -i $2 -o $3 -t all -m geo bern_geoend geo_entropy_above_10 geo_entropy_below_10 theta60_geoend Y_e04_geoend -d 0 --overwrite yes || exit 1

# do module_profile analusis. Warning! If there are many profiles. i will take long
# call python module_profile.py -s $sim -i $2 -o $3 -t all --it all --overwrite no --usemaxtime auto --plane all|| exit 1

#call touch postrocessing.done