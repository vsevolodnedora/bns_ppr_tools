#!/usr/bin/env bash

function call {
    echo $*
    eval $*
}

ANALYSIS_HOME=$(cd $(dirname $0); pwd)

if [[ (-z "$1") || (-z "$2") || (-z "$3") ]]; then
    echo "Usage: $0 sim_dir_name /path/to/simulation /path/to/output"
    exit 1
fi
sim=$1
target=$2
output=$3

# creates ittime.h5 file that maps iterations and timesteps of available data
call python preanalysis.py -s $sim -i $2 -o $3 -t update_status print_status collate || exit 1

# cmpute strain andwaveform. Attmpt to get tmerg and tcoll
call python gw.py -t all -i $3$1/collated/ -o $3$1/waveforms/ || exit 1

# creates .h5 file with remaped outflow data for a spherical grid
call python outflowed.py -s $sim -i $2 -o $3 -t reshape -d 0 -p 8 || exit 1

# do outflwoed analysis for detector -d 0  and for mask -m  geo (geodesic)
call python outflowed.py -s $sim -i $2 -o $3 -t all -m geo bern_geoend Y_e04_geoend -d 0 --overwrite no || exit 1

# do profile analusis. Warning! If there are many profiles. i will take long
call python profile.py -s $sim -i $2 -o $3 -t all --it all --overwrite no || exit 1

#call touch postrocessing.done