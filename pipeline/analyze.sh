#!/usr/bin/env bash

function call {
    echo $*
    eval $*
}

if [[ (-z "$1") ]]; then
    echo "Usage: $0 simulation_name"
    exit 1
fi

#call python preanalysis.py -s $1 -t update_status print_status || exit 1
#call python preanalysis.py -s $1 -t collate --overwrite yes || exit 1
#call python gw.py -t strain tmergtcoll -s $1 --overwrite yes || exit 1
call python ejecta.py -s $1 -t reshape -d 0 1 -p 8 --overwrite yes || exit 1
call python ejecta.py -s $1 -t all -d 0 1 -m all --v_n all --overwrite yes || exit 1
#call python profile.py -s $1 -t all --it all --mask disk remnant --plane all --v_n all --overwrite yes || exit 1

echo "All Done"







# template for running the pipeline with varios options

#function call {
#    echo $*
#    eval $*
#}
#
#ANALYSIS_HOME=$(cd $(dirname $0); pwd)
#
#if [[ (-z "$1") || (-z "$2") || (-z "$3") ]]; then
#    echo "Usage: $0 sim_dir_name /path/to/simulation /path/to/output"
#    exit 1
#fi
#sim=$1
#target=$2
#output=$3
#
## creates ittime.h5 file that maps iterations and timesteps of available data
#call python preanalysis.py -s $sim -i $2 -o $3 -t update_status print_status || exit 1
#
#call python preanalysis.py -s $sim -i $2 -o $3 -t collate --overwrite yes --usemaxtime auto || exit 1
#
## cmpute strain andwaveform. Attmpt to get tmerg and tcoll
#call python gw.py -t all -i $3$1/collated/ -o $3$1/waveforms/ --overwrite yes || exit 1
#
## creates .h5 file with remaped outflow data for a spherical grid
#call python outflowed.py -s $sim -i $2 -o $3 -t reshape -d 0 -p 8 --overwrite yes --usemaxtime auto || exit 1
#
## do outflwoed analysis for detector -d 0  and for mask -m  geo (geodesic)
#call python outflowed.py -s $sim -i $2 -o $3 -t all -m geo bern_geoend geo_entropy_above_10 geo_entropy_below_10 theta60_geoend Y_e04_geoend -d 0 --overwrite yes || exit 1
#
## do module_profile analusis. Warning! If there are many profiles. i will take long
#call python profile.py -s $sim -i $2 -o $3 -t all --it all --overwrite no --usemaxtime auto --plane all|| exit 1
#
##call touch postrocessing.done