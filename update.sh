#!/usr/bin/env bash

function call {
    echo $*
    eval $*
}

if (-z "$1") ; then
    echo "Usage: $0 eos_of_the_simlist"
    exit 1
fi

list=$1

simlist_blh="
    BLh_M10201856_M0_LK_SR
    BLh_M10651772_M0_LK_SR
    BLh_M11841581_M0_LK_SR
    BLh_M13641364_M0_LK_SR
    BLh_M16351146_M0_LK_LR"
simlist_dd2="
    DD2_M11461635_M0_LK_SR
    DD2_M13641364_M0_HR
    DD2_M13641364_M0_HR_R04
    DD2_M13641364_M0_LK_HR_R04
    DD2_M13641364_M0_LK_LR_R04
    DD2_M13641364_M0_LK_SR_R04
    DD2_M13641364_M0_LR
    DD2_M13641364_M0_LR_R04
    DD2_M13641364_M0_SR
    DD2_M13641364_M0_SR_R04
    DD2_M14321300_M0_LR
    DD2_M14351298_M0_LR
    DD2_M14861254_M0_HR
    DD2_M14861254_M0_LR
    DD2_M14971245_M0_HR
    DD2_M14971245_M0_SR
    DD2_M14971246_M0_LR
    DD2_M15091235_M0_LK_HR
    DD2_M15091235_M0_LK_SR
    DD2_M16351146_M0_LK_LR"
simlist_ls220="
    LS220_M10651772_M0_LK_LR
    LS220_M11461635_M0_LK_SR
    LS220_M13641364_M0_HR
    LS220_M13641364_M0_LK_HR
    LS220_M13641364_M0_LK_SR
    LS220_M13641364_M0_LK_SR_restart
    LS220_M13641364_M0_LR
    LS220_M13641364_M0_SR
    LS220_M14001330_M0_HR
    LS220_M14001330_M0_SR
    LS220_M14351298_M0_HR
    LS220_M14351298_M0_SR
    LS220_M14691268_M0_HR
    LS220_M14691268_M0_LK_HR
    LS220_M14691268_M0_LK_SR
    LS220_M14691268_M0_LR
    LS220_M14691268_M0_SR
    LS220_M16351146_M0_LK_LR"
simlist_sfho="
    SFHo_M10651772_M0_LK_LR
    SFHo_M11461635_M0_LK_SR
    SFHo_M13641364_M0_HR
    SFHo_M13641364_M0_LK_HR
    SFHo_M13641364_M0_LK_SR
    SFHo_M13641364_M0_LK_SR_2019pizza
    SFHo_M13641364_M0_SR
    SFHo_M14521283_M0_HR
    SFHo_M14521283_M0_LK_HR
    SFHo_M14521283_M0_LK_SR
    SFHo_M14521283_M0_LK_SR_2019pizza
    SFHo_M14521283_M0_SR
    SFHo_M16351146_M0_LK_HR
    SFHo_M16351146_M0_LK_LR"
simlist_sly4="
    SLy4_M10651772_M0_LK_LR
    SLy4_M11461635_M0_LK_SR
    SLy4_M13641364_M0_LK_LR
    SLy4_M13641364_M0_LK_SR
    SLy4_M13641364_M0_LR
    SLy4_M13641364_M0_SR
    SLy4_M14521283_M0_HR
    SLy4_M14521283_M0_LR
    SLy4_M14521283_M0_SR
"

if [[ $list == *"LS220"* ]]; then
  simlist=$simlist_ls220
  echo "Using LS220 EOS"
fi
if [[ $list == *"SLy4"* ]]; then
  simlist=$simlist_sly4
  echo "Using SLy4 EOS"
fi
if [[ $list == *"DD2"* ]]; then
  simlist=$simlist_dd2
  echo "Using DD2 EOS"
fi
if [[ $list == *"SFHo"* ]]; then
  simlist=$simlist_sfho
  echo "Using SFHo EOS"
fi
if [[ $list == *"BLh"* ]]; then
  simlist=$simlist_blh
  echo "Using BLh EOS"
fi



for i in $simlist; do
  echo "$i"
#  call python preanalysis.py -s $i -t update_status print_status
#  call python profile.py -s $i -t all --it all
  call python preanalysis.py -s $i -t init_data
done

echo "all done"