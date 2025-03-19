#!/bin/bash

# run w-score

/host/verges/tank/data/daniel/z-brains/zbrains --run "all"\
    --dataset ${dir_root}\
    --micapipe ${dir_mp}\
    --hippunfold ${dir_hu}\
    --zbrains ${zb}\
    --control_prefix ${ctrl_prefix}\
    --smooth_ctx ${smooth_ctx}\
    --smooth_hip ${smooth_hip}\
    --res ${res} \
    --sub "all"\
    --ses "all"\
    --feat "all"\
    --n_jobs 4\
    --n_jobs_wb 4\
    --wb_path /usr/bin/\
    --verbose 2\
	--volumetric 0\
	--dicoms 0\
    --pyinit=/data/mica1/03_projects/ian/anaconda3