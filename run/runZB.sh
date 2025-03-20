#!/bin/bash

# run z-score

/host/verges/tank/data/daniel/z-brains/zbrains --run "all"\
    --dataset $1\
    --micapipe $2\
    --hippunfold $3\
    --zbrains $4\
    --control_prefix $5\
    --demo_ref $6\
    --demo $7\
    --smooth_ctx $8\
    --smooth_hip $9\
    --res $10\
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


#    --column_map session_id=\"SES\"\
#    --column_map subject_id=\"ID\"\