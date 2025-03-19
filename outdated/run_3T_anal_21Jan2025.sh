#!/bin/bash

# Enable error handling
set -e

# Path to the dataset, or the folder containing the 'derivatives' folder
pth_dataset="/data/mica3/BIDS_MICs"

# Set the directories for micapipe, hippunfold, and zbrains, which will be looked for in the 'derivates' folder
zbrains_dir="zbrains_3T7T_daniel"
micapipe_dir="micapipe_v0.2.0"
hippunfold_dir="hippunfold_v1.3.0"

# Set the paths to the demographic control and patient files
# The demo_controls are only needed for the analysis, and define the control samples to compare against.
# The demo_patients can be provided if desired, which will run all the patients in the list when the "all" keyword is used,
# otherwise the 'all' keyword will run every patient possible, given the micapipe and hippunfold availability, or, for the analysis
# it will run on all patients who have had zbrains proc run before.

# get temp csv with all pt but current subject
control_csv="/host/verges/tank/data/daniel/zBrain_3T7T/data/tT_HC_zBrains_Norm_8Jan2025.csv"

#demo_controls="/host/oncilla/local_raid/oualid/zbrains_csvs/participants_mics_hc.csv"

# get current subject
subject="sub-PX000"

# WHAT SESSION TO USE?
# get session for current subject
session="all" 

# The code below runs zbrains preserving the old behaviour, with a smooth_ctx of 10, a smooth_hip of 5, and a label_ctx of 'white'
# The new defaults for this are a smooth_ctx of 5, a smooth_hip of 2, and a label_ctx of 'midthickness'
# Much of the new volumetric code is dependent on cortical midthickness, so it is recommended.
./zbrains --run "analysis"\
        --sub "${subject}" \
        --ses "${session}" \
        --dataset ${pth_dataset} \
        --zbrains ${zbrains_dir} \
        --micapipe ${micapipe_dir} \
        --hippunfold ${hippunfold_dir} \
        --dataset_ref ${pth_dataset} \
        --zbrains_ref ${zbrains_dir} \
        --demo_ref ${control_csv} \
        --column_map participant_id=ID session_id=SES \
        --smooth_ctx 10 \
        --smooth_hip 5 \
        --n_jobs 4 \
        --n_jobs_wb 4 \
        --label_ctx "white" \
        --feat thickness qT1 ADC FA flair \
        --wb_path /usr/bin/ \
        --verbose 2 \
        --pyinit=/data/mica1/03_projects/ian/anaconda3