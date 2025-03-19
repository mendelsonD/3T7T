#!/bin/bash
# Enable error handling
set -e

# Goal:
#   Iterate through all participant IDs in a master sheet
#   Assume that have control csv files for each participant in the list (can do using python script: controlSheets.ipynb)

# Inputs:
# - master_csv: path to the master sheet
# - ID_colname: column name for the participant ID in the master sheet
# - control_csv: path to directory containing the control csv's
# - path to the dataset and relevant directories

# Returns:
# - zbrains output for each participant in the master sheet, saved as usual

# Use config file
source config.shlib

# Path to the dataset, or the folder containing the 'derivatives' folder
pth_dataset=$(config_get PATH_DATASET "/DEFAULT/IN/PATH")

# Path to the zbrains, micapipe, and hippunfold directories
zb_dir=$(config_get PATH_ZBRAINS "/DEFAULT/IN/PATH/")
mp_dir=$(config_get PATH_MICAPIPE "micapipe_v0.2.0")
hu_dir=$(config_get PATH_HIPPUNFOLD "hippunfold_v1.3.0")

# PATH to sheet with IDs to run
pth_pts=$(config_get PATH_PT_LIST_CSV "/host/verges/tank/data/daniel/zBrain_3T7T/data/pt_2Jan2025.csv")

# Path directory containing lists of control participants
control_lists=$(config_get PATH_CONTROL_CSV "/DEFAULT/PATH/")
colName_ID=$(config_get COLNAME_ID "ID")
colName_ses=$(config_get COLNAME_SESSION "SES")

LOG_PATH=$(config_get PATH_LOG "/host/verges/tank/data/daniel/zBrain_3T7T/code/runZb_variableCtrl/logs")
LOG=$LOG_PATH"/log_zBrainsRun_"$(date +"%d%b%Y_%Hh%M")".txt"

# Initialize log file
echo "runZb_variableCtrl" >> ${LOG}
echo "Log file created at: $LOG"
echo -e "\nstart: $(date +"%d-%b-%Y %H:%M:%S")" 2>&1 | tee -a "$LOG"
echo -e "Variables: \n\t'pth_dataset': ${pth_dataset}\n\t'zb_di': ${zb_dir}\n\t'mp_dir': ${mp_dir}\n\t'hu_dir': ${hu_dir}\n\t'pth_pts': ${pth_pts}\n\t'control_lists': ${control_lists}\n\n\t'colName_ID': ${colName_ID}\n\t'colName_ses': ${colName_ses}" >> ${LOG}


# Check if the input CSV file exists
if [ ! -f $pth_pts ]; then
  echo -e "ERROR. Input CSV file not found. Check 'PATH_PT_LIST_CSV' in 'config.cfg'. \n\tPath: $pth_pts\nExiting script." 2>&1 | tee -a "$LOG"
  echo -e "Exiting script: $(date +"%d%b%Y_%Hh%M")" 2>&1 | tee -a "$LOG"
  exit 1
fi

# check that the control csv directory exists
if [ ! -d $control_lists ]; then
    echo -e "ERROR. Control CSV directory not found. Check 'PATH_CONTROL_CSV' in 'config.cfg'. \n\tPath: $control_lists\nExiting script." 2>&1 | tee -a "$LOG"
    echo -e "Exiting script: $(date +"%d%b%Y_%Hh%M")" 2>&1 | tee -a "$LOG"
    exit 1
fi

# get the column number for the ID and session columns
colNum_ID=$(awk -v header="$colName_ID" -F, '
BEGIN { colnum = -1 }
NR == 1 {
    for (i = 1; i <= NF; i++) {
        if ($i ~ header) {
            colnum = i-1
            break
        }
    }
    print colnum
}' "$pth_pts")

colNum_SES=$(awk -v header="$colName_ses" -F, '
BEGIN { colnum = -1 }
NR == 1 {
    for (i = 1; i <= NF; i++) {
        if ($i ~ header) {
            colnum = i-1
            break
        }
    }
    print colnum
}' "$pth_pts")

#echo "ID column number: $colNum_ID" >> "$LOG"
#echo "Session column number: $colNum_SES" >> "$LOG"

{
    read
    while IFS=, read -r -a row; do
        #echo "Row: ${row[@]}"

        # extract participant ID
        subject=${row[$colNum_ID]}
        formatted_subj=sub-${subject}

        session=${row[$colNum_SES]}
        formatted_ses=ses-${session}

        # check that subject is not empty
        if [ -z "$subject" ]; then
            echo -e "ERROR. Empty participant ID found in CSV. \n\tPATH: ${pth_pts} \n\tRow: ${row[@]}\nFix before rerunning, exiting script." 2>&1 | tee -a "$LOG"
            echo -e "Exiting script: $(date +"%d%b%Y_%Hh%M")" 2>&1 | tee -a "$LOG"
            continue
        fi

        if [[ $subject == *"HC"* ]]; then
            ctrl_prefix="PX"
        elif [[ $subject == *"PX"* ]]; then
            ctrl_prefix="HC"
        elif [[ $subject == *"PNC"* ]]; then
            ctrl_prefix="PNE"
        elif [[ $subject == *"PNE"* ]]; then
            ctrl_prefix="PNC"
        else
            echo -e "Warning ID naming pattern not recognized for ${subject}. Recognized patterns are 'HC', 'PX','PNC','PNE'. Setting ctrl_prefix to 'HC'"
            ctrl_prefix="HC"
        fi

        #echo ${ctrl_prefix}

        echo -e "\n\n-----------------------------\n${formatted_subj}-${formatted_ses}..." 2>&1 | tee -a "$LOG"
        
        # get the control csv for the current participant
        ## search `control_lists` directory for the most recent csv files with the current ID

        controls=$(ls -t ${control_lists}/*${subject}*.csv 2>/dev/null | head -n 1)
        # if get abnormal output from ls, skip to next participant
        if [ $? -ne 0 ] || [ -z "$controls" ]; then
            echo -e "ERROR. Control CSV file not found for subject ${subject}. Check 'PATH_CONTROL_CSV' in 'config.cfg'. \n\tPath: ${control_lists}/${subject}*.csv\nSkipping to next participant." 2>&1 | tee -a "$LOG"
            echo -e "Exiting script: $(date +"%d%b%Y_%Hh%M")" 2>&1 | tee -a "$LOG"
            continue
        fi

        echo -e "\tControls path: $controls" 2>&1 | tee -a "$LOG"

        #continue

        # Run zbrains
        /host/verges/tank/data/daniel/z-brains/zbrains --run "analysis"\
                --sub "${formatted_subj}" \
                --ses "${formatted_ses}" \
                --dataset ${pth_dataset} \
                --zbrains ${zb_dir} \
                --micapipe ${mp_dir} \
                --hippunfold ${hu_dir} \
                --dataset_ref ${pth_dataset} \
                --zbrains_ref ${zb_dir} \
                --demo_ref ${controls} \
		        --control_prefix "${ctrl_prefix}" \
                --column_map participant_id=ID session_id=SES \
                --smooth_ctx 10 \
                --smooth_hip 5 \
                --n_jobs 4 \
                --n_jobs_wb 4 \
                --label_ctx "white" \
                --feat thickness qT1 ADC FA flair \
                --wb_path /usr/bin/ \
                --verbose 2 \
		--volumetric 0 \
		--dicoms 0 \
                --pyinit=/data/mica1/03_projects/ian/anaconda3 2>&1 | tee -a "$LOG"
    done
} < ${pth_pts}

echo -e "\nComplete. $(date +"%d-%b-%Y: %H:%M:%S")" 2>&1 | tee -a "$LOG"

exit 1
