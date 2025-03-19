#!/bin/bash

# GOAL: Check if processing files for micapipe, hippunfold, zbrains are present for a list of subjects
# INPUT: CSV file with columns for IDs and session of interest
# OUTPUT: CSV file showing if processing directory exists for the participant-session and if so, the number of files in the sub-directories

# To Do:
## - make log file with variables as defined, paths searched (for auditing purposes)
## - add suffix to file name for context (e.g., 3T or 7T)
## - add variable to indicate prefix to session number in file coding (e.g., 0 in MICs, a in PNI)

source config.shlib

# Variables
## n.b. values in parantheses are the default values
IN_LIST=$(config_get IN_CSV_PATH "/DEFAULT/IN/PATH/")
ID_COL=$(config_get ID_COL "ID")
SES_COL=$(config_get SES_COL "SES")
SES_PREFIX=$(config_get SES_PREFIX "0")

# Path to directory containing processing files
ROOT_PATH=$(config_get ROOT_PATH "/data/mica3/BIDS_PNI")
SUB_ROOT=$(config_get SUB_ROOT "derivatives")
MP_PATH=$(config_get MP_PATH "micapipe_v0.2.0")
HU_PATH=$(config_get HU_PATH "hippunfold"_v1.3.0/hippunfold)
ZB_PATH=$(config_get ZB_PATH "zBrains_clinical")

# Identify column number
ID_colNUM=$(head -n 1 $IN_LIST | tr ',' '\n' | grep -n "$ID_COL" | cut -d: -f1)
SES_colNUM=$(head -n 1 $IN_LIST | tr ',' '\n' | grep -n "$SES_COL" | cut -d: -f1)

if [ -z "$ID_colNUM" ]; then
  echo -e "ERROR. ID column (${ID_COL}) not found in input data sheet. \n\t Check that column name and input path are properly defined."
  exit 1
else
  echo "ID col name (index): ${ID_COL} (${ID_colNUM})"
fi

if [ -z "$SES_colNUM" ]; then
  echo -e "ERROR. SES column (${SES_COL}) not found in input data sheet. \n\t Check that the column name and input path are properly defined."
  exit 1
else
  echo "SES col name (index): ${SES_COL} (${SES_colNUM})"
fi

# Create output file
OUT_PATH=$(config_get OUT_CSV_PATH "/host/verges/tank/data/daniel/zBrains_3T7T/data")
OUT_NAME=$(config_get OUT_NAME "")
DATE=$(date +%d%b%Y)
OUTPUT_FILE="${OUT_PATH}/processingCheck_${OUT_NAME}_${DATE}.csv"
touch $OUTPUT_FILE
echo "ID,SES,Micapipe,Hippunfold,ZBrains" > $OUTPUT_FILE

echo "Output file created: $OUTPUT_FILE"

counter=-1

# Read in list of IDs
while IFS=, read -r row;
do
    if [[ $counter -eq -1 ]]; then
      ((counter++))
      continue
    else
      ((counter++))
    fi

  # Extract ID and SES
    ID=$(echo $row | cut -d, -f$ID_colNUM)
    SES=$(echo $row | cut -d, -f$SES_colNUM)

    echo "ID, ses: $ID, $SES"

    # Initialize counts
    MP_COUNT=N
    HU_COUNT=N
    ZB_COUNT=N

    # Define directories

    MP_DIR=${ROOT_PATH}/${SUB_ROOT}/${MP_PATH}/sub-${ID}/ses-${SES_PREFIX}${SES}/

    echo -e "\tMICAPIPE dir: $MP_DIR"

    HU_DIR=${ROOT_PATH}/${SUB_ROOT}/${HU_PATH}/sub-${ID}/ses-${SES_PREFIX}${SES}/
    ZB_DIR=${ROOT_PATH}/${SUB_ROOT}/${ZB_PATH}/sub-${ID}/ses-${SES_PREFIX}${SES}/

    # Check and count files in directories
    for DIR in "$MP_DIR" "$HU_DIR" "$ZB_DIR"; do
        if [ -d "$DIR" ]; then
            COUNT=$(find "$DIR" -type f | wc -l)
            case "$DIR" in
                "$MP_DIR") MP_COUNT=$COUNT ;;
                "$HU_DIR") HU_COUNT=$COUNT ;;
                "$ZB_DIR") ZB_COUNT=$COUNT ;;
            esac
        fi
    done

    # Output to CSV
    echo "$ID,$SES,$MP_COUNT,$HU_COUNT,$ZB_COUNT" >> $OUTPUT_FILE

done < $IN_LIST
