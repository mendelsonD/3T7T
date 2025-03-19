#!/bin/bash

# GOAL: Check if processing files for micapipe, hippunfold, zbrains are present for a list of subjects
# INPUT: CSV file with columns for IDs and session of interest
# OUTPUT: CSV file showing if processing directory exists for the participant-session and if so, the number of files in the sub-directories

source config.shlib

# Variables
## n.b. values in parantheses are the default values
IN_LIST=$(config_get IN_CSV_PATH "/DEFAULT/IN/PATH/")
ID_COL=$(config_get ID_COL "ID")
SES_COL=$(config_get SES_COL "SES")

# Path to directory containing processing files
ROOT_PATH=$(config_get ROOT_PATH "/data/mica3/BIDS_PNI")
SUB_ROOT=$(config_get SUB_ROOT "derivatives")
MP_PATH=$(config_get MP_PATH "micapipe_v0.2.0")
HU_PATH=$(config_get HU_PATH "hippunfold"_v1.3.0/hippunfold)
ZB_PATH=$(config_get ZB_PATH "zBrains_clinical")

# Create output file
OUT_PATH=$(config_get OUT_CSV_PATH "/host/verges/tank/data/daniel/zBrains_3T7T/data")
DATE=$(date +%d%b%Y)
OUTPUT_FILE="$OUT_PATH/processingCheck_$DATE.csv"
touch $OUTPUT_FILE
echo "ID,SES" > $OUTPUT_FILE

echo "Output file created: $OUTPUT_FILE"

ID_colNUM=$(head -n 1 $IN_LIST | tr ',' '\n' | grep -n "$ID_COL" | cut -d: -f1)
SES_colNUM=$(head -n 1 $IN_LIST | tr ',' '\n' | grep -n "$SES_COL" | cut -d: -f1)

if [ -z "$ID_colNUM" ]; then
  echo -e "ERROR. ID column ($ID_COL) not found in input data sheet. \n\t Check that column name and input path are properly defined."
  exit 1
else
  echo "ID col name (index): $ID_COL ($ID_colNUM)"
fi

if [ -z "$SES_colNUM" ]; then
  echo -e "ERROR. SES column ($SES_COL) not found in input data sheet. \n\t Check that the column name and input path are properly defined."
  exit 1
else
  echo "SES col name (index): $SES_COL ($SES_colNUM)"
fi


# Read in list of IDs
while IFS=, read -r row;
do
    # Extract ID and SES
    ID=$(echo $row | cut -d, -f$ID_colNUM)
    SES=$(echo $row | cut -d, -f$SES_colNUM)

    echo "$ID,$SES" >> $OUTPUT_FILE

done < $IN_LIST
