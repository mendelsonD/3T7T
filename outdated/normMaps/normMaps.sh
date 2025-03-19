#!/bin/bash

# Goal: create normative maps using zBrains
# Daniel Mendelson, 7 Jan 2025
# Working under the supervision of Boris Bernhardt, PhD

source config.shlib

# set Working directory to path where z-brains program is saved

# path for dataset in BIDS structure
ROOT_PATH=$(config_get ROOT_PATH "/data/mica3/BIDS_MICs/")
RAW_PATH=$(config_get RAW_PATH "${ROOT_PATH}/rawdata")
MP_PATH=$(config_get MICAPIPE_PATH "${ROOT_PATH}/derivatives/micapipe_v0.2.0")
HU_PATH=$(config_get HIPPUNFOL_PATH "${ROOT_PATH}/derivatives/hippunfold_v1.3.0/hippunfold/")
OUT_PATH=$(config_get OUT_PATH "/host/verges/tank/data/daniel/zBrain_3T7T/outputs")

NORM_IDS=$(config_get ID_csv "/home/bic/danielme/Documents/daniel_verges/zBrain_3T7T/data/tT_HC-zBrains_Norm_7Jan2025.csv")

echo "Running zBrains. Outputs are pointed to ${OUT_PATH}."
# Make normative map
./z-brains -sub "$id" -ses "$ses" \
    -rawdir "${RAW_PATH}" \
    -micapipedir "${MP_PATH}" \
    -hippdir "${HU_PATH}" \
    -outdir "${OUT_PATH}" \
    -run regional \
    -approach "zscore" \
    -demo_cn "${NORM_IDS}" \
    -verbose 2
