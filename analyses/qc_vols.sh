#!/bin/bash

# Usage: ./qc_vols.sh input.csv

input_csv="$1"

if [[ -z "$input_csv" ]]; then
    echo "Usage: $0 input.csv"
    exit 1
fi

# Get header and find columns ending with '_path'
header=$(head -n 1 "$input_csv")
IFS=',' read -ra cols <<< "$header"

# Find indices of columns ending with '_path'
path_indices=()
for i in "${!cols[@]}"; do
    if [[ "${cols[$i]}" == *_path ]]; then
        path_indices+=($i)
    fi
done

# Find indices for UID, ID, SES (Date)
uid_idx=-1
id_idx=-1
ses_idx=-1
for i in "${!cols[@]}"; do
    case "${cols[$i]}" in
        UID) uid_idx=$i ;;
        ID) id_idx=$i ;;
        SES|Date) ses_idx=$i ;;
    esac
done

if [[ $uid_idx -eq -1 || $id_idx -eq -1 || $ses_idx -eq -1 || ${#path_indices[@]} -eq 0 ]]; then
    echo "Required columns (UID, ID, SES/Date, *_path) not found."
    exit 1
fi

# Read the CSV, skipping the header
tail -n +2 "$input_csv" | while IFS=',' read -ra row; do
    # Get first _path column value
    path="${row[${path_indices[0]}]}"
    if [[ "$path" == NA* ]]; then
        continue
    fi
    uid="${row[$uid_idx]}"
    id="${row[$id_idx]}"
    ses="${row[$ses_idx]}"
    echo "$uid-$id-$ses"
    mrview -load "$path" -mode 2 -voxel 125,0,150
done