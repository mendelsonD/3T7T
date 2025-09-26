#!/bin/bash

# Usage: ./qc_vols.sh input.csv <start index of csv (optional)>

input_csv="$1"
idx_start="$2"

# validate inputs
if [[ -z "$input_csv" ]]; then
    echo "Usage: $0 input.csv"
    exit 1
fi

# Count the total number of rows in the CSV (excluding the header)
total_rows=$(wc -l < "$input_csv")
total_rows=$((total_rows - 1))  # Exclude the header row

# Validate idx_start
if ! [[ "$idx_start" =~ ^[0-9]+$ ]] || [[ "$idx_start" -lt 1 ]] || [[ "$idx_start" -ge "$total_rows" ]]; then
    idx_start=1
    echo "WARNING: Starting with index 1. Ensure the second argument is a valid row index less than $total_rows."
fi

# Trap Ctrl+C (SIGINT) to exit the script
trap "echo 'Exiting script.'; exit 0" SIGINT

# Get header and find columns ending with '_path'
header=$(head -n 1 "$input_csv")
IFS=',' read -ra cols <<< "$header"

# Find indices of columns ending with '_path'
path_indices=()
for i in "${!cols[@]}"; do
    if [[ "${cols[$i]}" == *_pth ]]; then
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
        SES) ses_idx=$i ;;
        Date) date_idx=$i ;;
    esac
done

if [[ $uid_idx -eq -1 ]]; then
    echo "Required column 'UID' not found."
    exit 1
fi

if [[ $id_idx -eq -1 ]]; then
    echo "Required column 'ID' not found."
    exit 1
fi

if [[ $ses_idx -eq -1 ]]; then
    echo "Required column 'SES' not found."
    exit 1
fi

if [[ $date_idx -eq -1 ]]; then
    echo "Required column 'Date' not found. "
    exit 1
fi

if [[ ${#path_indices[@]} -eq 0 ]]; then
    echo "No columns ending with '_path' found."
    exit 1
fi

# Read the CSV, skipping the header
echo "Use 'ctrl+c' to exit the script."

tail -n +$((idx_start + 1)) "$input_csv" | while IFS=',' read -ra row; do
    id="${row[$id_idx]}"
    ses="${row[$ses_idx]}"
    uid="${row[$uid_idx]}"
    date="${row[$date_idx]}"
    echo "----------------------------------------"
    echo "$uid $id-$ses ($date)"

    for path_idx in "${path_indices[@]}"; do
        column_name="${cols[$path_idx]%%_pth}"  # Extract column name before '_pth'
        path="${row[$path_idx]}"
        if [[ "$path" == NA* ]]; then
            continue
        fi
        
        echo "  $column_name: $path"

        mrview -load "$path" -mode 2 -plane 1 -voxel 125,0,0 -noannotations
        
        if [[ $? -ne 0 ]]; then
            exit 0
        fi

    done
done

# bash /host/verges/tank/data/daniel/3T7T/z/code/analyses/qc_vols.sh /host/verges/tank/data/daniel/3T7T/z/outputs/qc_table_25Sep2025-141754.csv 