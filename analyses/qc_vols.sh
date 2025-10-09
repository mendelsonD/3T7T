#!/bin/bash

# Usage: ./qc_vols.sh input.csv <start index of csv (optional)>
# e.g.: 
# bash /host/verges/tank/data/daniel/3T7T/z/code/analyses/qc_vols.sh /host/verges/tank/data/daniel/3T7T/z/outputs/qc_table_25Sep2025-141754.csv 


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
    if [[ "${cols[$i]}" == *_pth ]] && ! [[ "${cols[$i]}" =~ _L_pth$ || "${cols[$i]}" =~ _R_pth$ ]]; then
        path_indices+=($i)
    fi
done

surf_indices=()
for i in "${!cols[@]}"; do
    if [[ "${cols[$i]}" == *_L_pth ]] && [[ "${cols[$i]}" == *_R_pth ]]; then
        surf_indices+=($i)
    fi
done


# Find indices for UID, ID, SES (Date)
uid_idx=-1
id_idx=-1
ses_idx=-1
date_idx=-1
QC_ctxSurf_idx=-1
QC_hipSurf_idx=-1

for i in "${!cols[@]}"; do
    case "${cols[$i]}" in
        UID) uid_idx=$i ;;
        ID) id_idx=$i ;;
        SES) ses_idx=$i ;;
        Date) date_idx=$i ;;
        QC_ctxSurf) QC_ctxSurf_idx=$i ;;
        QC_hipSurf) QC_hipSurf_idx=$i ;;
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

if [[ $QC_ctxSurf_idx -eq -1 ]]; then
    echo "Warning: Column 'QC_ctxSurf' not found."
fi

if [[ $QC_hipSurf_idx -eq -1 ]]; then
    echo "Warning: Column 'QC_hipSurf' not found."
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
    if [[ $QC_ctxSurf_idx -ne -1 ]]; then
        qc_ctx="${row[$QC_ctxSurf_idx]}"
        echo "  QC_ctxSurf: $qc_ctx"
    fi
    if [[ $QC_hipSurf_idx -ne -1 ]]; then
        qc_hip="${row[$QC_hipSurf_idx]}"
        echo "  QC_hipSurf: $qc_hip"
    fi

    # load surfaces onto t1w image using FreeView
    # if qc_ctx and qc_hip are not NA for this case (column may exist or not)

    for path_idx in "${path_indices[@]}"; do
        column_name="${cols[$path_idx]%%_pth}"  # Extract column name before '_pth'
        path="${row[$path_idx]}"
        if [[ "$path" == NA* ]]; then
            continue
        fi

        # Check if this is a T1w image (column name contains 'T1w')
        if [[ "$column_name" == *T1w* ]]; then
            # Check if surface QC has been completed (QC_ctxSurf and QC_hipSurf not NA)
            ctx_qc_done=true
            hip_qc_done=true
            surf_args=()

            if [[ $QC_ctxSurf_idx -ne -1 && "${row[$QC_ctxSurf_idx]}" == '' ]]; then
                ctx_qc_done=false
                # Add pial and white surfaces (L/R) for current T1w image
                for surf_type in pial white; do
                    for hemi in L R; do 
                        surf_col="${surf_type}_${hemi}_pth"
                        for surf_idx in "${!cols[@]}"; do
                            if [[ "${cols[$surf_idx]}" == "$surf_col" ]]; then
                                surf_path="${row[$surf_idx]}"
                                if [[ "$surf_path" != NA* && -n "$surf_path" ]]; then
                                    if [[ $surf_type == pial ]]; then 
                                        surf_args+=("-f" "$surf_path:edgecolor=red")
                                    elif [[ $surf_type == white ]]; then 
                                        surf_args+=("-f" "$surf_path:edgecolor=blue")
                                    fi
                                fi
                            fi
                        done
                    done
                done
            fi
            if [[ $QC_hipSurf_idx -ne -1 && ("${row[$QC_hipSurf_idx]}" == '' || "${row[$QC_hipSurf_idx]}" < "0.7") ]]; then
                hip_qc_done=false
                # add hippocampal surfaces if available
                for hemi in L R; do
                    for surf_type in inner outer; do
                        surf_col="${surf_type}_${hemi}_pth"
                        for surf_idx in "${!cols[@]}"; do
                            if [[ "${cols[$surf_idx]}" == "$surf_col" ]]; then
                                surf_path="${row[$surf_idx]}"
                                if [[ "$surf_path" != NA* && -n "$surf_path" ]]; then
                                    if [[ $surf_type == inner ]]; then 
                                        surf_args+=("-f" "$surf_path:edgecolor=yellow")
                                    elif [[ $surf_type == outer ]]; then 
                                        surf_args+=("-f" "$surf_path:edgecolor=green")
                                    fi
                                fi
                            fi
                        done
                    done
                done
            fi

            if $ctx_qc_done && $hip_qc_done; then
                # Both QC done, skip loading surfaces
                echo "  $column_name: $path"
                echo "    Surface QC already completed. No surfaces to load."
                mrview -load "$path" -mode 2 -plane 1 -voxel 125,0,0 -noannotations -interpolation 0
                continue
            else
                echo "  $column_name: $path"
                echo "    Loading surfaces for QC..."
                freeview -v "$path" "${surf_args[@]}"
               
            fi
        else
        
            echo "  $column_name: $path"

            mrview -load "$path" -mode 2 -plane 1 -voxel 125,0,0 -noannotations -interpolation 0
            
            if [[ $? -ne 0 ]]; then
                exit 0
            fi
        fi

    done
done
