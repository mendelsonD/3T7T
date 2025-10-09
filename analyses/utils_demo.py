import pandas as pd
import os
import sys
import importlib
import re
import numpy as np
import tTsTGrpUtils as tsutil

#sys.path.append("/Users/danielmendelson/Library/CloudStorage/OneDrive-McGillUniversity/Documents/PhD/Boris/code")
sys.path.append("/host/verges/tank/data/daniel/")
from genUtils import id, gen, t1

def get_demo(sheets, save_pth=None, save_name="01b_demo"):
    """
    Run all functions to generate demographic data for 3T-7T participants.

    input:
        sheets: list of dictionarys with source sheet information (path to sheet, key columns to extract
        save_pth: path to save the output demographic file

    outputs:
        list of IDs with paired 3T-7T data
        sheet with each row as separate session and with associated demographic information
    """
    importlib.reload(id)
    importlib.reload(t1)
    importlib.reload(tsutil)
    
    out = id.ids3T7T(sheets, save_pth=None) # determine participants with 3T and 7T data
    id_cols = out.columns.to_list()

    out = id.id_visits(sheets, out, save_pth=None) # get all sessions for these participants
    
    print("[get_demo] There are ", out['MICS_ID'].nunique(), "  unique participants and ", out.shape[0], " rows in datasheet.")
    #out.to_csv(f"{save_pth}/demo_debug_1-idVisits.csv", index=False)
   
    out = t1.demo(sheets, out, save_pth=None) # add demographic info
    #out.to_csv(f"{save_pth}/demo_debug_2-demo.csv", index=False) # debug

    out = rmvNADate(out, 'Date') # remove rows with missing scan dates. If only data from one study remains after this removal, then remove participant completely.
    print(out.columns)
    if 'scanDate' in out.columns:
        out = out.drop(columns=['scanDate'])

    out = dupSES(out, uniqCols=['MICS_ID', 'PNI_ID', 'study', 'SES'], mergeCols=['Date']) # merge repeated sessions
    print("[get_demo] After cleaning for missing scan dates, there are ", out['MICS_ID'].nunique(), " unique participants with a total of ", out.shape[0], " sessions (3T:",(out['study'] == '3T').sum(),", 7T:", (out['study'] == '7T').sum(),") in input datasheet.")
    #out.to_csv(f"{save_pth}/demo_debug_3-rmvNADate.csv", index=False) # debug
    
    # assign unique ID per participant
    uniqueIDName = "UID"

    out = uniqueID(out, idcols=id_cols, uniqueIDName=uniqueIDName)
    id_cols = id_cols + [uniqueIDName]
    print(id_cols)
    
    # save out
    if save_pth is not None: # save corresponding IDs
        import datetime

        save_name_tmp = f"{save_pth}/01a_ids_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.csv"
        toSave = out[id_cols].drop_duplicates()
        toSave.to_csv(save_name_tmp, index=False)
        print("[get_demo] Saved list of paired ids to: ", save_name_tmp)

    dob_col = None
    for sheet in sheets: # find DOB from sheet dictionary. If multiple sheets have DOB, use the first one found
        if 'DOB' in sheet and sheet['DOB'] is not None:
            dob_col = sheet['DOB']
            #print(f"[get_demo] Found DOB column in sheet {sheet['NAME']}: {dob_col}")
            break

    if dob_col is None:
        print("[get_demo] WARNING. No DOB column found in any input sheet. Age will not be computed.")
    
    out = tsutil.stdizeNA(out)
    out = mergeCols(out)
    print("[get_demo] Merged columns with the same name. Current columns (sorted): \n\t", sorted(out.columns.tolist(), key=lambda x: x.lower()))

    # Fill missing demo variables for each participant, warn if multiple unique values exist
    demo_vars = ['dob', 'sex', 'gender', 'handedness', 'ethnicity'] # these demo vars should be key values in the sheet dictionaries
    print("[get_demo] Filling missing demographic values for variables: ", demo_vars)

    for var in demo_vars:
        out = carryVals(out, var)

    out = t1.dateDif(out, [dob_col, "Date"], "age", save=False) # compute age
    
    out = group(out, out_col="grp", ID_col="MICS_ID", save_pth=None) # assign high level groups
    out = group(out, out_col="grp_detailed", ID_col="MICS_ID", save_pth=None) # assign detailed groups
    
    # Sanity checks:
    # Ensure all participants assigned a group
    grpChk(out, grp_cols=['grp', 'grp_detailed'])
    # Ensure no duplicate rows
    dupes = out.duplicated(subset=['UID', 'MICS_ID', 'PNI_ID', 'study', 'SES'], keep=False)
    if dupes.any():
        print("[WARNING] Duplicated rows:")
        print(out.loc[dupes, ['UID', 'MICS_ID', 'PNI_ID', 'study', 'SES'] + [col for col in out.columns if col not in ['MICS_ID', 'PNI_ID', 'study', 'SES']]])

    if save_pth is not None:
        import datetime

        save_name = f"{save_pth}/01b_demo_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.csv"
        out.to_csv(save_name, index=False)
        print("[get_demo] Saved to: ", save_name)
    else:
        print("[get_demo] WARNING. Not saving demographics sheet to file. To save, please provide a path to save_pth")
        
    return out, save_name

def uniqueID(df, idcols, uniqueIDName="UID"):
    """
    Assign unique ID to unique individual.

    Input:
        df: DataFrame with participant data
        idcols: list of columns that uniquely identify an individual (e.g., ['MICS_ID', 'PNI_ID'])
        uniqueIDName: name of the new column to store the unique ID

    Returns:
        df: DataFrame with new column for unique ID
    """
    df = df.copy()
    
    df_unique = df[idcols].drop_duplicates().reset_index(drop=True) 
    df_unique[uniqueIDName] = [f"UID{str(i+1).zfill(4)}" for i in range(df_unique.shape[0])] # create unique ID. Default format should be "UID0001", "UID0002", etc.

    # add as first row in original dataframe
    df = df.merge(df_unique, on=idcols, how='right')
    cols = [uniqueIDName] + [col for col in df.columns if col != uniqueIDName]
    df = df[cols]
    
    return df

def grp_summary(df_demo, col_grp= 'grp_detailed', save_pth=None, save_name="01c_grpSummary", toPrint=True):
    """
    Count number of participants, sessions for a grouping variable
    """
    # Calculate max and median number of sessions per participant for each group and study
    def max_sessions(df, group, study):
        subset = df[(df[col_grp] == group) & (df['study'] == study)]
        if subset.empty:
            return 0
        return subset.groupby('MICS_ID').size().max()

    def median_sessions(df, group, study):
        subset = df[(df[col_grp] == group) & (df['study'] == study)]
        if subset.empty:
            return 0
        return subset.groupby('MICS_ID').size().median()

    # Num of participants by group
    # Show number of unique participants per detailed group
    # Calculate unique participants, number of 3T sessions, and number of 7T sessions per group
    group_summary = (
        df_demo.groupby(col_grp)
        .agg(
            num_px=('MICS_ID', 'nunique'),
            num_ses_3T=('study', lambda x: ((x == '3T')).sum()),
            num_ses_7T=('study', lambda x: ((x == '7T')).sum()),
        )
    )

    group_summary['max_ses_3T'] = group_summary.index.map(lambda g: max_sessions(df_demo, g, '3T'))
    group_summary['max_ses_7T'] = group_summary.index.map(lambda g: max_sessions(df_demo, g, '7T'))
    group_summary['median_ses_3T'] = group_summary.index.map(lambda g: median_sessions(df_demo, g, '3T'))
    group_summary['median_ses_7T'] = group_summary.index.map(lambda g: median_sessions(df_demo, g, '7T'))
    # Add a total row at the end with sums for participant/session counts, leave median/max empty
    total_row = {
        'num_px': group_summary['num_px'].sum(),
        'num_ses_3T': group_summary['num_ses_3T'].sum(),
        'num_ses_7T': group_summary['num_ses_7T'].sum(),
        'max_ses_3T': '',
        'max_ses_7T': '',
        'median_ses_3T': '',
        'median_ses_7T': ''
    }

    group_summary = pd.concat([group_summary, pd.DataFrame([total_row], index=['TOTAL'])])

    group_summary = group_summary.sort_values('num_px', ascending=False)

    # save to csv
    if save_pth is not None:
        output_csv = os.path.join(save_pth, f"{save_name}_{pd.Timestamp.now().strftime('%d%b%Y')}.csv")
        group_summary.to_csv(output_csv, header=True)
        print(f"[grp_summary] Saved participant summary to {output_csv}")
    if toPrint:
        print("-"*40)
        pd.set_option('display.max_columns', None)     # Show all columns
        pd.set_option('display.width', None)          # No width limit
        pd.set_option('display.max_colwidth', None)   # No column width limit
        pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines

        print(group_summary)

def dupSES(df, uniqCols, mergeCols):
    """
    Identify and resolve repeated session codes for the same participant at the same study

    Input:
        df: DataFrame containing the session data
        uniqCols: List of columns whose combination should be unique in each row
        mergeCols: List of columns whose values may need merging (eg., scanDate)

    Return:
        DataFrame with resolved repeated sessions
    """
    
    if df.duplicated(subset=uniqCols).any():
        n_repeated = df.duplicated(subset=uniqCols, keep=False).groupby([df[c] for c in uniqCols]).any().sum()
        print(f"[dupSES] WARNING: There are {n_repeated} participant-study combinations with repeated rows:")
        dup_rows = df[df.duplicated(subset=uniqCols, keep=False)]
        grouped = dup_rows.groupby(['MICS_ID', 'PNI_ID', 'study', 'SES']).size().reset_index(name='count')
        for _, row in grouped.iterrows():
            # Get all rows for this individual/session
            rows = dup_rows[
                (dup_rows['MICS_ID'] == row['MICS_ID']) &
                (dup_rows['PNI_ID'] == row['PNI_ID']) &
                (dup_rows['study'] == row['study']) &
                (dup_rows['SES'] == row['SES'])
            ]
            merge_info = []
            for col in mergeCols:
                vals = rows[col].dropna().unique()
                # Conflict resolution logic
                if len(vals) > 1:
                    # If column name contains 'date', resolve by earliest date
                    if 'date' in col.lower():
                        # Try to parse dates, fallback to string sort if fails
                        try:
                            parsed = pd.to_datetime(vals, errors='coerce', dayfirst=True) # d.m.y format
                            chosen = vals[parsed.argmin()] if parsed.notna().any() else sorted(vals)[0]
                        except Exception:
                            chosen = sorted(vals)[0]
                        method = "earliest date"
                    else:
                        # Default: use first value (could add more strategies)
                        chosen = sorted(vals)[0]
                        method = "first (sorted) value"
                    # Update all rows for this group to use the chosen value
                    df.loc[
                        (df['MICS_ID'] == row['MICS_ID']) &
                        (df['study'] == row['study']) &
                        (df['SES'] == row['SES']),
                        col
                    ] = chosen
                else:
                    chosen = vals[0] if len(vals) > 0 else None
                    method = "unique/no conflict"
                merge_info.append(f"{col}={list(vals)} -> {chosen} ({method})")
            merge_str = "; ".join(merge_info)
            print(f"\t[{row['MICS_ID']}={row['PNI_ID']}-{row['study']}] ses-{row['SES']} (x{row['count']}): {merge_str}")
        
        df = df.drop_duplicates(subset=uniqCols, keep='first').reset_index(drop=True) # remove duplicated rows

    return df

def rmvNADate(df, dateCol):
    """
    Remove rows with missing scan date from the DataFrame.

    Input:
        df: DataFrame to process
        dateCol: Name of the date column to check for missing values
    """
    n_missing_dates = df[dateCol].isna().sum()
    if n_missing_dates != 0:
        missing_dates = df[df[dateCol].isna()]
        print(missing_dates[['MICS_ID', 'study', dateCol]])
        count_7T = (missing_dates['study'] == '7T').sum()
        count_3T = (missing_dates['study'] == '3T').sum()
        print(f"\n[rmvNADate] WARNING. {n_missing_dates} rows have missing scan dates:\n\t{count_7T} from 7T\n\t{count_3T} from 3T")
        print("[rmvNADate] Removing the following participants (study of the removed row):")
        missing_ids = missing_dates['MICS_ID'].unique()
        df = df.dropna(subset=[dateCol])
        for mid in missing_ids:
            # there should be at least one row with each study code
            studies = df[df['MICS_ID'] == mid]['study'].unique() # remaining studies for that ID
            if len(studies) < len(df['study'].unique()): # if this has ID has fewer unique studies than those present in the data, remove all rows for that ID
                print(f"\t{mid} (removed study: {studies})")
                df = df[df['MICS_ID'] != mid]
    else:
        print("\n[rmvNADate] No rows with missing scan dates found.")

    return df

def mergeCols(df):
    """
    Merge columns with similar names (case-insensitive, ignoring whitespace) by prioritizing non-null values.
    For each group of similar columns, select the column with the shortest name (or lowercase if tied),
    and fill its missing values with values from the other columns in the group (row-wise).
    Drops the other columns in the group.

    Special logic: If both 'date' and 'scanDate' columns exist (case-insensitive), merge them into a single 'date' column,
    preferring non-null and earliest date values, and print out the merge process.

    Input:
        df: DataFrame with potential duplicate/similar columns

    Output:
        df: DataFrame with merged columns
    """


    # Normalize column names: lower case, strip whitespace
    def norm(col):
        return re.sub(r'\s+', '', col).lower()

    cols = list(df.columns)
    norm_map = {}
    for col in cols:
        key = norm(col)
        norm_map.setdefault(key, []).append(col)

    # Special handling for 'date' and 'scanDate'
    date_cols = [c for c in cols if norm(c) in ('date', 'scandate')]
    if len(date_cols) >= 2:
        # Only proceed if both columns exist
        main_col = [c for c in date_cols if norm(c) == 'date']
        scan_col = [c for c in date_cols if norm(c) == 'scandate']
        if main_col and scan_col:
            main_col = main_col[0]
            scan_col = scan_col[0]
            # If 'date' column does not exist, create it
            if 'date' not in df.columns:
                df['date'] = np.nan
            # Only operate on rows where values differ and both are not null
            mask = (df[main_col].notna() & df[scan_col].notna() & (df[main_col] != df[scan_col]))
            if mask.any():
                print(f"[mergeCols] Merging '{main_col}' and '{scan_col}' into 'date' for rows where values differ")
                for idx, row in df[mask].iterrows():
                    val1 = row[main_col]
                    val2 = row[scan_col]
                    d1 = pd.to_datetime(val1, errors='coerce', dayfirst=True)
                    d2 = pd.to_datetime(val2, errors='coerce', dayfirst=True)
                    chosen = None
                    method = ""
                    if pd.notna(d1) and pd.notna(d2):
                        if d1 <= d2:
                            chosen = val1
                            method = "earliest (date)"
                        else:
                            chosen = val2
                            method = "earliest (scanDate)"
                    elif pd.notna(d1):
                        chosen = val1
                        method = "date only"
                    elif pd.notna(d2):
                        chosen = val2
                        method = "scanDate only"
                    else:
                        chosen = val1 if pd.notna(val1) else val2
                        method = "non-date fallback"
                    df.at[idx, 'date'] = chosen
                    # Use .get to avoid KeyError for missing columns
                    mics_id = row.get('MICS_ID', 'NA')
                    pni_id = row.get('PNI_ID', 'NA')
                    study = row.get('study', 'NA')
                    ses = row.get('SES', 'NA')
                    print(f"\t[mergeCols] [{mics_id}={pni_id}-{study}] ses-{ses}: {main_col}={val1}, {scan_col}={val2} -> {chosen}")
            # For all other rows, fill 'date' with available value (prefer main_col, then scan_col)
            for idx, row in df.iterrows():
                if pd.isna(df.at[idx, 'date']):
                    val1 = row[main_col]
                    val2 = row[scan_col]
                    df.at[idx, 'date'] = val1 if pd.notna(val1) else val2
            # Drop both original columns except 'date'
            drop_cols = [c for c in [main_col, scan_col] if c != 'date']
            df.drop(columns=drop_cols, inplace=True)
            # Remove from norm_map so not merged again
            for c in drop_cols:
                norm_map.pop(norm(c), None)
            if main_col != 'date':
                norm_map.pop(norm(main_col), None)
            if scan_col != 'date':
                norm_map.pop(norm(scan_col), None)

    # Merge other similar columns
    for group in norm_map.values():
        if len(group) > 1:
            # Skip if group contains both 'date' and 'scanDate' (already handled)
            if set([norm(c) for c in group]) == set(['date', 'scandate']):
                continue
            # Choose the column with shortest name, or lowercase if tied
            group_sorted = sorted(group, key=lambda x: (len(x), x.lower()))
            main_col = group_sorted[0]
            other_cols = [c for c in group if c != main_col]
            print(f"[mergeCols] Merging columns: {group} -> '{main_col}' (date/scandate in group: {any(norm(c) in ['date', 'scandate'] for c in group)})")
            # Fill missing values in main_col from other columns, row-wise
            df[main_col] = df[group].bfill(axis=1).iloc[:, 0]
            # Drop the other columns
            df.drop(columns=other_cols, inplace=True)
    return df

def carryVals(df, var):
    """
    Carry forward and backward fill demographic variables for each participant.

    For each participant (grouped by 'MICS_ID'), if multiple unique non-null values are found for the variable:
      - Use the value with the shortest number of characters.
      - If there is a tie, use the first value encountered.
      - Print a warning indicating all found values and which one was used, including study and session number if available.

    Input:
        df: DataFrame containing demographic information
        var: Column name to carry forward/backward fill

    Output:
        df: DataFrame with the variable filled per participant
    """
    df[var] = df.groupby('MICS_ID')[var].transform(lambda x: x.ffill().bfill())
    # Check for multiple unique values for the same ID
    dupes = df.groupby('MICS_ID')[var].nunique()
    multi_val_ids = dupes[dupes > 1].index
    for mid in multi_val_ids:
        vals = df[df['MICS_ID'] == mid][var].dropna().unique()
        # Choose value with shortest number of characters, if tie, use first
        vals_sorted = sorted(vals, key=lambda v: (len(str(v)), str(v)))
        chosen = vals_sorted[0]
        pni_id = df[df['MICS_ID'] == mid]['PNI_ID'].iloc[0] if 'PNI_ID' in df.columns else ''
        study = df[df['MICS_ID'] == mid]['study'].iloc[0] if 'study' in df.columns else ''
        ses = df[df['MICS_ID'] == mid]['SES'].iloc[0] if 'SES' in df.columns else ''
        used_idx = list(vals).index(chosen) + 1
        print(f"\t[carryVals] WARNING: [{mid}={pni_id} Study: {study}-ses{ses}] {var} : {' | '.join(map(str, vals))} --> {chosen}")
    
    # Use the chosen value for each group
    def choose_val(x):
        non_nulls = x.dropna()
        if len(non_nulls.unique()) > 1:
            vals_sorted = sorted(non_nulls.unique(), key=lambda v: (len(str(v)), str(v)))
            return vals_sorted[0]
        elif len(non_nulls) > 0:
            return non_nulls.iloc[0]
        else:
            return x
    df[var] = df.groupby('MICS_ID')[var].transform(choose_val)
    return df

def group(df, ID_col="MICS_ID", out_col="grp", save_pth=None, MFCL_col="Epilepsy classification:Focal,Generalized", lobe_col="Epileptogenic focus confirmed by the information of (sEEG/ site of surgical resection/ Ictal EEG abnormalities +/. MRI findings): FLE=forntal lobe epilepsy and cingulate epilepsy, CLE:central/midline epilepsy,ILE: insular epilepsy, mTLE=mesio.temporal lobe epilepsy, nTLE=neocortical lobe epilepsy, PQLE=posterior quadrant lobe epilepsy , multifocal epilepsy,IGE=ideopathic lobe epilepsy,unclear)", lat_col="Lateralization of epileptogenic focus"):
    """
    Requires pandas as pd

    Inputs:
    df: str or pd.DataFrame
        demographics data
    ID_col: str
        Column name for the ID
    out_col: str
        Column name for the output group classification. 
        Options: 'grp' returns high-level grouping. All other col_names return detailed grouping.
    MFCL_col: str
        Column name for the multifocal classification
    lobe_col: str  
        Column name for the lobe classification
    lat_col: str
        Column name for the lateralization classification

    Outputs:
    df: pd.DataFrame
        DataFrame with the group classification added
    """

    import pandas as pd
    import os

    print("[group] Identifying participant groups")
    # check if df is a string (path) or dataframe
    if isinstance(df, str):
        # check if file exists
        if not os.path.isfile(df):
            raise ValueError(f"[group] Error: {df} does not exist.")
        
        # read in file
        df = pd.read_csv(df, dtype=str)
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise ValueError("[group] Error: df must be a string (path) or dataframe")

    df[out_col] = df.apply(
        lambda row: f"PATTERN NOT RECOGNIZED: lobe={row.get(lobe_col, None)}, lat={row.get(lat_col, None)}, MFCL={row.get(MFCL_col, None)}",
        axis=1
    )

    if ID_col == "MICS_ID":
        ctrl_ptrn = ["HC"]
    elif ID_col == "PNI_ID":
        ctrl_ptrn = ["Pilot", "PNC"]
    else:
        raise ValueError("[group] Error: ID_col must be 'MICS_ID' or 'PNI_ID'")

    
    if out_col == "grp":
        print("\tReturning highlevel grouping to column: ", out_col)
       
        df.loc[df[ID_col].astype(str).str.contains('|'.join(ctrl_ptrn), na=False), out_col] = 'CTRL'
        
        # TLE and mTLE: L, left, R, right, unclear
        df.loc[
            (df[lobe_col].astype(str).str.lower().isin(['tle', 'mtle'])) & 
            (df[lat_col].astype(str).str.lower().str.contains('l|left|r|right|l>r|r>l|bl|bilateral|unclear', na=False)), 
            out_col
        ] = 'TLE'
        
        # FLE: L, R, right, unclear
        df.loc[
            (df[lobe_col] == 'FLE') & 
            (df[lat_col].astype(str).str.contains('l|r|right|unclear', case=False, na=False)), 
            out_col
        ] = 'FLE'
        
        # PLE: L, R, right, unclear
        df.loc[
            ((df[lobe_col] == 'PLE')) &
            (df[lat_col].astype(str).str.contains('l|r|right|unclear', case=False, na=False)),
            out_col
        ] = 'PLE'

        # PQLE: L, R, right, unclear
        df.loc[
            ((df[lobe_col] == 'PQLE')) &
            (df[lat_col].astype(str).str.contains('l|r|right|unclear', case=False, na=False)),
            out_col
        ] = 'PQLE'

        # Unclear: L, R, right, unclear
        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & (df[lat_col] == 'L'), out_col] = 'UKN'
        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & ((df[lat_col] == 'R') | (df[lat_col].astype(str).str.lower() == 'right')), out_col] = 'UKN'
        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & (df[lat_col].astype(str).str.contains('unclear', na=False)), out_col] = 'UKN'
        
        # Multifocal
        df.loc[(df[MFCL_col] == 'Multifocal'), out_col] = 'MFCL'
    
    else:  
        print("\tReturning detailed grouping to column: ", out_col) 
        
        df.loc[df[ID_col].astype(str).str.contains('|'.join(ctrl_ptrn), na=False), out_col] = 'CTRL'
        
        # Combine TLE and mTLE, label as TLE
        df.loc[
            (df[lobe_col].astype(str).str.lower().isin(['tle', 'mtle'])) & (df[lat_col] == 'L'),
            out_col
        ] = 'TLE_L'
        df.loc[
            (df[lobe_col].astype(str).str.lower().isin(['tle', 'mtle'])) & ((df[lat_col] == 'R') | (df[lat_col].astype(str).str.lower() == 'right')),
            out_col
        ] = 'TLE_R'
        df.loc[
            (df[lobe_col].astype(str).str.lower().isin(['tle', 'mtle'])) & (df[lat_col].astype(str).str.contains('unclear', na=False)),
            out_col
        ] = 'TLE_U'

        df.loc[(df[lobe_col] == 'FLE') & (df[lat_col] == 'L'), out_col] = 'FLE_L'
        df.loc[(df[lobe_col] == 'FLE') & ((df[lat_col] == 'R') | (df[lat_col].astype(str).str.lower() == 'right')), out_col] = 'FLE_R'

        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & (df[lat_col] == 'L'), out_col] = 'UKN_L'
        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & ((df[lat_col] == 'R') | (df[lat_col].astype(str).str.lower() == 'right')), out_col] = 'UKN_R'
        df.loc[(df[lobe_col].astype(str).str.contains('unclear', na=False)) & (df[lat_col].astype(str).str.contains('unclear', na=False)), out_col] = 'UKN_U'
        
        df.loc[(df[MFCL_col] == 'Multifocal'), out_col] = 'MFCL'
        df.loc[(df[lobe_col] == 'TLE') & (df[lat_col] == 'L>R'), out_col] = 'TLE_L'
        df.loc[(df[lobe_col] == 'TLE') & (df[lat_col] == 'R>L'), out_col] = 'TLE_R'
        df.loc[(df[lobe_col] == 'TLE') & (df[lat_col] == 'BL'), out_col] = 'TLE_BL'
    
    return df

def grpChk(df, grp_cols=['grp', 'grp_detailed']):
    """
    Check that all cases are assigned a group.
    If not, print warning

    Input:
        df: DataFrame to check
        grp_col: List of group columns to check
    """
    missing = []
    for col in grp_cols:
        df_missing = df[df[col] == ''][["MICS_ID", "PNI_ID", "SES", col]]
        if not df_missing.empty:
            missing.append(df_missing)  # Use append, not extend

    if len(missing) > 0:
        missing_df = pd.concat(missing)
        print("[grpChk] The following cases are missing group assignments:")
        print(missing_df[["MICS_ID", "PNI_ID", "SES", "grp", "grp_detailed", 
            "Epileptogenic focus confirmed by the information of (sEEG/ site of surgical resection/ Ictal EEG abnormalities +/. MRI findings): FLE=forntal lobe epilepsy and cingulate epilepsy, CLE:central/midline epilepsy,ILE: insular epilepsy, mTLE=mesio.temporal lobe epilepsy, nTLE=neocortical lobe epilepsy, PQLE=posterior quadrant lobe epilepsy , multifocal epilepsy,IGE=ideopathic lobe epilepsy,unclear)", 
            "Lateralization of epileptogenic focus"]])
        return False
    else:
        print("[grpChk] All assigned a group")
        return True


# QC Functions ####################################################
def mk_qcSheet(df, fts, studies, ctx_surf_qc, save_name, save_pth, currentQC=None):
    """
    From a dataframe with demographic information and features of interest, create a QC sheet to complete manually.
    TODO. Add also surface QC column as well as paths to surfaces (cortical, hippocampal).
    NOTE. Inclues minimum dice score for hippocampal surfaces as QC value

    NOTE. Currently assumes df has columns 'MICS_ID', 'PNI_ID', 'study', 'SES'
    NOTE. For hippocampal surfaces, import dice score as QC value (values > 0.7 are good, < 0.7 need checking, 0 = failed)

    Parameters:
        df: pd.DataFrame
            DataFrame with demographic information and features of interest
        fts: lst
            List of strings refering to features of interest whose source volumes should be included in the QC sheet
        studies: lst
            List of dictionary items with study information.
        ctx_surf_qc: lst
            List of dictionary items with details about existing sheets with QC for cortical surfaces
        

        save_name: str
            Name of the output QC file (without path)
        save_pth: str
            Path to save the output QC file
        currentQC: str
            Path to an existing QC sheet to pull volume/surface QC from
            NOTE. Assumes presences of columns 'UID', 'ID', 'SES'
    Returns:
        df: pd.DataFrame
            DataFrame with demographic information and features of interest
        pth: str
            Path to the saved QC file
        
    """

    import os
    import datetime
    import numpy as np
    import tTsTGrpUtils as tsutil
    
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")

    logger = tsutil._get_file_logger(__name__, log_file_path)
    print(f"[mk_qcSheet] Log file: {log_file_path}")
    logger.info(f"[mk_qcSheet] Saving log to {log_file_path}")
    logger.info(f"Start time: {start}")

    logger.info("Parameters:")
    logger.info(f"\tdf shape: <{df.shape}>")
    logger.info(f"\tfeatures: {fts}")
    logger.info(f"\tstudies: {studies}")
    logger.info(f"\tctx_surf_qc: {ctx_surf_qc}")
    logger.info(f"\tsave_name: {save_name}")
    logger.info(f"\tsave_pth: {save_pth}\n")

    try:

        assert isinstance(fts, list), "[mk_qcSheet] Error: fts must be a list of strings" 
        
        vol_names = tsutil.get_RawVolumeNames(fts)
        logger.info(f"Volumes identified for QC: {vol_names}")

        # create ID column
        df_pths_cp = df
        df_pths_cp['ID'] = df_pths_cp.apply(lambda row: row['MICS_ID'] if row['study']=='3T' else row['PNI_ID'] if row['study']=='7T' else 'UNKNOWN STUDY', axis=1)
        df_pths_cp['ID'].head()

        ID_cols = ['UID', 'study', 'ID', 'SES', 'Date']

        # Initialize qc_sheet
        qc_sheet = pd.DataFrame(columns=ID_cols + vol_names)

        # Add columns
        for col in ID_cols:
            if col in df_pths_cp.columns:
                qc_sheet[col] = df_pths_cp[col]
            else:
                logger.info(f"Column {col} not in input df. Not including in QC_sheet.")

        for vol in vol_names:
            qc_sheet[vol] = np.nan # default value
            qc_sheet[f"{vol}_pth"] = qc_sheet[vol]
        
        ctx_surfs = [surf for surf in ["pial", "white"]]
        hipp_surfs = [surf for surf in ["outer", "inner"]]
        
        for surf in ctx_surfs + hipp_surfs:
            qc_sheet[f"{surf}_L_pth"] = np.nan
            qc_sheet[f"{surf}_R_pth"] = np.nan
        
        # Add existing surface QC information
        dict_tT_surfQC = next((s for s in ctx_surf_qc if s['STUDY'] == '3T'))
        df_tT_surfQC = pd.read_excel(dict_tT_surfQC['PATH'], sheet_name = dict_tT_surfQC['SHEET'], dtype=str).rename(columns={dict_tT_surfQC['ID']: 'ID', dict_tT_surfQC['SES']: 'SES', dict_tT_surfQC['QC_col']: 'surf_QC'})

        dict_sT_surfQC = next((s for s in ctx_surf_qc if s['STUDY'] == '7T'))
        df_sT_surfQC = pd.read_excel(dict_sT_surfQC['PATH'], sheet_name = dict_sT_surfQC['SHEET'], dtype=str).rename(columns={dict_sT_surfQC['ID']: 'ID', dict_sT_surfQC['SES']: 'SES', dict_sT_surfQC['QC_col']: 'surf_QC'})
        df_sT_surfQC['surf_QC'] = df_sT_surfQC['surf_QC'].fillna('2') # If surf_QC is NAN for 7T, assume it is acceptable. There are comments in problematic cases

        # Combine surface QC dfs
        common_cols = ['ID', 'SES', 'surf_QC']
        qc_surf = pd.concat([df_tT_surfQC[common_cols], df_sT_surfQC[common_cols]], ignore_index=True)
        
        logger.info(f"Loaded existing cortical surface QC values from {dict_tT_surfQC['PATH']} and {dict_sT_surfQC['PATH']}")
        logger.info(f"QC surface sheet has shape: {qc_surf.shape}")

        qc_sheet = qc_sheet.merge(qc_surf, on=['ID', 'SES'], how='left')

        # Some ID-SES are not present in the recent 7T QC sheet. 
        # However, all of such ommissions before 26 Sept 2025 are used to train the segmentation model and can be assumed good. 
        # For all rows with study = 7T and surf_QC is NaN, set surf_QC to 2 if Date < 26 Sept 2025
        qc_sheet['Date'] = pd.to_datetime(qc_sheet['Date'], errors='coerce', dayfirst=True)
        cutoff_date = pd.to_datetime("2025-09-26")
        condition = (qc_sheet['study'] == '7T') & (qc_sheet['surf_QC'].isna()) & (qc_sheet['Date'] < cutoff_date)
        qc_sheet.loc[condition, 'surf_QC'] = '2'
        qc_sheet.rename(columns={'surf_QC': 'QC_ctxSurf'}, inplace=True)

        logger.info(f"QC sheet after merging surface QC has shape: {qc_sheet.shape}")

        counter = 0
        for idx, row in qc_sheet.iterrows(): # Add paths to volumes, surfaces and previous QC values of surface
            counter += 1
            if counter % 10 == 0:
                print(f"Participant {counter}/{qc_sheet.shape[0]}...")
            id = row['ID']
            ses = row['SES']
            logger.info(f"{id}-{ses} [row: {idx}]")
            
            # determine root
            study_code = row['study']
            study_dict = next((s for s in studies if s['study'] == study_code), None)
            if study_dict is None:
                logger.error(f"[mk_qcSheet] Error: No study information found for '{study_code}'")
                continue
            
            root_raw = study_dict.get('dir_root', None)
            root_dir = root_raw + study_dict.get('dir_deriv', None)
            root_mp = root_dir + study_dict.get('dir_mp', None)
            root_hu = root_dir + study_dict.get('dir_hu', None)

            if root_raw is None or root_mp is None:
                logger.error(f"[mk_qcSheet] Error: Missing directory keys in study dictionaries. Expected keys: 'dir_root', 'dir_deriv', 'dir_mp', 'dir_hu'")
                continue

            for vol in vol_names: # get micapipe volumes in nativepro space
                vol_path = tsutil.get_mapVol_pth(root_mp, row['ID'], row['SES'], study = study_code, feature = vol, raw=False, space="nativepro")
                             
                if vol_path is None:
                    logger.info(f"\t{vol}: No path found")
                    continue
                elif not tsutil.chk_pth(vol_path):
                    logger.warning(f"\t{vol}: Path does not exist: {vol_path}")
                    vol_path = "NA"
                     
                qc_sheet.at[idx, f"{vol}_pth"] = vol_path
            logger.info(f"\tVolume paths processed.\n")

            # add path to surfaces
            for lbl in ctx_surfs:
                pth_L, pth_R = tsutil.get_surf_pth(root = root_mp, sub = id, ses = ses, 
                                                     lbl = lbl, space="nativepro", surf = "fsLR-5k") # get path to each surface
                if pth_L is None or pth_R is None:
                    logger.info(f"\t{lbl} cortical surface: No path found")
                    continue
                if not tsutil.chk_pth(pth_L):
                    logger.warning(f"\t{lbl} cortical surface L: Path does not exist: {pth_L}")
                    pth_L = "NA"
                if not tsutil.chk_pth(pth_R):
                    logger.warning(f"\t{lbl} cortical surface R: Path does not exist: {pth_R}")
                    pth_R = "NA"

                qc_sheet.at[idx, f"{lbl}_L_pth"] = pth_L
                qc_sheet.at[idx, f"{lbl}_R_pth"] = pth_R
            logger.info(f"\tCortical surfaces processed.")

            for lbl in hipp_surfs:
                pth_L, pth_R = tsutil.get_surf_pth(root = root_hu, sub = id, ses = ses, 
                                                     lbl = lbl, space="T1w", surf = "den-0p5mm") # get path to each surface
                if pth_L is None or pth_R is None:
                    logger.info(f"\t{lbl} hippocampal surface: No path found")
                    continue
                if not tsutil.chk_pth(pth_L):
                    logger.warning(f"\t{lbl} hippocampal surface L: Path does not exist: {pth_L}")
                    pth_L = "NA"
                if not tsutil.chk_pth(pth_R):
                    logger.warning(f"\t{lbl} hippocampal surface R: Path does not exist: {pth_R}")
                    pth_R = "NA"

                qc_sheet.at[idx, f"{lbl}_L_pth"] = pth_L
                qc_sheet.at[idx, f"{lbl}_R_pth"] = pth_R
            logger.info(f"\tHippocampal surfaces processed.")
            
            # extract dice score for hippocampal surfaces
            d_L, d_R = tsutil.get_huDice(root = root_hu, sub = id, ses = ses, rtn_ERR = True)
            logger.info(f"\tDice object types: {type(d_L)}, {type(d_R)}")
            
            if isinstance(d_L, type(None)) or isinstance(d_R, type(None)):
                logger.warning(f"\t{d_L}") # d_L holds error message from get_huDice
            else:
                d_min = min(d_L, d_R)
                logger.info(f"\tHippocampal dice L: {d_L:0.3f} | R: {d_R:0.3f} -> min: {d_min:0.3f}")
                qc_sheet.at[idx, 'QC_hipSurf'] = f"{d_min:0.3f}"
            
            # sort columns
            cols = qc_sheet.columns.tolist()
            cols_pth = [col for col in cols if col.endswith('_pth')]
            cols_non_pth = [col for col in cols if not col.endswith('_pth')]
            new_order = cols_non_pth + cols_pth
            qc_sheet = qc_sheet[new_order]
            logger.info(f"\tQC sheet shape post: {qc_sheet.shape}")    
            logger.info(f"\n")
        
        # Carry over values from QC_sheets that are partially filled in
        if currentQC is not None:
            if os.path.isfile(currentQC):
                df_currentQC = pd.read_csv(currentQC, dtype=str, keep_default_na=False, na_values=[])
                # match on ID, SES
                # NOTE. UID assignment may differ between sheets, so do not match on that
                logger.info(f"[mk_qcSheet] Merging existing QC sheet from {currentQC}")
                logger.info(f"Existing QC sheet shape: {df_currentQC.shape}")
                # get values from {vol} and QC_ctxsurf
                # should not add new rows
                assert 'UID' in df_currentQC.columns, "[mk_qcSheet] Error: currentQC must have column 'UID'"
                assert 'ID' in df_currentQC.columns, "[mk_qcSheet] Error: currentQC must have column 'ID'"
                assert 'SES' in df_currentQC.columns, "[mk_qcSheet] Error: currentQC must have column 'SES'"

                extract_col = vol_names + ['QC_ctxSurf', 'QC_hipSurf']
                extract_col = [col for col in extract_col if col in df_currentQC.columns]
                df_currentQC_sub = df_currentQC[['ID', 'SES'] + extract_col]
                logger.info(f"Extracting columns from existing QC sheet: {extract_col}")
                logger.info(f"Existing QC sheet shape: {df_currentQC_sub.shape}")

                # print col 'SES'
                if 'SES' in df_currentQC_sub.columns:
                    logger.info(f"Existing QC sheet SES values: {df_currentQC_sub['SES'].unique()}")
                    logger.info(f"New QC sheet SES values: {qc_sheet['SES'].unique()}")

                # For each extract_col, fill missing values in qc_sheet from df_currentQC_sub
                for col in extract_col:
                    if col in qc_sheet.columns and col in df_currentQC_sub.columns:
                        # Create a mapping from (ID, SES) to value in currentQC
                        value_map = df_currentQC_sub.set_index(['ID', 'SES'])[col].to_dict()
                        # Fill missing values in qc_sheet[col] using the mapping
                        mask = qc_sheet[col].isna() | (qc_sheet[col] == '')  # treat empty string as missing
                        for idx in qc_sheet[mask].index:
                            key = (qc_sheet.at[idx, 'ID'], qc_sheet.at[idx, 'SES'])
                            val = value_map.get(key, None)
                            if val not in [None, '']:
                                qc_sheet.at[idx, col] = val

                logger.info(f"QC sheet shape after merging existing QC: {qc_sheet.shape}")
            else:
                logger.warning(f"[mk_qcSheet] WARNING: currentQC path does not exist: {currentQC}. Skipping merge.")

        # save
        out_pth = os.path.join(save_pth, f"{save_name}_{start}.csv")
        qc_sheet.to_csv(out_pth, index=False)

        logger.info(f"[mk_qcSheet] Saved QC sheet: {out_pth}")
        print(f"[mk_qcSheet] Saved QC sheet: {out_pth}")
        print(f"Complete QC sheet manually then proceed.")
        print("Values to indicate: 0 (unacceptable), 1 (poor), or 2 (acceptable).")
    except Exception as e:
        logger.error({e}, exc_info=True)
        print(f"[mk_qcSheet] EXITING WITH ERROR. See log file {log_file_path}")
        return None, None
    
    return qc_sheet, out_pth