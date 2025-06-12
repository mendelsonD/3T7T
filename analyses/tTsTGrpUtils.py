# Finding paths and checking if they exist

def chk_pth(pth):
    """
    Check if the path exists and is a file.
    
    inputs:
        pth: path to check
    
    outputs:
        True if the path exists and is a file, False otherwise
    """
    
    import os
    
    if os.path.exists(pth) and os.path.isfile(pth):
        return True
    else:
        return False
    

def mp_mapsPth(dir, sub, ses, hemi, surf, lbl, ft):
    """
    Returns path to maps along a surface produced by micapipe.
    """
    if lbl == "thickness":
        return f"{dir}/sub-{sub}_ses-{ses}_hemi-{hemi}_surf-{surf}_label-{lbl}.func.gii"
    else:
        return f"{dir}/sub-{sub}_ses-{ses}_hemi-{hemi}_surf-{surf}_label-{lbl}_{ft}.func.gii"


def get_surf_pth(root, sub, ses, lbl, res="fsLR-32k"):
    """
    Get the nativepro surface positions for the left and right hemispheres.
    
    input:
        root: root directory of study derivative of interest (ie. study's micapipe or hippunfold root directory)
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        res: surface type and resolution (e.g., "fsLR-32k", "fsLR-5k", "0p5mm")
        lbl: surface label (e.g., "white", "pial", "midthickness", "hipp_inner", "hipp_outer)
    """

    if  "micapipe" in root:
        lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-nativepro_surf-{res}_label-{lbl}.surf.gii"
        rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-nativepro_surf-{res}_label-{lbl}.surf.gii"
    elif "hippunfold" in root:
        #print("Hipp detected")
        lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-T1w_den-{res}_label-hipp_{lbl}.surf.gii"
        rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-T1w_den-{res}_label-hipp_{lbl}.surf.gii"
    else:
        raise ValueError("Invalid root directory. Choose from 'micapipe' or 'hippunfold'.")

    return lh, rh 

def get_vol_pth(mp_root, sub, ses, metric):
    """
    Get path to map (volume in nativepro space).
    
    input:
        mp_root: root directory to micapipe output
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        metric: type of map to retrieve (e.g., "T1map", "FLAIR", "ADC", "FA")
            File naming pattern:
                T1map: map-T1
                FLAIR: map-flair
                ADC: DTI_map-ADC
                FA: DTI_map-FA

    output:
        Path to the map file in nativepro space.
    """

    if metric == "T1map": mtrc = "map-T1map"
    elif metric == "FLAIR": mtrc = "map-flair"
    elif metric == "ADC": mtrc = "model-DTI_map-ADC"
    elif metric == "FA": mtrc = "model-DTI_map-FA"
    else:
        raise ValueError(f"Invalid metric: {metric}. Choose from 'T1map', 'FLAIR', 'ADC', or 'FA'.")

    return f"{mp_root}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_space-nativepro_{mtrc}.nii.gz"

    
# when working, add to Utils scripts
def get_map_pth(root, deriv_fldr, sub, ses, feature, label="midthickness", surf="fsLR-5k", space="nativepro", hemi="LR", check_pth=True,silence=True):
    """
    Get the path to the surface data for a given subject and session.
    Assumes BIDS format of data storage.

    inputs:
        root: root directory of the study
        deriv_fldr: name of derivative folder containing the surface data
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        surf: surface type and resolution (e.g., fsLR-32k, fsLR-5k)
        label: surface label (e.g., "pial", "white", "midThick")
        space: space of the surface data (e.g., "nativepro", "fsnative")
        hemi: hemisphere to extract (default is "LR" for both left and right hemispheres)

        check_pth: whether to check if the path exists (default is True)
        silence: whether to suppress print statements (default is True)
    outputs:
        path to the surface data files
    """

    # make surf to lower case
    label = label.lower()

    # ensure that label is well defined
    if label == "thickness":
        label = "thickness"
    elif label == "pial":
        label = "pial"
    elif label == "white":
        label = "white"
    elif label == "midthick" or label == "midthickness":
        label = "midthickness"
    else:
        raise ValueError(f"{label} Invalid label type. Choose from 'pial', 'white', 'midThick' or 'thickness'.")
    
    # construct the path to the surface data file
    hemi = hemi.upper()
    if hemi == "LEFT" or hemi == "L":
        hemi = "L"
    elif hemi == "RIGHT" or hemi == "R":
        hemi = "R"
    elif hemi != "LR":
        raise ValueError("Invalid hemisphere. Choose from 'L', 'R', or 'LR'.")

    # handle hippunfold naming convention
    if "micapipe" in deriv_fldr.lower():
        pth = f"{root}/derivatives/{deriv_fldr}/sub-{sub}/ses-{ses}/maps"
        if hemi == "L" or hemi == "R":
            pth = mp_mapsPth(dir=pth, sub=sub, ses=ses, hemi=hemi, surf=surf, lbl=label, ft=feature)
            if not silence: print(f"[get_map_pth] Returning paths for both hemispheres ([0]: L, [1]: R)")
            
        else:         
            pth_L = mp_mapsPth(dir=pth, sub=sub, ses=ses, hemi="L", surf=surf, lbl=label, ft=feature)
            pth_R = mp_mapsPth(dir=pth, sub=sub, ses=ses, hemi="R", surf=surf, lbl=label, ft=feature)        
            pth = [pth_L, pth_R]
            if not silence: print(f"[get_map_pth] Returning paths for both hemispheres ([0]: L, [1]: R)")
    elif "hippunfold" in deriv_fldr.lower():
        raise ValueError("Hippunfold derivative not yet implemented. Need to create feature maps using hippunfold surfaces.")
        
        # space usually: "T1w"
        # surf usually: "fsLR"
        # label options: "hipp_outer", "hipp_inner", "hipp_midthickness"

        pth = f"{root}/derivatives/{deriv_fldr}/sub-{sub}/ses-{ses}/surf"

        if hemi == "L" or hemi == "R":
            pth = f"{pth}/sub-{sub}_ses-{ses}_hemi-{hemi}_space-{space}_den-{surf}_label-{label}.surf.gii"
            if not silence: print(f"[surf_pth] Returning hippunfold path for {hemi} hemisphere")
        else:
            pth = f"{pth}/sub-{sub}_ses-{ses}_hemi-{hemi}_surf-{surf}_label-{label}_{feature}.func.gii"
            pth_L = f"{pth}/sub-{sub}_ses-{ses}_hemi-L_-{surf}_label-{label}.surf.gii"
            pth_R = f"{pth}/sub-{sub}_ses-{ses}_hemi-R_space-{space}_den-{surf}_label-{label}.surf.gii"
            pth = [pth_L, pth_R]
            if not silence: print(f"[surf_pth] Returning hippunfold paths for both hemispheres ([0]: L, [1]: R)")

    else:
        raise ValueError("Invalid derivative folder. Choose from 'micapipe' or 'hippunfold'.")


    if check_pth:
        if isinstance(pth, list):
            for idx, p in enumerate(pth):
                if not chk_pth(p):
                    if label == "thickness": feature = "(thickness)"
                    print(f"\t[get_map_pth] FILE NOT FOUND (ft: {feature}, sub-{sub}_ses-{ses}): {p}")
                    pth[idx] = "ERROR:" + p
        else:
            if not chk_pth(pth):
                print(f"\t[get_map_pth] FILE NOT FOUND (ft: {feature}, sub-{sub}_ses-{ses}): {pth}")
                pth = "ERROR:" + pth
    
    return pth   

def get_Npths(demographics, study, groups, feature="FA", derivative="micapipe", label="midthickness", hemi="LR", space="nativepro", surf="fsLR-5k"):
    """
    Get path to surface files for individual groups

    Input:
    demographics: dict  regarding demographics file. 
        Required keys: 
            'pth'
            'ID_7T'
            'ID_3T'
            'SES'
            'date'
            'grp'
    study: dict  regarding study.
        Required keys: 
            'name'
            'dir_root'
            'study'
            'dir_mp'
            'dir_hu'
    groups: dict    of groups to extract surfaces for. 
        Each key should be a group name, and the value should be a list of labels in the 'grp' column of demographics file assigned to that group.
    label: str  surface label to extract
    hemi: str  hemisphere to extract. Default is "LR" for both left and right hemispheres.
    space: str  space of the surface data. Default is "nativepro".
    surf: str  surface type and resolution. Default is "fsLR-5k".
    """
    import pandas as pd

    demo = pd.read_csv(demographics['pth'], dtype=str)
    
    out = []

    if derivative == "hippunfold":
        deriv_fldr = study['dir_hu']
    elif derivative == "micapipe":
        deriv_fldr = study['dir_mp']
    else:
        deriv_fldr = study['dir_mp']
        print(f"[get_Npths] WARNING: derivative not recognized. Defaulting to micapipe.")


    for grp_name, grp_labels in groups.items():
        print(f"{study['name']} {grp_name} ({grp_labels})")

        # get IDs for this group
        ids = demo.loc[
            (demo[demographics['grp']].isin(grp_labels)) &
            (demo['study'] == study['study']),
            [ID_col, demographics['SES'], 'study', 'Date']
        ].copy()

        for i, row in ids.iterrows():
            ID = row[ID_col]
            SES = row[demographics['SES']]
            date = row[demographics['date']]
            #print(f"\tsub-{ID}_ses-{SES}")
            pth = get_map_pth(root=study['dir_root'], deriv_fldr=deriv_fldr, sub=ID, ses=SES, label=label, surf=surf, feature=feature, space=space, hemi=hemi)
            # add this pth to the dataframe
            if isinstance(pth, list):
                ids.loc[i, f'pth_L'] = pth[0]
                ids.loc[i, f'pth_R'] = pth[1]
            else:
                ids.loc[i, f'pth_{hemi}'] = pth 
        # if paths are duplicated, then keep only one of those rows
        if hemi == "LR":
            ids = ids.drop_duplicates(subset=[f'pth_L', f'pth_R'])
        else:
            ids = ids.drop_duplicates(subset=[f'pth_{hemi}'])

        # create dictionary item for each group, add to output list
        out.append({
            'study': study['name'],
            'grp': grp_name,
            'grp_labels': grp_labels,
            'label': label,
            'feature': feature,
            'map_pths': ids
        })

    return out

def smooth_map(surf, map, out, kernel=10):
    """
    Apply smoothing to a feature map.

    Input:
        surf: path to the surface file (.surf.gii)
        map: path to unsmoothed feature map file (.func.gii)
        out: output file name stem (root/prefix)
        smth_kernel: smoothing kernel size in mm (default is 10mm)
    """

    import subprocess as sp
    
    out_pth = f"{out}_smth-{kernel}.func.gii"
    
    cmd = [
        "wb_command",
        "-metric-smoothing", 
        surf,
        map, str(kernel), 
        out_pth
        ]
    
    sp.run(cmd, check=True)

    if not chk_pth(out_pth):
        print(f"WARNING: Smoothed map not properly saved. Expected file: {out_pth}")
        return None
    else:
        print(f"\t[smooth_map] Smoothed map saved to: {out_pth}")
        return out_pth


def create_dir(dir_path):
    """
    Create a directory if it does not exist.

    input:
        dir_path: path to the directory to create

    output:
        None
    """
    import os

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"\t[create_dir] Created directory: {dir_path}")
    else:
        return

def add_date(demo_pths, demo):
    """
    Add 'Date' column from demo to demo_pths using overlapping columns.
    
    Parameters:
    - demo_pths: DataFrame containing paths and metadata.
    - demo: DataFrame containing demographic information including 'Date'.
    
    Returns:
    - Updated demo_pths with 'Date' column added.
    """
    
    if 'Date' not in demo_pths.columns:
        print("Adding 'Date' column from demo to demo_pths")
        
        merge_cols = ['MICS_ID', 'PNI_ID', 'study', 'SES', 'grp', 'grp_detailed']
        demo_pths = demo_pths.merge(
            demo[merge_cols + ['Date']],
            on=merge_cols,
            how='left',
            suffixes=('', '_from_demo')
        )
        return demo_pths
    else:
        print("skipping. 'Date' column already exists in demo_pths.")
        return demo_pths

def make_map(sub, ses, surf_pth, vol_pth, smoothing, out_name, out_dir):
    """
    Create feature map from surface and feature volume files.

    input:
        surf_pth: path to the surface file (e.g., midthickness surface)
        vol_pth: path to the volume file (e.g., T1 map, FLAIR, ADC, FA)
        smoothing: vertex value smoothing (in mm)
        out_name: output file name stem
        out_dir: output directory

    output:
        map: a feature map created by projecting the volume data onto the surface.
    """
    import subprocess as sp
    import os
    import nibabel as nib
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.mesh.array_operations import smooth_array

    pth_noSmth = f"{out_dir}/sub-{sub}_ses-{ses}_{out_name}.func.gii"
    # check if directory exists, if not create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"\t\t[make_map] Created output directory: {out_dir}")

    cmd = [
        "wb_command",
        "-volume-to-surface-mapping", vol_pth, surf_pth,
        pth_noSmth,
        "-trilinear"
    ]

    sp.run(cmd, check=True)

    if not chk_pth(pth_noSmth):
        print(f"WARNING: Map not properly saved. Expected file: {pth_noSmth}")
        return None
    else:
        print(f"\t\t[make_map]\t Map saved to: {pth_noSmth}")

    if smoothing is not None and smoothing > 0:
        print(f"\t\t[make_map] Smoothing applied: {smoothing} mm")
        pth_smth = f"{out_dir}/sub-{sub}_ses-{ses}_{out_name}_smth-{smoothing}.func.gii"

        # Load the unsmoothed map and surface geometry
        surf_mesh = read_surface(surf_pth)
        gii = nib.load(pth_noSmth)
        data = gii.darrays[0].data

        # Apply smoothing
        smoothed = smooth_array(surf_mesh, data, kernel='gaussian', sigma=smoothing)

        # Save the smoothed data as a new GIFTI file
        new_gii = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(smoothed.astype('float32'))])
        nib.save(new_gii, pth_smth)

        if not chk_pth(pth_smth):
            print(f"\tWARNING: Smoothed map not properly saved. Expected file: {pth_smth}")
        else:
            print(f"\t[make_map] Map saved to: {pth_smth}")
        return pth_smth
    else:
        return pth_noSmth
    
def clean_pths(dl, method="newest", silent=True):
    """
    Keeps only one session per ID
    input:
        dl (for dictionary list): List of dictionary items (e.g. outputs from get_Npths). 
            These dict should contain a df under the key 'map_pths'
        method: method to use for choosing session.
            "newest": use most recent session
            "oldest": use oldest session in the list
            {number}: session code to use (e.g. '01' or 'a1' etc)

    output:
        dl: List of dictionary items with cleaned dataframes

    """
    dl_out = []
    
    for i, d in enumerate(dl):
        if not silent: print(f"[clean_pths] {d['study']} {d['grp']}: {df.shape}, num unique IDs: {df[ID_col].nunique()}")

        df = d['map_pths']
                
        if d['study'] == "PNI": ID_col = "PNI_ID"
        else: ID_col = "MICS_ID"
        #print(ID_col)

        if df.empty: # check if the dataframe is empty
            print(f"\t[clean_pths] WARNING: Empty dataframe for {d['study']} {d['grp']}")
            continue
        else:
            df_clean = ses_clean(df, ID_col, method=method, silent=True)
            #dl[i]['map_pths'] = df_clean

        if df_clean.empty:  # check if the cleaned dataframe is empty
            print(f"\t[clean_pths] WARNING: Cleaned dataframe is empty for {d['study']} {d['grp']}")
            continue

        dl_out.append({
            'study': d['study'],
            'grp': d['grp'],
            'grp_labels': d['grp_labels'],
            'label': d['label'],
            'feature': d['feature'],
            'map_pths': df_clean
        })

    return dl_out


def ses_clean(df, ID_col, method="newest", silent=True):
    """
    Choose the session to use for each subject.
        If subject has multiple sessions with map path should only be using one of these sessions.

    inputs:
        df: pd.dataframe with columns for subject ID, session, date and map_paths
            Assumes map path is missing if either : map_pth
        ID_col: column name for subject ID in the dataframe
        method: method to use for choosing session. 
            "newest": use most recent session
            "oldest": use oldest session in the list
            {number}: session code to use (e.g. '01' or 'a1' etc)
    """
    
    import pandas as pd
    import datetime

    # check if the dataframe is empty
    if df.empty:
        print(f"[ses_clean] WARNING: Empty dataframe. Skipping.")
        return

    if not silent: print(f"[ses_clean] Choosing session according to method: {method}")
    
    
    df = df.copy()  # Avoid modifying the original dataframe

    # remove rows whose path col is empty or starts with "ERROR:"
    path_cols = [col for col in df.columns if col.startswith('pth_') or col.startswith('surf_') or col.startswith('map_')]
    df_clean = df.dropna(subset=path_cols, how='all')  # Keep rows where at least one path column is not NaN
    df_clean = df_clean[~df_clean[path_cols].apply(lambda x: x.str.startswith("ERROR:")).any(axis=1)]  # Remove rows where any path column starts with "ERROR:"
    if df_clean.empty:
        if not silent:
            print(f"[ses_clean] WARNING: All rows removed due to empty or ERROR paths. Returning empty dataframe.")
        return pd.DataFrame()
    
    # Find repeated IDs (i.e., subjects with multiple sessions)
    repeated_ids = df_clean[df_clean.duplicated(subset=ID_col, keep=False)][ID_col].unique()
    
    if not silent:
        if len(repeated_ids) > 0:
            print(f"\tIDs with multiple sessions: {repeated_ids}")
        else:
            print(f"\tNo repeated IDs found")

    rows_to_remove = []
    
    # Convert 'Date' column to datetime for comparison
    df_clean['Date_dt'] = pd.to_datetime(df_clean['Date'], format='%d.%m.%Y', errors='coerce')
    today = pd.to_datetime('today').normalize()

    if len(repeated_ids) > 0:
        if method == "newest":
            for id in repeated_ids:
                sub_df = df_clean[df_clean[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_dt'].idxmax()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        elif method == "oldest":
            for id in repeated_ids:
                sub_df = df_clean[df_clean[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_dt'].idxmin()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        else:
            # Assume method is a session code (e.g., '01', 'a1', etc)
            for id in repeated_ids:
                sub_df = df_clean[df_clean[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_remove = sub_df[sub_df['SES'] != method].index
                    rows_to_remove.extend(idx_to_remove)

    # Remove the rows marked for removal
    df_clean = df_clean.drop(rows_to_remove)
    #if not silent: print(df_clean[[ID_col, 'SES']].sort_values(by=ID_col))

    # if num rows =/= to num unique IDs then write warning
    if df_clean.shape[0] != df_clean[ID_col].nunique():
        print(f"[ses_clean] WARNING: Number of rows ({df_clean.shape[0]}) not equal to num unique IDs ({df_clean[ID_col].nunique()})")
        print(f"\tMultiple sessions for IDs: {df_clean[df_clean.duplicated(subset=ID_col, keep=False)][ID_col].unique()}")

    if not silent: 
        print(f"\t{df.shape[0] - df_clean.shape[0]} rows removed, Change in unique IDs: {df_clean[ID_col].nunique() - df[ID_col].nunique()}")
        print(f"\t{df_clean.shape[0]} rows remaining")

    return df_clean

def get_finalSES(dl, demo, save_pth=None, long=False, silent=True): 
    """
    From a list of dictionary items, create a DF with sessions retained for each participant and each feature 

    input:
        dl: List of dictionary items with cleaned dataframes
        demo: dictionary with demographics file information.
        save_pth: path to save the dataframe to. If None, do not save.
        long: if True, return a long format dataframe with one row per subject and session feature. If False, return wide format.

    output:
        df: pd.dataframe with columns for subject ID, session_feature, grp, study and map_paths
            Assumes map path is missing if either : map_pth
    """
    import datetime
    import pandas as pd
    import numpy as np

    demo_df = pd.read_csv(demo['pth'], dtype=str)
    out = pd.DataFrame()  # Will collect all unique IDs and their session columns

    for i, d in enumerate(dl):
        feature = d['feature']
        label = d['label']

        df = d['map_pths']

        id_col = [col for col in df.columns if 'ID' in col.upper()][0]  
        ses_col = 'SES'
        
        if not silent: print(f"[get_finalSES] {d['study']} {d['grp']}: {feature}, {label} ({df.shape[0]} rows)")

        # Use correct study prefix and correct ID column for merge
        if d['study'] == "PNI": 
            study_prefix = "7T"
            merge_id_col = demo['ID_7T']
        elif d['study'] == "MICs":
            study_prefix = "3T"
            merge_id_col = demo['ID_3T']
        else: 
            study_prefix = "Unknown"
            merge_id_col = None

        if label == "thickness": lbl_ft = f"{label}"
        else: lbl_ft = f"{label}-{feature}"

        new_col = f'{study_prefix}-ses_{lbl_ft}'

        # Mark SES as NA if all path columns are ERROR or missing
        path_cols = [col for col in df.columns if col.startswith('pth_') or col.startswith('surf_') or col.startswith('map_')]
        def ses_na_row(row):
            if not path_cols:
                return row[ses_col]
            # If all path columns are missing or start with ERROR
            if all((not isinstance(row[c], str)) or row[c].startswith("ERROR") or row[c] == "" for c in path_cols):
                return "NA"
            return row[ses_col]
        df_tmp = df[[id_col, ses_col] + path_cols].copy()
        df_tmp[new_col] = df_tmp.apply(ses_na_row, axis=1)
        df_tmp = df_tmp.rename(columns={id_col: "ID"})
        df_tmp = df_tmp[["ID", new_col]]

        # If column already exists, add to it
        if new_col in out.columns:
            # Merge on ID, but keep both values for comparison
            merged = pd.merge(out[['ID', new_col]], df_tmp, on="ID", how="outer", suffixes=('_old', '_new'))
            
            def resolve(row): # For each ID, resolve conflicts
                vals = set([row[f"{new_col}_old"], row[f"{new_col}_new"]])
                vals = {v for v in vals if pd.notnull(v)}
                if len(vals) == 1:
                    return vals.pop()
                elif len(vals) > 1:
                    # Print warning and keep the latest value (assuming SES is string, keep max)
                    if not silent:
                        print(f"[get_finalSES] WARNING: Multiple values for {row['ID']} in {new_col}: {vals}. Keeping latest.")
                    # If SES is numeric string, sort as int, else as string
                    try:
                        return sorted(vals, key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else str(x))[-1]
                    except Exception:
                        return sorted(vals)[-1]
                else:
                    return None
                
            merged[new_col] = merged.apply(resolve, axis=1)
            
            # Update out with resolved column
            out = pd.merge(out, merged[['ID', new_col]], on="ID", how="outer", suffixes=('', '_resolved'))
            out[new_col] = out[new_col + '_resolved'].combine_first(out[new_col])
            out = out.drop(columns=[new_col + '_resolved'])
        else:
            if out.empty:
                out = df_tmp
            else:
                out = pd.merge(out, df_tmp, on="ID", how='outer')

    if not long:
        # under construction
        print("[get_finalSES] Wide format not yet implemented. Returning long format instead.")
        long = True

    if save_pth is not None:
        date = datetime.datetime.now().strftime("%d%b%Y-%H%M")
        if long: save = f"{save_pth}/sesXfeat_long_{date}.csv"
        else: save = f"{save_pth}/sesXfeat_{date}.csv"
        out.to_csv(save, index=False)
        print(f"[get_finalSES] Saved dataframe to {save}")

    return out

# support print functions
def print_dict(dict, df_print=False, idx=None):
    """
    Print the contents of a dictionary with DataFrames in a readable format.
    Input:
        List of dict items.
        df_print: if True, prints DataFrame contents; if False, only print the shape of the DF keys
        df_idx: if provided, only print the items at these indices in the dict list.
    Output:
        Prints the keys and values of each dictionary item.
    """
    import pandas as pd
    
    if idx is not None:
        print(f"\n Printing the following {len(idx)} indices: {idx}")
        for i in idx:
            d = dict[i]
            print(f"\n[{i}]")
            print(f"\tKeys: {list(d.keys())}")
            for k, v in d.items():
                if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                    print(f"\t{k}: <DataFrame shape={v.shape}>")
                    if df_print: print(f"\t{k}: {v}")
                else:
                    print(f"\t{k}: {v}")
        return
    else:
        print(f"\n Dict list length ({len(dict)} items)")
        for i, item in enumerate(dict):
            d = item
            print(f"\n[{i}]")
            print(f"\tKeys: {list(d.keys())}")

            for k, v in d.items():
                if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                    print(f"\t{k}: <DataFrame shape={v.shape}>")
                    if df_print == True: print(f"\t{k}: {v}")
                else:
                    print(f"\t{k}: {v}")

def print_grpDF(dict, grp, study, hipp=False, df="pth"):
    # hipp option: only print items where 'hippocampal'==True
    for item in dict:
        if item['study'] == study and item['grp'] == grp:
            if hipp and not item.get('hippocampal', False):
                continue
            if df == "pth":
                df_keys = ['map_pths']
            elif df == "maps":
                # identify keys with prefix 'map'
                df_keys = [k for k in item.keys() if k.startswith('map_') or k.startswith('pth_')]
            else:
                df_keys = ['map_pths']

            print(f"{item['study']}-{item['grp']} ({item['grp_labels']})")
            # Set display options to not truncate cell values
            with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
                for k in df_keys:
                    print(item[k])
            break

def ctrl_index(maps, comp_idx, ctrl_code='ctrl'):
    """
    Find the index of the control group in the maps list for a given study.
    
    inputs:
        maps: list of dictionary items with keys 'study', 'grp', 'label', 'feature'
        study: name of the study to search in
        comp_idx: index of the item to find the control group for
        ctrl_code: code for control group (default is 'ctrl')

    outputs:
        index of the control group item in the maps list, or None if not found
    """
    import pandas as pd
    
    item = maps[comp_idx]
    study = item['study']
    label = item['label']
    feature = item.get('feature', None)
    surface = item.get('surface', None)  # Default to 'fsLR-5k' if not specified
    hipp = item.get('hippocampal', False)  # Check if hippocampal is True or False

    for i, other in enumerate(maps):
        if (
            other['study'] == study and
            other['grp'] == ctrl_code and
            other['label'] == label and
            other.get('feature', None) == feature and
            other.get('surface', None) == surface and
            other.get('hippocampal', False) == hipp
        ):
            return i
    return None  # Not found

def find_paired_TLE_index(dl, idx, mtch=['study','label']):
    """
    Given the index of a TLE_L or TLE_R item in dict, find the index of the paired item
    (same study, label, feature, but opposite group).

    Input:
        dl: list of dictionary items with keys 'study', 'grp', 'label', 'feature'
        idx: index of the item to find the pair for
        mtch: list of keys to match on. NOTE: the key 'grp' is always matched on.
    """
    item = dl[idx]
    grp = item['grp']
    if grp not in ['TLE_L', 'TLE_R']:
        raise ValueError("Item at idx is not TLE_L or TLE_R")
    paired_grp = 'TLE_R' if grp == 'TLE_L' else 'TLE_L'
    
    for j, other in enumerate(dl):
        if j != idx and other['grp'] == paired_grp:
            match = True
            for key in mtch:
                if other.get(key) != item.get(key):
                    match = False
                    break
            if match:
                return j
    return None  # Not found

def get_pair(dl, idx, mtch=['study', 'grp', 'label']):
    """
    Get corresponding idx for item in dictionary list.

    Input:
        dl: list of dictionary items with keys found in mtch
        idx: index of the item to find the pair for
        mtch: list of keys to match on.
    """

    item = dl[idx]
    for j, other in enumerate(dl):
        if j != idx:
            match = True
            for key in mtch:
                if other.get(key) != item.get(key):
                    match = False
                    break
            if match:
                return j
    return None  # Not found

def ipsi_contra(df, hemi_ipsi='L', rename_cols = True):
    """
    Given a dictionary item, with vertex-wise dataframes, relabel columns to ipsi and contra. Put ipsi before contra in the output and rename the column names accordingly.

    Input: 
        df: vertex-wise dataframe with vertex in columns, pts in rows. 
        hemi_ipsi: what side is ipsi ('L' or 'R'). <default is 'L'>.

    Returns:
        df: vertex-wise dataframe with columns renamed to ipsi and contra, and ipsi columns placed before contra columns.
        hemi_ipsi: string indicating which hemisphere is ipsi ('L' or 'R').
    """
    df = df.copy()  # Avoid modifying the original dataframe
    if hemi_ipsi == "L":
        if rename_cols:
            # should rename all columns with '_L' to '_ipsi' and '_R' to '_contra'
            df.columns = [col.replace('_L', '_ipsi').replace('_R', '_contra') for col in df.columns]
    
    elif hemi_ipsi == "R":
        if rename_cols:
            # should rename all columns with '_L' to '_contra' and '_R' to '_ipsi'
            df.columns = [col.replace('_L', '_contra').replace('_R', '_ipsi') for col in df.columns]
    
    else:
        raise ValueError("Invalid hemi_ipsi value. Choose 'L' or 'R'.")
    
    # Identify columns
    ipsi_cols = [col for col in df.columns if '_ipsi' in col]
    contra_cols = [col for col in df.columns if '_contra' in col]
    other_cols = [col for col in df.columns if col not in ipsi_cols + contra_cols]
    # Reorder: other columns first, then ipsi, then contra
    df_ic = df[other_cols + ipsi_cols + contra_cols]

    return df_ic, hemi_ipsi

def get_z(x, col_ctrl):
    """
    Calculate z-scores for a given value in a DataFrame, using the control group as the reference.
    
    inputs:
        x: value for specific subject and same column as col_ctrl
        col_ctrl: column name for control group in the DataFrame

    outputs:
        z: z-scores for the specified column
    """
    import pandas as pd
    
    ctrl_mean = col_ctrl.mean()
    ctrl_std = col_ctrl.std()
    
    return (x - ctrl_mean) / ctrl_std

######################### VISUALIZATION FUNCTIONS #########################

def visMean(dl, df_name='df_z_mean', indices=None, ipsiTo="L", title=None, save_name=None, save_path=None):
    """
    Create brain figures from a list of dictionary items with vertex-wise dataframes.
    Input:
        dl: list of dictionary items with keys 'study', 'grp', 'label', 'feature', 'df_z_mean'
        df_name: name of the dataframe key to use for visualization (default is 'df_z_mean')
        indices: list of indices to visualize. If None, visualize all items in the list.
        ipsiTo: hemisphere to use for ipsilateral visualization ('L' or 'R').
    """
    from IPython.display import display

    for i, item in enumerate(dl):
        print(f"[visMean] [{i}] ({item.get('study','')} {item.get('grp','')} {item.get('label','')})")
        
        if indices is not None and i not in indices:
            continue
        if df_name not in item:
            print(f"[visMean] WARNING: {df_name} not found in item {i}. Skipping.")
            continue
        df = item[df_name]
        #print(f"\tdf of interest: {df.shape}")

        # remove SES or ID columns if they exist
        df = df.drop(columns=[col for col in df.columns if col in ['SES', 'ID', 'MICS_ID', 'PNI_ID']], errors='ignore')
        #print(f"\tdf after removing ID/SES: {df.shape}")
        
        # surface from size of df
        if item['grp'].endswith('_ic'):
            if ipsiTo == "L":
                lh_cols = [col for col in df.columns if col.endswith('_ipsi')]
                rh_cols = [col for col in df.columns if col.endswith('_contra')]
            else:
                # if ipsiTo is not L, then assume it is R
                lh_cols = [col for col in df.columns if col.endswith('_contra')]
                rh_cols = [col for col in df.columns if col.endswith('_ipsi')]
        else:
            #print(df.columns)
            lh_cols = [col for col in df.columns if col.endswith('_L')]
            rh_cols = [col for col in df.columns if col.endswith('_R')]
        #print(f"\tNumber of relevant columns: L={len(lh_cols)}, R={len(rh_cols)}")
        assert len(lh_cols) == len(rh_cols), f"[visMean] WARNING: Left and right hemisphere columns do not match in length for item {i}. Skipping."
        if len(lh_cols) == 32492:
            surface = 'fsLR-32k'
        else: 
            surface = 'fsLR-5k'

        lh = df[lh_cols]
        rh = df[rh_cols]
        #print(f"\tL: {lh.shape}, R: {rh.shape}")
        fig = showBrains(lh, rh, surface, ipsiTo=ipsiTo, save_name=save_name, save_pth=save_path, title=title, min=-2, max=2, inflated=True)
        display(fig)

def showBrains(lh, rh, surface='fsLR-5k', ipsiTo=None, title=None, min=-2.5, max=2.5, inflated=True, save_name=None, save_pth=None, cmap="seismic"):
    """
    Returns brain figures

    inputs:
        lh: column with values for left hemisphere surface (each row represents a vertex and is properly indexed) 
        rh: column right hemisphere surface (each row represents a vertex and is properly indexed) 
        surface: surface type (default is 'fsLR-5k')
        inflated: whether to use inflated surfaces (default is False)
        title: title for the plot (default is None)
        save_name: name to save the figure (default is None)
        save_pth: path to save the figure (default is None)

    """    
    
    import os
    import glob
    import numpy as np
    import nibabel as nib
    import seaborn as sns
    from brainspace.plotting import plot_hemispheres
    from brainspace.mesh.mesh_io import read_surface
    from brainspace.datasets import load_conte69
    import datetime

    micapipe=os.popen("echo $MICAPIPE").read()[:-1]
    
    # set wd to save_pth
    if save_pth is not None:
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        os.chdir(save_pth)

    if surface == 'fsLR-5k':
        if inflated == True:
            # Load fsLR 5k inflated
            surf_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
            surf_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
        else:
            # Load Load fsLR 5k
            surf_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.surf.gii', itype='gii')
            surf_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.surf.gii', itype='gii')
    elif surface == 'fsLR-32k':
        if inflated == True:
            # Load fsLR 32k inflated
            surf_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
            surf_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
        else:
            # Load fsLR 32k
            surf_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.surf.gii', itype='gii')
            surf_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.surf.gii', itype='gii')
    else:
        raise ValueError(f"Surface {surface} not recognized. Use 'fsLR-5k' or 'fsLR-32k'.")

    #print(f"L: {lh.shape}, R: {rh.shape}")
    data = np.hstack(np.concatenate([lh, rh], axis=0))
    #print(data.shape)

    lbl_text = {'top': title}
    
    if ipsiTo is not None and ipsiTo == "L":
        lbl_text = {
            'left': 'ipsi',
            'right': 'contra'
        }    
    elif ipsiTo is not None and ipsiTo == "R":
        lbl_text = {
            'left': 'contra',
            'right': 'ipsi'
        }
    else:
        lbl_text = {
            'left': 'L',
            'right': 'R'
        }

    # Ensure all values are strings (robust against accidental lists/arrays)
    lbl_text = {k: str(v) for k, v in lbl_text.items()}

    date = datetime.datetime.now().strftime("%d%b%Y-%H%M")
    filename = f"{save_name}_{surface}_{date}.png" if save_name and save_pth else None
    
    # Plot the surface with a title
    if filename : 
        print(f"[showBrains] Plot saved to {filename}")
        return plot_hemispheres(
                surf_lh, surf_rh, array_name=data, 
                size=(900, 250), color_bar='bottom', zoom=1.25, embed_nb=True, interactive=False, share='both',
                nan_color=(0, 0, 0, 1), color_range=(min,max), cmap=cmap, transparent_bg=False, 
                screenshot=True, filename=filename,
                #, label_text = lbl_text
            )
    else:
        return plot_hemispheres(
                surf_lh, surf_rh, array_name=data, 
                size=(900, 250), color_bar='bottom', zoom=1.25, embed_nb=True, interactive=False, share='both',
                nan_color=(0, 0, 0, 1), color_range=(min,max), cmap=cmap, transparent_bg=False, 
                #, label_text = lbl_text
            )