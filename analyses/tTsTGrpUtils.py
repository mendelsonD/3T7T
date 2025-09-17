################### DATA PREPERATION ####################################
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

def stdizeNA(df, missing_patterns=None, verbose=True):
    """
    Replace various missing value patterns in the DataFrame with np.nan.

    Parameters:
        df: pd.DataFrame
        missing_patterns: list or None
            List of patterns to treat as missing. If None, uses a default set.
        verbose: bool
            If True, prints detailed information about changes made.

    Returns:
        df: pd.DataFrame with standardized missing values
    """
    import numpy as np

    # Default patterns for missing values
    default_patterns = [
        '', ' ', 'nan', 'NaN', 'NAN', 'null', 'NULL', '?', 'NA', 'na', 'n/a', 'N/A', '.', '-', '--', 'missing', 'MISSING'
    ]
    
    if missing_patterns is not None:
        patterns = set(default_patterns) | set(missing_patterns)
    else:
        patterns = set(default_patterns)

    # Find all values that match the patterns before replacement
    mask = df.isin(patterns)
    changed = mask.stack()[lambda x: x].index.tolist()

    if verbose:
        if changed:
            print(f"[stdizeNA] Replacing values matching following patterns with standard NA. Patterns: {patterns}")
            for idx, col in changed:
                old_val = df.at[idx, col]
                print(f"\t[{df.at[idx, 'MICS_ID'] if 'MICS_ID' in df.columns else 'NA'}={df.at[idx, 'PNI_ID'] if 'PNI_ID' in df.columns else 'NA'} study: {df.at[idx, 'study'] if 'study' in df.columns else 'NA'}-ses{df.at[idx, 'SES'] if 'SES' in df.columns else 'NA'}] {{{col[:10]}}}: {old_val} --> NaN")

    # Replace all matching patterns with np.nan
    df = df.replace(list(patterns), np.nan)

    # Only count the number of values that were actually changed (not total missing)
    num_changed = len(changed)
    print(f"[stdizeNA] Standardized {num_changed} missing values")
    return df


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

def get_surf_pth(root, sub, ses, lbl, space="nativepro", surf="fsLR-32k", verbose=False):
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
        lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-{space}_surf-{surf}_label-{lbl}.surf.gii"
        rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-{space}_surf-{surf}_label-{lbl}.surf.gii"
    
    elif "hippunfold" in root:
        if verbose: print("Hipp detected")
        # Note: for MICA studies, hippocampal maps are in 'T1w' space which is equivalent to nativepro space 
        if lbl == "thickness" or lbl == "hipp_thickness":
            lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-{space}_den-{surf}_label-hipp_thickness.shape.gii"
            rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-{space}_den-{surf}_label-hipp_thickness.shape.gii"
        else:
            lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-{space}_den-{surf}_label-hipp_{lbl}.surf.gii"
            rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-{space}_den-{surf}_label-hipp_{lbl}.surf.gii"
    else:
        raise ValueError("[get_surf_path] Unknown root directory. Choose from 'micapipe' or 'hippunfold'.")

    return lh, rh 

def get_mapVol_pth(root, sub, ses, feature, space="nativepro"):
    """
    Get path to map volume.
    
    input:
        root: root directory to micapipe output
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        feature: type of map to retrieve (e.g., "T1map", "FLAIR", "ADC", "FA")
            File naming pattern:
                T1map: map-T1
                FLAIR: map-flair
                ADC: DTI_map-ADC
                FA: DTI_map-FA
        space: space of the map (default is "nativepro")

    output:
        Path to the map file in nativepro space.
    """
    feature = feature.upper()
    if feature == "T1MAP": mtrc = "map-T1map"
    elif feature == "FLAIR": mtrc = "map-flair"
    elif feature == "ADC": mtrc = "model-DTI_map-ADC"
    elif feature == "FA": mtrc = "model-DTI_map-FA"
    else:
        raise ValueError(f"[get_vol_path] Invalid metric: {feature}. Choose from 'T1map', 'FLAIR', 'ADC', or 'FA'.")

    return f"{root}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_space-{space}_{mtrc}.nii.gz"
    
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
        
        # space usually: "T1w" equivalent to nativepro
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

def map(vol, surf, out, method="trilinear", verbose=True):
    """
    From volume and surface, generate an unsmoothed feature map.

    Input:
        vol: path to unsmoothed feature map file (.func.gii)
        surf: path to the surface file (.surf.gii)
        out: output path and name
        smth: smoothing kernel size in mm (default is 10mm)
        method <optional>: see options in help file for wb_command -volume-to-surface-mapping
            Default is "trilinear"

    Return:
        out: path to the feature map created by projecting the volume data onto the surface.
    """

    import subprocess as sp
    
    cmd = [
        "wb_command",
        "-volume-to-surface-mapping", 
        vol,
        surf,
        out,
        f"-{method}"
    ]

    sp.run(cmd, check=True)

    if not chk_pth(out):
        print(f"\t[map] WARNING: Unsmoothed map not properly saved. Expected file: {out}")
        return None
    else:
        if verbose:
            print(f"\t[map] Unsmoothed map saved to: {out}")
        return out

def smooth_map(surf, map, out_name, kernel=10, verbose=True):
    """
    Apply smoothing to a feature map.

    Input:
        surf: path to the surface file (.surf.gii)
        map: path to unsmoothed feature map file (.func.gii)
        out_name: output file name
        smth_kernel: smoothing kernel size in mm (default is 10mm)
    """

    import subprocess as sp
    
    cmd = [
        "wb_command",
        "-metric-smoothing", 
        surf,
        map, str(kernel), 
        out_name
        ]
    
    sp.run(cmd, check=True)

    if not chk_pth(out_name):
        print(f"\t[smooth_map] WARNING: Smoothed map not properly saved. Expected file: {out_name}")
        return None
    else:
        if verbose:
            print(f"\t[smooth_map] Smoothed map saved to: {out_name}")
        return out_name

def create_dir(dir_path, verbose=False):
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
        if verbose:
            print(f"\t[create_dir] Directory at path `{dir_path}` exists. Skipping creation.")    
    return

def get_rawvol_pth(root, sub, ses, ft):
    """
    Get string name for raw volume path variable.
    
    input:
        dir: root directory to raw data
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        ft: feature name (e.g., "T1map", "FLAIR", "ADC", "FA")
    
    output:
        path to the raw volume file

    """

    ft = ft.upper()
    if ft == "T1MAP": 
        dir_sub = "/anat"
        ptrn = "T1map"
    elif ft == "FLAIR":
        dir_sub = "/anat"
        ptrn = "FLAIR"
    elif ft == "T1W":
        dir_sub = "/anat"
        ptrn = "T1w"
    elif ft == "ADC" or ft == "FA":
        dir_sub = "/dwi"
        ptrn = "dwi"
    else:
        raise ValueError(f"[get_rawvol_pth] Unrecognized feature {ft}. Defined features 'T1w', 'T1map', 'FLAIR', 'ADC', 'FA'.")

    pth = root + f"/sub-{sub}/ses-{ses}" + dir_sub + f"/sub-{sub}_ses-{ses}*{ptrn}*.nii.gz"
    return pth

def matchPathPtrn(pth):
    """
    Find all files matching a given pattern.
    input:
        pth: path pattern to search for (can include wildcards)
    """
    import glob
    matches = glob.glob(pth)
    return matches

def checkRawPth(root, sub, ses, ft, return_names=False):
    """
    Given a root directory, subject ID, session ID, and feature name, check if the raw volume path exists.
    """
    if ft.upper() == "THICKNESS":
        ft = "T1w" # thickness is derived from T1w image
        
    pth = get_rawvol_pth(root, sub, ses, ft)
    matches = matchPathPtrn(pth)
    if matches:
        if return_names: return matches
        else: return True
    else:
        if return_names: return None
        else: return False

def idToMap(df_demo, studies, dict_demo, specs, verbose=False):
    """
    From demographic info, add path to unsmoothed, smoothed maps. If smoothed map does not exist, compute. 
    Do this for all parameter combinations (surface, label, feature, smoothing kernel) provided in a dictionary item.

    Parameters
    ----------
    df_demo : pd.DataFrame
        DataFrame containing demographic information for each subject/session.
    studies : list of dict
        List of dictionaries, each containing study-specific directory and naming information.
    dict_demo : dict
        Dictionary with demographic column names and study info.
    specs : dict
        Dictionary specifying which maps to process (features, surfaces, smoothing, labels, etc).
    verbose : bool, optional
        If True, print detailed progress and warnings.

    Returns
    -------
    df_demo : pd.DataFrame
        Updated DataFrame with new columns for each map (smoothed/unsmoothed) found or computed.
    log_contents : str
        String containing all print/log output during function execution.

    Notes
    -----
    - For each row in df_demo, determines the study and subject/session.
    - For each combination of feature, surface, smoothing, and label in specs, attempts to find or compute the corresponding map.
    - If map does not exist, attempts to compute it using wb_command or other mapping functions.
    - Adds the resulting map paths to new columns in df_demo.
    - Returns both the updated DataFrame and a string log of all output.
    """
    import os
    import re
    import time
    import sys
    import io

    class TeeLogger(io.StringIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._stdout = sys.stdout

        def write(self, s):
            self._stdout.write(s)
            super().write(s)

        def flush(self):
            self._stdout.flush()
            super().flush()

    log_stream = TeeLogger()
    old_stdout = sys.stdout
    sys.stdout = log_stream

    try:
        def ctx_maps(out_dir, study, df, idx, sub, ses, surf, ft, smth, lbl, verbose=False):
            """
            Get or compute cortical smoothed maps for a given subject and session, surface, label, feature, and smoothing kernel size.
            """
            import os

            print(f"\t{ft}, {lbl}, {surf}, smth-{smth}mm")
            root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
            study_code = study['study']

            skip_L, skip_R = False, False

            # Ø. Declare output names and final file of interest paths
            if ft == "thickness":
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{ft}"
            else:
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{lbl}_{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{lbl}_{ft}"
                                            
            pth_map_unsmth_L = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_L_filename}.func.gii"
            pth_map_unsmth_R = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_R_filename}.func.gii"
            
            out_pth_L  = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_ctx_{out_pth_L_filename}_smth-{smth}mm.func.gii",
            )

            out_pth_R = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_ctx_{out_pth_R_filename}_smth-{smth}mm.func.gii",
            )
            
            col_base_L = os.path.basename(out_pth_L).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            col_base_R = os.path.basename(out_pth_R).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            
            # Replace smoothing with appropriate suffix
            col_unsmth_L = col_base_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = col_base_R.replace(f"_smth-{smth}mm", "_unsmth")

            if chk_pth(out_pth_L) and chk_pth(out_pth_R): # If smoothed map already exists, do not recompute. Simply add path to df.
                if verbose:
                    print(f"\t\tSmoothed maps exists, adding to df: {out_pth_L}\t{out_pth_R}\n")
                
                # Add paths to unsmoothed maps
                df.loc[idx, col_unsmth_L] = pth_map_unsmth_L
                df.loc[idx, col_unsmth_R] = pth_map_unsmth_R
                
                # Add paths to smoothed maps
                df.loc[idx, col_base_L] = out_pth_L
                df.loc[idx, col_base_R] = out_pth_R
                return df
            elif chk_pth(out_pth_L):
                if verbose:
                    print(f"\t\tSmoothed L map exists, adding path to df:\t\t{out_pth_L}")
                
                df.loc[idx, col_unsmth_L] = pth_map_unsmth_L
                df.loc[idx, col_base_L] = out_pth_L
                skip_L = True
            elif chk_pth(out_pth_R):
                if verbose:
                    print(f"\t\tSmoothed R map exists, adding path to df:\t\t{out_pth_R}")
                
                df.loc[idx, col_unsmth_R] = pth_map_unsmth_R
                df.loc[idx, col_base_R] = out_pth_R
                skip_R = True

            # A. Search for unsmoothed map
            if not chk_pth(pth_map_unsmth_L) and not chk_pth(pth_map_unsmth_R): # both unsmoothed maps mising
                
                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    pth_map_unsmth_L, pth_map_unsmth_R = "NA: NO RAWDATA", "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Unsmoothed maps missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                   
                else: # Must be micapipe problem if raw data exists
                    
                    dir_surf = " ( " + os.path.commonpath([pth_map_unsmth_L, pth_map_unsmth_R]) + " ) " if pth_map_unsmth_L and pth_map_unsmth_R else "" # Find the common directory of both pth_map_unsmth_L and pth_map_unsmth_R
                    pth_map_unsmth_L, pth_map_unsmth_R = "NA: MISSING MP PROCESSING (unsmoothed map)", "NA: MISSING MP PROCESSING (unsmoothed map)"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Unsmoothed maps MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")
                    
                df.loc[idx, col_base_L] = pth_map_unsmth_L
                df.loc[idx, col_base_R] = pth_map_unsmth_R
                return df
            
            elif not chk_pth(pth_map_unsmth_L): # missing L hemi only
                
                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    pth_map_unsmth_R = "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Hemi-L unsmoothed map due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                
                else: # Must be micapipe problem if raw data exists

                    dir_surf = os.path.basename(pth_map_unsmth_L)
                    pth_map_unsmth_L = "NA: MISSING MP PROCESSING (unsmoothed map)"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses}  ({ft}, {lbl}, {surf}): Hemi-L unsmoothed map MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")

                skip_L = True
                df.loc[idx, col_base_L] = pth_map_unsmth_L

            elif not chk_pth(pth_map_unsmth_R): # missing R hemi only

                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    pth_map_unsmth_R = "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Hemi-R unsmoothed map due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                    
                else: # Must be micapipe problem if raw data exists
                    dir_surf = os.path.basename(pth_map_unsmth_R)
                    pth_map_unsmth_L = "NA: MISSING MP PROCESSING (unsmoothed map)"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses}  ({ft}, {lbl}, {surf}): Hemi-R unsmoothed map MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")

                skip_R = True
                df.loc[idx, col_base_R] = pth_map_unsmth_R
            
            else: # unsmoothed maps exist for both
                if verbose:
                    print(f"\t\tUnsmoothed maps:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
            
            # add path to unsmoothed maps to df
            if not skip_L:
                df.loc[idx, col_unsmth_L] = pth_map_unsmth_L
                print(f"\t\tAdded L unsmoothed path: {pth_map_unsmth_L}")
            if not skip_R:
                df.loc[idx, col_unsmth_R] = pth_map_unsmth_R
                print(f"\t\tAdded R unsmoothed path: {pth_map_unsmth_R}")

            # B. Smooth map and save to project directory
            surf_L, surf_R = get_surf_pth( # Get surface .func files
                root=root_mp,
                sub=sub,
                ses=ses,
                surf=surf,
                lbl=lbl
            )
            
            if not chk_pth(surf_L) and not chk_pth(surf_R) and not skip_L and not skip_R: # check that surfaces exist
                dir_surf = os.path.commonpath([surf_L, surf_R]) # Find the common directory of both surf_L and surf_R
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface not found. Skipping. Check micapipe processing ( {dir_surf} ). Missing: {surf_L}\t{surf_R}\n")
                surf_L, surf_R = "NA: MISSING MP PROCESSING (surface)", "MISSING MP PROCESSING (surface)"
                return df
            elif not chk_pth(surf_L):
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface missing for hemi-L. Check micapipe processing ( {os.path.dirname(surf_L)} ). Skipping smoothing for this hemi. Expected: {surf_L}")
                surf_L = "NA: MISSING MP PROCESSING (surface)"
                
                skip_L = True
                df.loc[idx, col_base_L] = surf_L
            elif not chk_pth(surf_R):
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface missing for hemi-R. Check micapipe processing ( {os.path.dirname(surf_R)} ). Skipping smoothing for this hemi. Expected: {surf_R}")
                surf_R = "NA: MISSING MP PROCESSING (surface)"
                
                skip_R = True
                df.loc[idx, col_base_R] = surf_R
            else:
                if verbose:
                    print(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")

            # ii. Smooth
            if not skip_L:
                pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, out_pth_L, kernel=smth, verbose=False)
            else: pth_map_smth_L = None
            
            if not skip_R:
                pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_R, out_pth_R, kernel=smth, verbose=False)
            else:
                pth_map_smth_R = None

            if pth_map_smth_L is None or pth_map_smth_R is None and not skip_L and not skip_R:
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Smoothing failed.\n")
                pth_map_smth_L, pth_map_smth_R = f"NA: SMOOTHING FAILED. Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_R}, kernel: {smth}", f"NA: SMOOTHING FAILED.  Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_R}, kernel: {smth}"
            else:
                # Add paths to DataFrame
                if verbose:
                    print(f"\t\tAdding to df: {pth_map_smth_L}\t{pth_map_smth_R}\n")
                    
                df.loc[idx, col_base_L] = pth_map_smth_L
                df.loc[idx, col_base_R] = pth_map_smth_R

            return df

        def hipp_maps(out_dir, study, df, idx, sub, ses, surf, ft, smth, lbl, verbose=False):
            """
            Get or compute hippocampal smoothed maps for a given subject and session, surface, label, feature, and smoothing kernel size.
            """
            import os 
            
            print(f"\t{ft}, {lbl}, {surf}, smth-{smth}mm")
            root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
            root_hu = f"{study['dir_root']}{study['dir_deriv']}{study['dir_hu']}"
            study_code = study['study']

            skip_L, skip_R = False, False

            # Ø. Declare output names and final file of interest paths                 
            if ft == "thickness":
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{ft}"
            else:
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{lbl}_{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{lbl}_{ft}"

            out_pth_L  = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-{smth}mm.func.gii",
            )

            out_pth_R = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-{smth}mm.func.gii",
            )
            
            col_base_L = os.path.basename(out_pth_L).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            col_base_R = os.path.basename(out_pth_R).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            
            col_unsmth_L = col_base_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = col_base_R.replace(f"_smth-{smth}mm", "_unsmth")

            if chk_pth(out_pth_L) and chk_pth(out_pth_R): # If smoothed map already exists, do not recompute. Simply add path to df.
                if verbose:
                    print(f"\t\tSmoothed maps exist, adding paths to df:\t{out_pth_L}\t{out_pth_R}\n")
                
                if ft == "thickness":
                    pth_map_unsmth_L, pth_map_unsmth_R = get_surf_pth(root=root_hu, sub=sub, ses=ses, lbl="thickness", surf=surf, space="T1w")
                else:
                    pth_map_unsmth_L = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-NA.func.gii"
                    pth_map_unsmth_R = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-NA.func.gii"
                
                # add unsmth map path
                df.loc[idx, col_unsmth_L] = pth_map_unsmth_L
                df.loc[idx, col_unsmth_R] = pth_map_unsmth_R
                
                # add smth map path
                df.loc[idx, col_base_L] = out_pth_L
                df.loc[idx, col_base_R] = out_pth_R
                return df
            
            elif chk_pth(out_pth_L):
                if verbose:
                    print(f"\t\tSmoothed L map exists, adding path to df:\t\t{out_pth_L}")
                df.loc[idx, col_base_L] = out_pth_L
                skip_L = True

            elif chk_pth(out_pth_R):
                if verbose:
                    print(f"\t\tSmoothed R map exists, adding path to df:\t\t{out_pth_R}")
                df.loc[idx, col_base_R] = out_pth_R
                skip_R = True

            # A. Find hippunfold surface
            surf_L, surf_R = get_surf_pth( # Get surface .func files
                    root=root_hu,
                    sub=sub,
                    ses=ses,
                    surf=surf,
                    lbl=lbl,
                    space="T1w"
                )
            
            if not chk_pth(surf_L) and not chk_pth(surf_R): # Check that surfaces exist
                
                # if missing hippunfold surface, three options:
                # 1- rawdata is missing
                # 2- micapipe processing error (e.g., no T1w image)
                # 3- hippunfold processing error (e.g., segmentation failed) 

                T1w_pth = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_space-fsnative_T1w.nii.gz"

                if not checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft): # if missing raw data
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    surf_L, surf_R = "NA: NO RAWDATA", "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe and hippunfold once resolved.\n")

                elif not chk_pth(pth = T1w_pth): # Check T1w from Micapipe outputs
                    dir_t1w = os.path.dirname(T1w_pth)
                    surf_L, surf_R = "NA: MISSING MP PROCESSING (nativepro T1w)", "NA: MISSING MP PROCESSING (nativepro T1w)"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING Nativepro T1w in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_t1w} ).\n")                    
                
                else: # hippunfold processing error
                    dir_surf = " ( " + os.path.commonpath([surf_L, surf_R]) + " ) " if surf_L and surf_R else "" # Find the common directory of both surf_L and surf_R
                    surf_L, surf_R = "NA: MISSING HU PROCESSING (surf)", "NA: MISSING MP PROCESSING (surf)"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface MISSING due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs {dir_surf}.\n") # could also check that the dir exists and further specify
                
                
                df.loc[idx, col_base_L] = surf_L
                df.loc[idx, col_base_R] = surf_R
                return df

            elif not chk_pth(surf_L) and not skip_L: # Must be hippunfold processing error (e.g., segmentation failed). Rawdata of micapipe processing problem would affect both hemis.                 
            
                dir_surf = os.path.dirname(surf_L)
                surf_L = "NA: MISSING HU PROCESSING (surf)"
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): L-hemi hippocampal surface missing due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs ( {dir_surf} ).\n") # could also check that the dir exists and further specify
            
                df.loc[idx, col_base_L] = surf_L

                if skip_R: 
                    return df
                else:
                    skip_L = True # cannot continue to smoothing for this hemi
                    if verbose: print(f"\t\tSurface (R only):\t{surf_R}")
 
            elif not chk_pth(surf_R) and not skip_R:

                dir_surf = os.path.dirname(surf_R)
                surf_R = "NA: MISSING HU PROCESSING (surf)"
                print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): R-hemi hippocampal surface missing due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs ( {dir_surf} ).\n") # could also check that the dir exists and further specify
            
                df.loc[idx, col_base_L] = surf_R

                if skip_L: 
                    return df
                else:
                    skip_R = True # cannot continue to smoothing for this hemi
                    if verbose: print(f"\t\tSurface (L only):\t{surf_L}")

            else:
                if verbose:
                    print(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")
            

            # A. Generate unsmoothed maps
            if ft == "thickness": # get path to unsmoothed map from hippunfold outputs
                pth_map_unsmth_L, pth_map_unsmth_R = get_surf_pth(root=root_hu, sub=sub, ses=ses, lbl="thickness", surf=surf, space="T1w")
            
            else: # generate feature map from volume and hippunfold surface
                vol_pth = get_mapVol_pth(root=root_mp, sub=sub, ses=ses, feature=ft) # get the volume path from micapipe outputs

                if chk_pth(vol_pth): # Check that volume exists.
                    
                    pth_save_map_unsmth_L = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-NA.func.gii"
                    pth_save_map_unsmth_R = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-NA.func.gii"
                    
                    if chk_pth(pth_save_map_unsmth_L): # if unsmoothed map already exists, do not recompute
                        pth_map_unsmth_L = pth_save_map_unsmth_L
                    else:
                        if not skip_L:
                            pth_map_unsmth_L = map(vol=vol_pth, surf=surf_L, out=pth_save_map_unsmth_L, verbose=verbose)
                        else:
                            pth_map_unsmth_L = None # must be declared
                    
                    if chk_pth(pth_save_map_unsmth_R): 
                        pth_map_unsmth_R = pth_save_map_unsmth_R
                    else:
                        if not skip_R:
                            pth_map_unsmth_R = map(vol=vol_pth, surf=surf_R, out=pth_save_map_unsmth_R, verbose=verbose)
                        else:
                            pth_map_unsmth_R = None # must be declared
                    
                    if not chk_pth(pth_map_unsmth_L) and not chk_pth(pth_map_unsmth_R): # check that unsmoothed paths exist. If not, it is a local processing error.
                        print(f"\t\t[ERROR] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}) Unsmoothed map could not compute.\n")
                        pth_map_unsmth_L, pth_map_unsmth_R = "NA: SCRIPT ERROR (unsmoothed map compute)", "NA: SCRIPT ERROR (unsmoothed map compute)"
                        df.loc[idx, col_base_L] = pth_map_unsmth_L
                        df.loc[idx, col_base_R] = pth_map_unsmth_R
                        return df
                    
                    elif not chk_pth(pth_map_unsmth_L) and not skip_L:
                    
                        print(f"\t\t[ERROR] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}). Could not compute unsmoothed map for L hemi.")
                        pth_map_unsmth_L = "NA: PROCESSING ERROR (unsmoothed map compute)"
                        df.loc[idx, col_base_L] = pth_map_unsmth_L
                        
                        if skip_R: 
                            return df
                        else:
                            skip_L = True
                    
                    elif not chk_pth(pth_map_unsmth_R) and not skip_R:
                        print(f"\t\t[ERROR] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}). Could not compute unsmoothed map for R hemi.")
                        pth_map_unsmth_R = "NA: PROCESSING ERROR (unsmoothed map compute)"
                        df.loc[idx, col_base_R] = pth_map_unsmth_R
                        
                        if skip_L:
                            return df
                        else:
                            skip_R = True
                    else:
                        if verbose:
                            print(f"\t\tUnsmoothed map paths:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
                    
                elif not checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft): # Unsmoothed map doesn't exist. Check raw data.
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    df.loc[idx, col_base_L] = "NA: NO RAWDATA"
                    df.loc[idx, col_base_R] = "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe and hippunfold once resolved.\n")
                    return df
                
                else: # Must be due to micapipe error
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Feature volume not found. Check micapipe processing ( {os.path.dirname(vol_pth)} ). Skipping.\n")
                    df.loc[idx, col_base_L] = "NA: MISSING MP PROCESSING (volume ft map)"
                    df.loc[idx, col_base_R] = "NA: MISSING MP PROCESSING (volume ft map)"
                    return df   
            
            # Add unsmoothed map column
            # Create col name
            base_name_L = os.path.basename(out_pth_L).replace('.func.gii', '')
            base_name_R = os.path.basename(out_pth_R).replace('.func.gii', '')
            base_name_L = base_name_L.replace(f"sub-{sub}_ses-{ses}_", '')
            base_name_R = base_name_R.replace(f"sub-{sub}_ses-{ses}_", '')
            col_unsmth_L = base_name_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = base_name_R.replace(f"_smth-{smth}mm", "_unsmth")

            print(f"\t\tUnsmoothed map cols:\t{col_unsmth_L}\t{col_unsmth_R}") 
            
            if not skip_L:
                df.loc[idx, col_unsmth_L] = pth_map_unsmth_L
                print(f"\t\tAdded L unsmoothed path: {pth_map_unsmth_L}")
            if not skip_R:
                df.loc[idx, col_unsmth_R] = pth_map_unsmth_R
                print(f"\t\tAdded R unsmoothed path: {pth_map_unsmth_R}")

            # B. Smooth map
            if not skip_L:
                pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, out_pth_L, kernel=smth, verbose=False)
                if not chk_pth(pth_map_smth_L):
                    pth_map_smth_L = f"NA: SCRIPT ERROR. Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}"
                df.loc[idx, col_base_L] = pth_map_smth_L
                if verbose:
                    print(f"\t\tSmoothed map L: {pth_map_smth_L}")
            
            if not skip_R:
                pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_L, out_pth_R, kernel=smth, verbose=False)
                if not chk_pth(pth_map_smth_R):
                    pth_map_smth_R = f"NA: SCRIPT ERROR. Surf: {surf_R}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}"
                
                df.loc[idx, col_base_R] = pth_map_smth_R
                if verbose:
                    print(f"\t\tSmoothed map R: {pth_map_smth_R}\n")
            
            return df

        
        start_time = time.time()
        print(f"[idToMap] idToMap function start time: {time.strftime('%d-%b-%Y %H:%M:%S', time.localtime(start_time))}.")
        if verbose:
            print("\t Finding/computing smoothed maps for provided surface, label, feature and smoothing combinations. Adding paths to dataframe...")

        if dict_demo['nStudies']: # Check if 'study' column exists, else skip or set default
            assert 'study' in df_demo.columns, "[idToMap] 'study' column not found in df_demo, but 'nStudies' is True in dict_demo."
        else: 
            if len(studies) > 1:
                raise ValueError("[idToMap] 'study' column not found in df_demo, but multiple studies provided.\n\tEither a) provide a 'study' column to df_demo to indicate which study each row belongs to OR b) keep a single dictionary item with study directory information.")
        
        for idx, row in df_demo.iterrows():
            study_code = row['study']
            
            if dict_demo['nStudies']: # determine study dictionary item
                study_item = next((s for s in studies if s['study'] == study_code), None)
                if study_item is None:
                    print(f"[idToMap] WARNING. Unknown study code `{study_code}`. Skipping row.")
                    continue
                else:
                    if verbose:
                        print(f"[idToMap] {idx} of {df_demo.shape[0]-1}...")
                    ID_col = dict_demo['ID_' + study_item['study']]
            else: # if no matches, then take first study
                print(f"[idToMap] No 'study' column provided. Defaulting to first study in studies list: {studies[0]['study']}.")
                study_item = studies[0]
                ID_col = dict_demo['ID']
            
            sub = row[ID_col]
            ses = row['SES']

            if idx % 10 == 0 and idx > 0: # progress statement every 10 rows
                percent_complete = 100 * idx / len(df_demo)
                print(f"Progress: {percent_complete:.1f}% of rows completed ({idx}/{len(df_demo)})")

            print(f"\n{study_code} sub-{sub} ses-{ses}")

            out_dir = f"{specs['prjDir_root']}{specs['prjDir_maps']}/sub-{sub}_ses-{ses}" # for saving smoothed maps
            create_dir(out_dir)

            if specs['ctx']:
                print(f"\n\tCORTICAL MAPS [{study_code} sub-{sub} ses-{ses}]...")
                
                for ft in specs['ft_ctx']:
                    for surf in specs['surf_ctx']:
                        for smth in specs['smth_ctx']:
                            for lbl in specs['lbl_ctx']:
                                df_demo = ctx_maps(out_dir=out_dir, study=study_item, df=df_demo, idx=idx, sub=sub, ses=ses, surf=surf, ft=ft, smth=smth, lbl=lbl, verbose=verbose)

                                
            if specs['hipp']:
                print(f"\n\tHIPPOCAMPAL MAPS [{study_code} sub-{sub} ses-{ses}]...")
                
                for ft in specs['ft_hipp']:
                    for surf in specs['surf_hipp']:
                        for smth in specs['smth_hipp']:
                            for lbl in specs['lbl_hipp']:
                                df_demo = hipp_maps(out_dir=out_dir, study=study_item, df=df_demo, idx=idx, sub=sub, ses=ses, surf=surf, ft=ft, smth=smth, lbl=lbl, verbose=verbose)
                                
          
            print('-'*100)
        
        end_time = time.time()
        elapsed = end_time - start_time
        mins, secs = divmod(elapsed, 60)
        print(f"\n[idToMap] idToMap completed at {time.strftime('%d-%b-%Y %H:%M:%S', time.localtime(end_time))} (run duration: {int(mins)}m:{int(secs):02d}s).")
        log_contents = log_stream.getvalue()
        return df_demo, log_contents

    finally:
        sys.stdout = old_stdout


def get_maps(df, mapCols, col_grp="grp", col_ID='MICs_ID', verbose=False):
    """
    Create dict item for each, study, feature, label, smoothing pair (including hippocampal)
    Note: multiple groups should be kept in same DF. Seperate groups later on

    Input:
        df: DataFrame with columns for ID, SES, Date, and paths to left and right hemisphere maps.
            NOTE. Asusme path columns end with '_L' and '_R' for left and right hemisphere respectively.
        ID_col: Column name for participant ID in the DataFrame. Default is 'MICS_ID'.
    
    Output:
        df_clean: Cleaned DataFrame with only valid ID-SES combinations, and paths to left and right hemisphere maps.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd
    
    assert col_ID in df.columns, f"[get_maps] df must contain 'ID' column. Cols in df: {df.columns}"
    assert 'SES' in df.columns, f"[get_maps] df must contain 'SES' column. Cols in df: {df.columns}"
    assert col_grp in df.columns, f"[get_maps] df must contain '{col_grp}' column. Cols in df: {df.columns}"
    
    # Assert that all columns in mapCols exist in df
    missing_cols = [col for col in mapCols if col not in df.columns]
    assert not missing_cols, f"[get_maps] The following columns from mapCols are missing in df: {missing_cols}"

    # find appropriate cols
    col_L = [i for i in mapCols if  ('hemi-L' in i)]
    col_R = [i for i in mapCols if ('hemi-R' in i)]
    assert len(col_L) == 1, f"[get_maps] more than one col with 'hemi-L'. {col_L}"
    assert len(col_R) == 1, f"[gete_maps] more than one col ending with 'hemi-R'. {col_R}"
    if verbose:
        print(f"[get_maps] {col_L}, {col_R}")
    col_L = col_L[0]
    col_R = col_R[0]

    # read in the maps and append to df_maps
    if 'UID' in df.columns:
        df_maps = df[['UID', col_ID, 'SES', col_L, col_R]]
    else:
        df_maps = df[[col_ID, 'SES', col_L, col_R]]
    
    if df_maps[col_L].shape[0] == 0 or df_maps[col_R].shape[0] == 0:
        print(f"[get_maps] WARNING. No valid entries found in df for columns {col_L} and/or {col_R}. Returning None.")
        return None
    # Stack all hemisphere maps into a DataFrame (vertices as columns)
    map_L_matrix = np.vstack([nib.load(x).darrays[0].data for x in df_maps[col_L]])
    map_R_matrix = np.vstack([nib.load(x).darrays[0].data for x in df_maps[col_R]])
    
    # Convert to DataFrame for easier handling
    map_L_df = pd.DataFrame(map_L_matrix, index=df_maps.index)
    map_R_df = pd.DataFrame(map_R_matrix, index=df_maps.index)

    # Rename columns to indicate hemisphere and vertex index
    map_L_df.columns = [f"{v}_L" for v in map_L_df.columns]
    map_R_df.columns = [f"{v}_R" for v in map_R_df.columns]
    
    df_maps = df_maps.drop(columns=[col_L, col_R]) # append map_L_df and map_R_df to df_maps. Remove the original columns col_L and col_R from df_maps
    #print(f"\tdf_maps cols: {df_maps.columns}")
    
    df_maps = pd.concat([df_maps, map_L_df, map_R_df], axis=1)
    #print(f"\tFinal shape:{df_maps.shape}")

    df_maps = setIndex(df=df_maps, col_ID=col_ID, sort=True) # index: <UID_>ID_SES
    
    # Keep only the vertex columns (map data columns)
    vertex_cols = [col for col in df_maps.columns if col.endswith('_L') or col.endswith('_R')]
    df_maps_clean = df_maps[vertex_cols]

    if verbose:
        print(f"\t[get_maps] Maps retrieved. Size: {df_maps_clean.shape}")
    
    return df_maps_clean

def setIndex(df, col_ID='MICs_ID', sort = True):
    """
    Set index of DataFrame to a combination of ID and SES.

    Input:
        df: DataFrame with columns for ID and SES.
        col_ID: Column name for participant ID in the DataFrame. Default is 'MICS_ID'.

    Output:
        DataFrame with index set to 'UID_ID_SES' (if UID in df) else 'ID_SES'.
    """
    import pandas as pd
    
    assert col_ID in df.columns, f"[setIndex] df must contain 'ID' column. Cols in df: {df.columns}"
    assert 'SES' in df.columns, f"[setIndex] df must contain 'SES' column. Cols in df: {df.columns}"
    
    if 'UID' in df.columns:
        df['UID_ID_SES'] = df.apply(lambda row: f"{row['UID']}_{row[col_ID]}_{row['SES']}", axis=1)
        df = df.set_index('UID_ID_SES')
    else:
        df['ID_SES'] = df.apply(lambda row: f"{row[col_ID]}_{row['SES']}", axis=1)
        df = df.set_index('ID_SES')
    
    if sort:
        df = df.sort_index()

    return df

def uniqueNAvals(df):
    """
    Determine unique NA values across all subjects.

    Input:
        df: DataFrame with map paths columns only (df_pths)

    Output:
        list of unique NA values (including "BLANK" for empty strings)
    """
    import pandas as pd
    
    out = pd.unique([
        x if x != None else "BLANK"
        for x in df.values.flatten()
        if isinstance(x, str) and (x == "" or x.startswith("NA:"))
    ])

    return out

def countErrors(df, cols, save=None, show=True):
    """
    From df_paths, count number of errors by row.

    Input:
        df: DataFrame with map paths columns only (df_pths)
        cols: List of columns to check for NA values/errors
        save: Path to save summary DataFrame. If None, do not save.
        show: If True, print unique error values and their counts.

    """
    import pandas as pd
    import datetime

    error_values = df[cols].values.flatten() # Flatten all error values in the selected columns
 
    error_values = [x for x in error_values if isinstance(x, str) and (x == None or x.startswith("NA:"))]  # Filter to only error values (blank or start with "NA:")

    error_counts = pd.Series(error_values).value_counts()  # Count occurrences of each unique error value

    # Determine all unique NA values across all subjects
    unique_na_values = uniqueNAvals(df[cols])

    # For each unique NA value, create a column listing which columns have this error for each row
    for na_val in unique_na_values:
        def colnames_with_error(row):
            # Use "" for BLANK, otherwise the actual na_val
            check_val = "" if na_val == "BLANK" else na_val
            return [col for col in cols if isinstance(row[col], str) and (row[col] == check_val)]
        colname = f'{na_val}'
        df[colname] = df.apply(colnames_with_error, axis=1)
        # Convert empty lists to blank string for clarity
        df[colname] = df[colname].apply(lambda x: x if x else "")
   
    def extract_features_from_cols(cols_with_error):  # For each row, extract features from columns with "NA:rawdata" error
        features = []
        if isinstance(cols_with_error, list):
            for col in cols_with_error:
                # Check for each feature in the column name
                if any(f in col for f in ['FA', 'ADC']):
                    features.append('DWI')
                elif 'T1map' in col:
                    features.append('T1map')
                elif 'flair' in col:
                    features.append('flair')
                elif 'thickness' in col:
                    features.append('thickness')
        # Remove duplicates and sort
        return ','.join(sorted(set(features)))
    #print(df.columns)
    
    df['NA: NO RAWDATA'] = df['NA: NO RAWDATA'].apply(extract_features_from_cols)
    df['NA: MISSING MP PROCESSING (volume ft map)'] = df['NA: MISSING MP PROCESSING (volume ft map)'].apply(extract_features_from_cols)

    # Prepare summary DataFrame
    df['NAs'] = df[cols].apply(lambda row: sum(1 for x in row if isinstance(x, str) and (x == "" or x.startswith("NA:"))), axis=1)
    error_cols = [f'{na_val}' for na_val in unique_na_values]
    error_summary = df[df['NAs'] > 0][['study', 'PNI_ID', 'MICS_ID', 'SES', 'NAs'] + error_cols].copy()
    
    if show:
        print("Unique error values and their counts:")
        print(error_counts) 
    
    if save:
        if save.endswith('/'):
            savePth = f"{save}03a_ERRsummary_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.csv"
        else:
            savePth = f"{save}/03a_ERRsummary_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.csv"
        
        error_summary.to_csv(savePth, index=False)
        print(f"[countErrors] Summary saved to {savePth}")

    return error_summary, savePth if save else None

def clean_demoPths(df, nStudies, save="/host/verges/tank/data/daniel/3T7T/z/maps/paths/",verbose=True):
    """
    Clean demoPths dataframe for missing data and saves changes. 
    
    Amend/remove rows in case:
        1) [amend] Missing one hemisphere pair, make complimentary hemisphere NA (to prevent unbalanced analyses) 
        2) [removal] NA for all smoothed maps
        3) [removal] Missing one study (3T or 7T) for a given ID-SES combination

    Input:
        df: DataFrame with columns for ID, SES, Study, and paths to left and right hemisphere maps.
            NOTE. Assume path columns contain 'hemi-*' indicating L and R hemisphere maps.
        nStudies: Number of studies (e.g., 2 for 3T and 7T)
        save: Directory to save cleaned DataFrame and removed cases DataFrame. If None, do not save.
    Output:
        df_clean: Cleaned DataFrame with only valid ID-SES combinations.
        df_rmv: DataFrame with removed cases and reason for removal.
    """
    
    import numpy as np
    import pandas as pd
    
    uniqueIDs = df[['MICS_ID']].drop_duplicates()
    print(f"[clean_demoPths] df has {df.shape[0]} rows with {len(uniqueIDs)} unique IDs")
    # Determine columns refering to L/R hemi of the same ft-lbl-surf-smth combination. 
    cols_L = [col for col in df.columns if 'hemi-L' in col]
    cols_R = [col for col in df.columns if 'hemi-R' in col]
    cols_L_smth = [col for col in cols_L if '_smth-' in col]
    cols_R_smth = [col for col in cols_R if '_smth-' in col]

    #print(f"cols_L: {cols_L}")
    #print(f"cols_R: {cols_R}")

    pairs = []
    for col_L in cols_L_smth: # Determine pairs
        col_R = col_L.replace('hemi-L', 'hemi-R') # check if hemi-R col exists
        if col_R in cols_R_smth:
            pairs.append((col_L, col_R))
        else:
            print(f"[clean_demoPths] WARNING: No matching hemi-R column found for {col_L}.")

    assert len(pairs) == len(cols_L_smth) == len(cols_R_smth), "[clean_demoPths] Mismatch in number of L/R hemisphere columns or pairs."

    cols = [col for pair in pairs for col in pair] # flatten list of pairs to a simple list of strings
    if verbose:
        print(f"Cols ({len(cols)}): {cols}")

    # Select only the relevant columns for map processing
    df_maps = df.copy()
    if verbose:
        print(f"\tInitial df: {df_maps.shape} (3T participants: {df_maps[df_maps['study'] == '3T']['MICS_ID'].nunique()}, 7T participants: {df_maps[df_maps['study'] == '7T']['PNI_ID'].nunique()}).")
        n_hipp = sum([col.startswith('hipp') for col in cols])
        n_ctx = sum([col.startswith('ctx') for col in cols])
        print(f"\tPaired cols ({len(cols)} cols making {len(pairs)} pairs; ctx = {n_ctx}, hipp = {n_hipp}): {cols}")

    df_rmv = pd.DataFrame(columns=list(df.columns) + ['rmv_reason']) # to store removed cases

    # Standardize missing value names
    unique_na_values = uniqueNAvals(df_maps)
    #print(f"Unique NA vals {len(unique_na_values)}: {unique_na_values}")

    for na_val in unique_na_values: # replace all these with np.nan
        check_val = np.nan if na_val == "BLANK" else na_val
        df_maps.replace(check_val, np.nan, inplace=True)

    # 1. [Ammend] unbalanced maps (ie. if missing one hemi, make map for corresponding map NA)
    for pair in pairs:
        col_L, col_R = pair
        
        # Find rows where one hemi is missing and the other is present
        unbalanced = df_maps[(df_maps[col_L].isnull() & df_maps[col_R].notnull()) | 
                                (df_maps[col_L].notnull() & df_maps[col_R].isnull())]
        
        if not unbalanced.empty:
            if verbose:
                print(f"[clean_demoPths] Found {len(unbalanced)} unbalanced rows for pair ({col_L}, {col_R}). Setting missing hemi to NA.")
                print(unbalanced[['MICS_ID', 'PNI_ID', 'SES', 'study', col_L, col_R]])
            # Set the present hemi to NA to maintain balance
            for idx, row in unbalanced.iterrows():
                if pd.isnull(row[col_L]) and pd.notnull(row[col_R]):
                    df_maps.at[idx, col_R] = np.nan
                elif pd.notnull(row[col_L]) and pd.isnull(row[col_R]):
                    df_maps.at[idx, col_L] = np.nan

    # 2. [Remove] NA for all maps    
    missingAll = df_maps[cols].isnull().all(axis=1)  # check if all columns are missing all paired values
    id_missingAll = df_maps[missingAll][['MICS_ID', 'PNI_ID', 'SES', 'study']]

    if not id_missingAll.empty:
        # Add to df_rmv with reason
        to_add = df_maps[missingAll].copy()
        for col in cols:
            to_add[col] = df.loc[to_add.index, col].values
        to_add['rmv_reason'] = 'missingAllMaps'
        df_rmv = pd.concat([df_rmv, to_add], ignore_index=True)

    if missingAll.sum() > 0:
        df_clean = df_maps[~missingAll]
        # Extract only ID-SES-Study combinations and format: [study] ID-SES
        id_ses_study = id_missingAll.apply(lambda row: f"[{row['study']}] {row['MICS_ID']}-{row['SES']}", axis=1).tolist()
    else:
        df_clean = df_maps.copy()
        if verbose:
            print("\tNo rows missing all columns.")
    
    print(f"[clean_demoPths] Removed {missingAll.sum()} rows for NA across all maps.")
    if missingAll.sum() > 0:
        print(f"\t{df_clean.shape[0]} rows remain with {df_clean['MICS_ID'].nunique()} unique IDs.")
        if verbose:
            print(f"Participants removed: {id_ses_study}")

    # 3. Ensure that at least one session per ID for each study. If not, remove the participant completely.
    # list of unique IDs
    # for each ID, check how many unique values are in the study column. If < nStudies, remove the participant completely.
    for id in uniqueIDs['MICS_ID'].values:
        studies_for_id = df_clean[df_clean['MICS_ID'] == id]['study']
        if studies_for_id.nunique() < nStudies:
            
            # Add to df_rmv with reason
            to_add = df_clean[df_clean['MICS_ID'] == id].copy()
            for col in cols:
                to_add[col] = df.loc[to_add.index, col].values
            to_add['rmv_reason'] = 'missingStudy'
            df_rmv = pd.concat([df_rmv, to_add], ignore_index=True)
            # Remove from df_clean
            df_clean = df_clean[df_clean['MICS_ID'] != id]

    print(f"[clean_demoPths] {df_rmv[df_rmv['rmv_reason'] == 'missingStudy']['MICS_ID'].nunique()} cases removed for missing study pair.")
    if verbose:
        print(f"\tParticipants removed for missing study pair: {df_rmv[df_rmv['rmv_reason'] == 'missingStudy']['MICS_ID'].unique().tolist()}")

    # Print number of unique participants in MICS_ID before and after cleaning
    print(f"\t{df_clean.shape[0]} rows remain with {df_clean['MICS_ID'].nunique()} unique IDs (total sessions: 3T={len(df_clean[df_clean['study'] == '3T']['MICS_ID'])}, 7T={len(df_clean[df_clean['study'] == '7T']['PNI_ID'])}).")

    if save is not None:
        date = pd.Timestamp.now().strftime("%d%b%Y-%H%M%S")
        out_pth = f"{save}/03b_demoPths_clean_{date}.csv"
        df_clean.to_csv(out_pth, index=False)
        print(f"[clean_demoPths] Saved cleaned df: {out_pth}")
        
        if not df_rmv.empty:
            rmv_pth = f"{save}/03c_demoPths_removed_{date}.csv"
            df_rmv.to_csv(rmv_pth, index=False)
            print(f"[clean_demoPths] Saved removed cases df: {rmv_pth}")

    return df_clean, df_rmv

def clean_pths(dl, method="newest", silent=True): # TODO. Add option to choose session combination to maximize present data
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
            df_clean = clean_ses(df, ID_col, method=method, verbose=True)
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

def clean_ses(df_in, col_ID="UID", method="oldest", save=None, col_study=None, verbose=False):
    """
    Choose the session to use for each subject.
        If subject has multiple sessions with map path should only be using one of these sessions.

    inputs:
        df: pd.dataframe with columns for subject ID, session, date and map_paths
            Assumes map path is missing if either : map_pth
        col_ID: column name for subject ID in the dataframe
        method: method to use for choosing session. 
            TODO. "max": return session code for each column such that you maximize present data.
            "newest": use most recent session
            "oldest": use oldest session in the list - equivalent to taking the first session
            {number}: session code to use (e.g., '01' or 'a1' etc)

    output:
        currently:
        df: pd.dataframe with only one session per subject, same columns as before
        TODO. df with unique participants in the rows and cols for each map path. Values are ',' sep list of session codes that have data for this map, ordered from most recent to least recent.  
    """
    
    import pandas as pd
    import datetime

    # checks
    if df_in.empty:
        print(f"[ses_clean] WARNING: Empty dataframe. Skipping.")
        return
    if col_ID not in df_in.columns:
        raise ValueError(f"[ses_clean] df must contain 'ID' column. Cols in df: {df_in.columns}")
    if 'SES' not in df_in.columns:
        raise ValueError(f"[ses_clean] df must contain 'SES' column. Cols in df: {df_in.columns}")
    if 'Date' not in df_in.columns:
        raise ValueError(f"[ses_clean] df must contain 'Date' column. Cols in df: {df_in.columns}")
    else: # format to allow date comparisons
        df_in['Date_fmt'] = pd.to_datetime(df_in['Date'], format='%d.%m.%Y', errors='coerce')
    if verbose: print(f"[ses_clean] Choosing session according to method: {method}")
    
    # find sessions per unique ID, order
    df = df_in.copy()
    
    #map_cols = get_mapCols(df_in.columns)
    #map_cols = [col for col in df_in.columns if col.contains('hemi-') and ('hemi-L' in col or 'hemi-R' in col)]

    if col_study is not None and col_study not in df_in.columns:
        raise ValueError(f"[ses_clean] df must contain 'study' column if col_study is provided. Cols in df: {df_in.columns}")
    elif col_study is not None: # for each ID, make a list of each session for that study. Order from most to least recent
        df_unique = df[[col_ID, col_study]].drop_duplicates()
        for row in df_unique.itertuples(index=False):
            id = getattr(row, col_ID)
            study = getattr(row, col_study)
            sessions = df[(df[col_ID] == id) & (df[col_study] == study)].sort_values(by='Date_fmt', ascending=False)['SES'].tolist()
            df_unique.loc[(df_unique[col_ID] == id) & (df_unique[col_study] == study), 'SES_list'] = ','.join(sessions)
    else:
        df_unique = df[[col_ID]].drop_duplicates()
        for row in df_unique.itertuples(index=False):
            id = getattr(row, col_ID)
            sessions = df[df[col_ID] == id].sort_values(by='Date_fmt', ascending=False)['SES'].tolist()
            df_unique.loc[df_unique[col_ID] == id, 'SES_list'] = ','.join(sessions)    

    # check each map col to see what sessions it is properly defined for
    
    #for col in map_cols:

    # Find repeated IDs (i.e., subjects with multiple sessions)
    # sort df by col_ID
    df = df_in.sort_values(by=[col_ID]).copy()
    if col_study is not None and col_study in df.columns:
        # find repeated IDs within each study
        df = df.sort_values(by=[col_study, col_ID])
        repeated_ids = df[df.duplicated(subset=[col_study, col_ID], keep=False)][[col_study, col_ID]].drop_duplicates()
    else:
        repeated_ids = df[df.duplicated(subset=col_ID, keep=False)][col_ID].unique()
    
    if verbose:
        if len(repeated_ids) == 0:
            print(f"\tNo repeated IDs found")

    rows_to_remove = []
    
    # Handle different studies
    if col_study is not None and len(repeated_ids) > 0:
        studies = df[col_study].unique()
        for study in studies:
            sub_df = df[df[col_study] == study]
            repeated_ids_study = sub_df[sub_df.duplicated(subset=col_ID, keep=False)][col_ID].unique()
            if len(repeated_ids_study) > 0:
                if verbose: print(f"\t[study] {len(repeated_ids_study)} IDs with multiple sessions found. Processing...")
                if method == "newest":
                    for id in repeated_ids_study:
                        sub_sub_df = sub_df[sub_df[col_ID] == id]
                        if sub_sub_df.shape[0] > 1:
                            idx_to_keep = sub_sub_df['Date_fmt'].idxmax()
                            idx_to_remove = sub_sub_df.index.difference([idx_to_keep])
                            rows_to_remove.extend(idx_to_remove)
                elif method == "oldest":
                    for id in repeated_ids_study:
                        sub_sub_df = sub_df[sub_df[col_ID] == id]
                        if sub_sub_df.shape[0] > 1:
                            idx_to_keep = sub_sub_df['Date_fmt'].idxmin()
                            idx_to_remove = sub_sub_df.index.difference([idx_to_keep])
                            rows_to_remove.extend(idx_to_remove)
                else:
                    # Assume method is a session code (e.g., '01', 'a1', etc)
                    for id in repeated_ids_study:
                        sub_sub_df = sub_df[sub_df[col_ID] == id]
                        if sub_sub_df.shape[0] > 1:
                            idx_to_remove = sub_sub_df[sub_sub_df['SES'] != method].index
                            rows_to_remove.extend(idx_to_remove)
            # check that for each study there are only as many rows as unique IDs
            
            # count all rows for this study
            rowsInStudy = df[df[col_study] == study].shape[0]
            uniqueIDsInStudy = df[df[col_study] == study][col_ID].nunique()
            if rowsInStudy != uniqueIDsInStudy:
                print(f"\t[WARNING] [{study}] there remain more rows ({rowsInStudy}) than unique IDs ({uniqueIDsInStudy}) in this study. Check output and code to ensure unique sessions per ID")
    elif len(repeated_ids) > 0:
        if verbose: 
            print(f"\t{len(repeated_ids)} IDs with multiple sessions found. Processing...")
        if method == "newest":
            for id in repeated_ids:
                sub_df = df[df[col_ID] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_fmt'].idxmax()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        elif method == "oldest":
            for id in repeated_ids:
                sub_df = df[df[col_ID] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_fmt'].idxmin()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        else:
            # Assume method is a session code (e.g., '01', 'a1', etc)
            for id in repeated_ids:
                sub_df = df[df[col_ID] == id]
                if sub_df.shape[0] > 1:
                    idx_to_remove = sub_df[sub_df['SES'] != method].index
                    rows_to_remove.extend(idx_to_remove)
        if df.shape[0] != df[col_ID].nunique():
            print(f"[ses_clean] [WARNING] There remain more rows ({df.shape[0]}) than unique IDs ({df[col_ID].nunique()}) for this study. Check output and code to ensure unique sessions per ID")
        if verbose:
            print(f"\tMultiple sessions for IDs: {df[df.duplicated(subset=col_ID, keep=False)][col_ID].unique()}")

    # Remove the rows marked for removal
    df = df.drop(rows_to_remove)

    if not verbose: 
        print(f"\t{df_in.shape[0] - df.shape[0]} rows removed, Change in unique IDs: {df_in[col_ID].nunique() - df[col_ID].nunique()}")
        print(f"\t{df.shape[0]} rows remaining")

    if save is not None:
        date = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        save_pth = f"{save}/03d_ses_clean_{date}.csv"
        df.to_csv(save_pth, index=False)
        print(f"[ses_clean] Saved cleaned dataframe to {save_pth}")

    return df, save_pth

def get_mapCols(allCols, split=True, verbose=False):
    """
    From a list of column names, return a list of map columns (one for each of L, R). 
    NOTE. Does not differentiate smoothed and unsmoothed maps.

    Assumes:
        `hemi-L` and `hemi-R` for L/R maps
        `_smth-` for smoothed maps
        `_unsmth-` for unsmoothed maps

    Input:
        allCols: pandas.core.indexes.base.Index <df.columns>
            list of column names
    
    Output:
        if unsmth = True: 
            cols_smth_L, cols_smth_R: list of smoothed map columns
            cols_unsmth_L, cols_unsmth_R: list of unsmoothed map columns
        else:
            cols_smth_L, cols_smth_R: list of smoothed map columns
    """
    cols_L = [col for col in allCols if 'hemi-L' in col]
    cols_R = [col for col in allCols if 'hemi-R' in col]
    if verbose:
        print(f"{len(cols_L) + len(cols_R)} map columns found.")

    if split:
        return cols_L, cols_R
    else:
        return cols_L + cols_R

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

def get_IDCol(study_name, demographics):
    """
    Determine study's ID column name

    Input:
        study_name: Name of the study ("MICs" or "PNI")
        demographics: dict  regarding demographics file. 
            Required keys: 
                'ID_7T'
                'ID_3T'
    Output:
        col_ID: column name for subject ID in the demographics file
    """
    if study_name == "MICs":
        col_ID = demographics['ID_3T']
    elif study_name == "PNI":
        col_ID = demographics['ID_7T']
    else:
        raise ValueError(f"Unknown study name: {study_name}")
    return col_ID

## Can be removed? ##
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

def extractMap(df_mapPaths, cols_L, cols_R, studies, demographics, region=None, verbose=False):
    """
    Extract map paths from a dataframe based on specified columns and optional subset string in col name.

    Input:
        df_mapPaths: pd.DataFrame 
            map paths kept in column passed in cols.
        cols_L, cols_R: list of str
            column names to extract from the dataframe.
        studies:
            list of dicts  regarding studies in the analysis.
            Each dict should contain:
                'name'
                'dir_root'
                'study'
                'dir_mp'
                'dir_hu'
        demographics: dict  regarding demographics file.
            Required keys:
                'pth'
                'ID_7T'
                'ID_3T'
                'SES'
                'date'
                'grp'
        region: string, optional
            specify cortex or hippocampus. If none, all columns passed will be extracted and region=None will be added to dict item.

        Returns:
        out_dl: list of dicts
            Each dict contains:
                'study': study name
                'region': 'cortex' or 'hippocampus'
                'surf': surface type and resolution (e.g., 'fsLR-5k')
                'label': label type (e.g., 'thickness', 'T1', 'FA', etc)
                'feature': feature type (e.g., 'thickness', 'T1', 'FA', etc)
                'smth': smoothing level (in mm)
                'df_demo': pd.DataFrame with demographics and map paths for the specific map
                'df_maps_unsmth': pd.DataFrame with only the unsmoothed map paths for the specific map (if applicable)
                'df_maps_smth': pd.DataFrame with only the smoothed map paths for the specific map
    """

    out_dl = []
    
    if region is not None:
        if region == "cortex" or region == "ctx":
            region == "cortex"
            subset = "ctx"
        elif region == "hippocampus" or region == "hipp":
            region == "hippocampus"
            subset = "hipp"
        else:
            raise ValueError(f"[extractMap] Unknown region: {region}. Should be 'cortex' or 'hippocampus'.")
        
        cols_L = [col for col in cols_L if subset in col]
        cols_R = [col for col in cols_R if subset in col]
        print(f"\nRegion {region}: {len(cols_L) + len(cols_R)} map columns found (col name pattern: {subset}).")
    else:
        print(f"\n{len(cols_L) + len(cols_R)} map columns found.")

    if cols_L == [] or cols_R == []:
        print("\n[extractMap] WARNING. No map columns found. Skipping.")
        return out_dl
        
    for col_L, col_R in zip(cols_L, cols_R):
        
        assert col_L.replace('hemi-L', '') == col_R.replace('hemi-R', ''), f"Left and right hemisphere columns do not match: {col_L}, {col_R}"
        
        # Find the substring after 'hemi-L' and 'hemi-R' that is common between col_L and col_R
        hemi_L_idx = col_L.find('hemi-L_') + len('hemi-L_')
        commonName = col_L[hemi_L_idx:]
        
        if verbose:
            print(f"\n\tProcessing {commonName}... (cols: {col_L} {col_R})")
        
        df_tmp = df_mapPaths.dropna(subset=[col_L, col_R]) # remove IDs with missing values in col_L or col_R
        if verbose:
            print(f"\t\t{len(df_mapPaths) - len(df_tmp)} rows removed due to missing values for these maps. [{(len(df_mapPaths))} rows before, {len(df_tmp)} rows remain]")
        
        # Remove participants who do not have data for all MRI studies in this analysis
        required_studies = [s['study'] for s in studies]
        participant_counts = df_tmp.groupby('UID')['study'].nunique()
        valid_ids = participant_counts[participant_counts == len(required_studies)].index.tolist()
        df_tmp_drop = df_tmp[~df_tmp['UID'].isin(valid_ids)].copy()
        df_tmp = df_tmp[df_tmp['UID'].isin(valid_ids)]

        n_before = df_mapPaths['UID'].nunique()
        n_after = df_tmp['UID'].nunique()
        n_removed = n_before - n_after

        if verbose:
            if n_removed > 0:
                print(f"\t\t{n_after} unique patients remain after removing {n_removed} IDs due to incomplete study.")
                print(f"\t\tIDs removed: {sorted(df_tmp_drop['UID'].unique())}")
        if n_after == 0:
            print(f"\t\t[extractMap] WARNING. No participants remain after filtering for complete study data. Skipping this map.")
            continue
        # TODO: extract only the columns relevant for statistics: IDs, age, sex, grp
        for study in studies:
            study_name = study['name']
            study_code = study['study']
            
            col_ID = get_IDCol(study_name, demographics) # determine ID col name for this study
            
            df_tmp_study = df_tmp[df_tmp['study'] == study_code] # filter for rows from this study
            
            if verbose:
                print(f"\t[{study_code}] {len(df_tmp_study)} rows")

            maps = get_maps(df_tmp_study, mapCols=[col_L, col_R], col_grp = demographics['grp'], col_ID = col_ID, verbose=verbose)
            
            # add to dict list
            surf = col_L.split('surf-')[1].split('_label')[0]
            lbl = col_L.split('_label-')[1].split('_')[0]
            if lbl == 'thickness':
                ft = 'thickness'
            else:
                ft = col_L.split('_label-')[1].split('_')[1]
            
            if '_unsmth' in commonName:
                smth = 'NA'
            else:
                smth = col_L.split('_smth-')[1].split('mm')[0]
            
            out_dl.append({
                'study': study_name,
                'region': region,
                'surf': surf,
                'label': lbl,
                'feature': ft,
                'smth': smth,
                'df_demo': df_tmp_study,
                'df_maps': maps,
            })
    
    if verbose:
        print(f"\n[extractMap] Returning list with {len(out_dl)} dictionary items (region: {region}).")
    
    return out_dl

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

######################### ANALYSIS FUNCTIONS ####################################

def winComp(dl, demographics, ctrl_grp, z, w, covars, col_grp, 
            save=True, save_pth=None, save_name="05_stats_winStudy", test=False, 
            verbose=False, dlPrint=False):
    """
    Comput within study comparisons between control distribution and all participants

    Input:
        dl: (list) of dictionaries with map and demographic data for each comparison
        demographics: (dict) with demographic column names
        ctrl_grp: (dict) with all control group patterns in the grouping column
        z: (bool) whether to compute z-scores
        w: (bool) whether to compute w-scores
        covars: (list) of covariates to ensure complete data for and to include in w-scoring.
        col_grp: (str) name of the grouping column in demographics dataframe

        save: (bool) <default: True>
            whether to save the output dict list as a pickle file
        save_name: (str) <default: "05_stats_winStudy">
        test: (bool) <default: False>
            whether to run in test mode (randomly select 2 dict items to run)
        verbose: (bool) <default: False>
            whether to print shapes and other info
        dlPrint: (bool) <default: False>
            whether to print summary of output dict list
    
    Output:
        dl_winStats: (list) of dictionaries with map and demographic data for each comparison, with added z and/or w score columns
    """
    import numpy as np
    import pandas as pd
    import os
    import pickle
    import time
    import datetime
    import logging

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.txt")
    print(f"[winComp] Saving log to: {log_file_path}")

    # Configure logging
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger()

    # Log the start of the function
    logger.info("Log started for winComp function.")

    try:
        print(f"[winComp] Saving log to: {log_file_path}")
        print(f"Computing within study comparisons. Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\tParameters: z={z}, w={w}, covars={covars}, col_grp={col_grp}, ctrl_grp={ctrl_grp}")
        print(f"\tDemographics columns: {demographics}")
        print(f"\tNumber of dictionary items to process: {len(dl)}")

        ctrl_values = [val for sublist in ctrl_grp.values() for val in sublist]

        if test:
            idx_len = 2 # number of indices
            idx = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
            dl_winStats = dl.copy()
            dl_iterate = [dl[i] for i in idx]
            print(f"[TEST MODE] Running z-scoring on {idx_len} randomly selected dict items: {idx}")
        else:
            dl_winStats = dl.copy()
            dl_iterate = dl.copy()  # Create a copy of the original list to iterate over
            
        for i, item in enumerate(dl_iterate): # can be parallelized
            
            study = item['study']
            col_ID = get_IDCol(study, demographics) # determine ID col name based on study name

            if test:
                printItemMetadata(item, idx[i])
            else:
                printItemMetadata(item, i)
                
            demo = item[f'df_demo'].copy() # contains all participants
            maps = item[f'df_maps'].copy() # contains all participants, indexed by <IUD_>ID_SES

            if verbose: 
                print(f"\tInput shapes:\t\t[demo] {demo.shape} | [maps] {maps.shape}")
            if demo.shape[0] == 0 or maps.shape[0] == 0:
                logger.warning(f"\tWARNING. No data in demo or maps dataframe. Skipping this dict item.")
                continue

            # 0.i Index appropriately
            col_ID = get_IDCol(study, demographics)
            col_SES = demographics['SES']
            if 'UID' in demo.columns:
                demo['UID_ID_SES'] = demo['UID'].astype(str) + '_' + demo[col_ID].astype(str) + '_' + demo[col_SES].astype(str) # concat UID, ID and SES into single col 
                demo.set_index('UID_ID_SES', inplace=True)
            else:
                demo['ID_SES'] = demo[col_ID].astype(str) + '_' + demo[col_SES].astype(str) # concat ID and SES into single col
                demo.set_index('ID_SES', inplace=True)

            # 0.ii Prepare covars
            if covars is None: # set defaults
                covars = [demographics['age'], demographics['sex']]
            
            covars_copy = covars.copy()
            for c in list(covars_copy): 
                if c not in demographics.keys(): # ensure covar is a key in demographics dict
                    logger.warning(f"\tWARNING. Covariate '{c}' not a key in the demographics dictionary. Skipping this covar.")
                    covars_copy.remove(c)
                    continue
                if c not in demo.columns: # ensure covar column exists in demo dataframe
                    logger.warning(f"\tWARNING. Covariate '{c}' not found in demographics dataframe. Skipping this covar.")
                    covars_copy.remove(c)
                    continue

            # 0.iia Format covar to numeric, dummy code if categorical
            exclude_cols = [col for col in demo.columns if col not in covars_copy]  # exclude all columns but covars
            demo_numeric, catTodummy_log = catToDummy(demo, exclude_cols = exclude_cols)
            print(f"\tConverted categorical covariates to dummy variables: \n\t{catTodummy_log}")

            # 0.iib Remove rows with missing covariate data
            missing_idx = []
            if w and (covars_copy == [] or covars_copy is None):
                logger.warning("[winComp] WARNING. No valid covariates specified. Skipping w-scoring.")
                w_internal = False
                demo_num = demo_numeric.copy() # keep all rows in demo_numeric
            elif w:
                w_internal = True
                covar_cols = [demographics[c] for c in covars_copy]
                demo_num = demo_numeric.loc[:, covar_cols].copy() # keep only covariate columns in demo dataframe
                missing_cols = demo_num.columns[demo_num.isnull().any()]

                if len(missing_cols) > 0:
                    # count total number of cases with missing data
                    missing_idx = demo_num.index[demo_num.isnull().any(axis=1)].tolist()
                    logger.warning(f"\t[winComp] WARNING. {demo_num.isnull().any(axis=1).sum()} indices with missing covariate values: {missing_idx}")
                    demo_num_clean = demo_num.dropna().copy()
                    maps_clean = maps.loc[demo_num_clean.index, :].copy()
                else:
                    missing_idx = []
                    demo_num_clean = demo_num.copy()
                    maps_clean = maps.copy()
            else: # remove rows with missing covariate data
                w_internal = False
            
            if w_internal and demo_num_clean.shape[0] < 5:
                logger.warning("[winComp] WARNING. Skipping w-scoring, ≤5 controls.")
                w = False
            
            # A. Create control and comparison subsets
            ids_ctrl = [j for j in demo[demo[col_grp].isin(ctrl_values)].index if j not in missing_idx]
            demo_ctrl = demo_num_clean.loc[ids_ctrl].copy() # extract indices from demo_num_clean
            maps_ctrl = maps_clean.loc[ids_ctrl].copy() # extract indices from maps_clean

            if verbose: 
                print(f"\tControl group shapes:\t[demo] {demo_ctrl.shape} | [maps] {maps_ctrl.shape}")
            
            demo_test = demo_num_clean.copy()
            maps_test = maps_clean.copy()

            if col_grp in demo_num_clean.columns:
                    demo_num_clean.drop(columns=[col_grp], inplace=True)

            if verbose: 
                print(f"\tTest group shapes:\t[demo] {demo_test.shape} | [maps] {maps_test.shape}")
            
            demo_ctrl = demo_numeric.loc[demo_ctrl.index, :].copy() # keep only rows in demo_ctrl
            demo_test = demo_numeric.loc[demo_test.index, :].copy() # keep only rows in demo_test

            if not test: # add ctrl_IDS to output dictionary item
                dl_winStats[i][f'ctrl_IDs'] = maps_ctrl.index # TODO. format to a list rather than an index object
            else:
                dl_winStats[idx[i]][f'ctrl_IDs'] = maps_ctrl.index
            
            # B. Calculate statistics    
            # B.i. Prepare output dataframes
            df_out = pd.DataFrame(index=maps_test.index, columns=maps.columns)
            if verbose:
                print(f"\tOutput shape:\t\t[map stats] {df_out.shape}")
            
            if z and demo_ctrl.shape[0] > 3:
                print(f"\tComputing z scores [{demo_ctrl.shape[0]} controls]...")
                start_time = time.time()
                
                z_scores = get_z(x = maps_test, ctrl = maps_ctrl)
                if not test:
                    dl_winStats[i]['df_z'] = z_scores
                else: dl_winStats[idx[i]]['df_z'] = z_scores
                duration = time.time() - start_time
                print(f"\t\tZ-scores computed in {int(duration // 60):02d}:{int(duration % 60):02d} (mm:ss).")

            elif z:
                if not test:
                    dl_winStats[i]['df_z'] = None
                else: dl_winStats[idx[i]]['df_z'] = None
                logger.warning("\tWARNING. Skipping z-score: ≤2 controls.")

            if w_internal and demo_ctrl.shape[0] > 5 * len(covars_copy): #  SKIP if fewer than 5 controls per covariate
                print(f"\tComputing w scores [{demo_ctrl.shape[0]} controls, {len(covars_copy)} covars]...")
                start_time = time.time()
                if demo_ctrl.shape[0] < 10 * len(covars_copy):
                    logger.warning(f"\t\tWARNING. INTERPRET WITH CAUTION: Few participants for number of covariates. Linear regression likely to be biased.")

                df_w_out = df_out.copy() # n row by p map cols
                
                df_w_out, w_models = get_w(map_ctrl = maps_ctrl, demo_ctrl=demo_ctrl, map_test = maps_test, demo_test = demo_test, covars=covars_copy)
                if not test:
                    dl_winStats[i]['df_w'] = df_w_out
                    dl_winStats[i]['df_w_models'] = w_models
                else: 
                    dl_winStats[idx[i]]['df_w'] = df_w_out
                    dl_winStats[idx[i]]['df_w_models'] = w_models
                duration = time.time() - start_time
                print(f"\t\tW-scores computed in {int(duration // 60):02d}:{int(duration % 60):02d} (mm:ss).")

            elif w_internal:
                if not test:
                    dl_winStats[i]['df_w'] = None
                    dl_winStats[i]['df_w_models'] = None
                else:
                    dl_winStats[idx[i]]['df_w'] = None
                    dl_winStats[idx[i]]['df_w_models'] = None
                logger.warning(f"\tWARNING. Skipping w-scoring, ≤{5 * len(covars_copy)} controls (5 * number of covars).\n\t\tInsufficient number of controls for number of covariates. [{demo_ctrl.shape[0]} controls, {len(covars_copy)} covars].\n\t\tGuidelines suggest at least 5-10 controls per covariate to ensure stable regression estimates.")

        # Save dictlist to pickle file
        if save:
            date = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
            if test:
                save_name = f"TEST_{save_name}"
            out_pth = f"{save_pth}/{save_name}_{date}.pkl"
            with open(out_pth, "wb") as f:
                pickle.dump(dl_winStats, f)
            logger.info(f"Saved map_dictlist with z-scores to {out_pth}")
        
        print(f"\nCompleted within study comparisons.\nEnd time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if dlPrint: # print summary of output dict list
            try:
                if test:
                    print_dict(dl_winStats, df_print=False, idx=idx)
                else:
                    print_dict(dl_winStats)
            except Exception as e:
                logger.error(f"Error printing dict: {e}")
                logger.error(dl_winStats)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

    logger.info("Log ended for winComp function.")
    return dl_winStats

def print_dict(dict, df_print=False, idx=None):
    """
    Print the contents of a dictionary with DataFrames in a readable format.
    Input:
        dict: list of dict items.
        df_print: bool
            if True, prints DataFrame contents; if False, only print the shape of the DF keys
        idx: list of ints
            if provided, only print the items at these indices in the dict list.
    Output:
        Prints the keys and values of each dictionary item.
    """
    import pandas as pd
    
    def mainPrint(k, v, df_print):
        import pandas as pd
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
            print(f"\t{k}: <DataFrame shape={v.shape}>")
            if df_print: print(f"\t{k}: {v}")
        elif isinstance(v, list) and all(isinstance(x, pd.DataFrame) for x in v):
            if df_print:
                for idx_df, df_v in enumerate(v):
                    print(f"\t{k}[{idx_df}]: {df_v}")
            else:
                shapes = [df_v.shape for df_v in v]
                print(f"\t{k}: <list (len {len(v)}) of DataFrames. Shapes : {shapes}>")
            #print(f"\t{k}: <DataFrame shape={v.shape}>")
            if df_print: print(f"\t{k}: {v}")
        else:
            print(f"\t{k}: {v}")

    if idx is not None:
        print(f"\n Printing the following {len(idx)} indices: {idx}")
        for i in idx:
            d = dict[i]
            print(f"\n[{i}]")
            print(f"\tKeys: {list(d.keys())}")
            for k, v in d.items():
                mainPrint(k,v, df_print)
        return
    else:
        print(f"\n Dict list length ({len(dict)} items)")
        for i, item in enumerate(dict):
            d = item
            print(f"\n[{i}]")
            print(f"\tKeys: {list(d.keys())}")

            for k, v in d.items():
                mainPrint(k,v, df_print)

def print_grpDF(dict, grp, study, hipp=False, df="pth"):
    # hipp option: only print items where 'hippocampal'==True
    import pandas as pd

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

def get_pair(dl, idx, mtch=['study', 'grp', 'label'], skip_idx=None):
    """
    Get corresponding idx for item in dictionary list.
    NOTE. Assumes only a single match exists.

    Input:
        dl: list of dictionary items with keys found in mtch
        idx: index of the item to find the pair for
        mtch: list of keys to match on.
        skip_idx: index to skip (e.g. if you want to avoid matching to a specific item or if previously processed others)
    
    Output:
        index of the matching item in the list, or None if not found
    """

    item = dl[idx]
    if skip_idx is not None:
        if idx not in skip_idx:
            skip_idx.append(idx)
    else:
        skip_idx = [idx]
        
    for j, other in enumerate(dl):
        if j not in skip_idx:
            match = True
            for key in mtch:
                if other.get(key, None) != item.get(key, False):
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

def get_d_old(col1, col2):
    """
    Given two columns, calculate cohen's D 
    """
    m1 = col1.mean()
    m2 = col2.mean()
    s1 = col1.std()
    s2 = col2.std()
    n1 = col1.count()
    n2 = col2.count()
    pooled_std = (((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))**0.5
    
    if pooled_std == 0:
        print("[get_d] WARNING: Pooled standard deviation is zero. Returning NaN for Cohen's D.")
        return float('nan')  # Avoid division by zero
    
    d = (m1 - m2) / pooled_std
    
    if d == float('inf') or d == float('-inf'):
        print("[get_d] WARNING: Cohen's D is infinite. Returning NaN.")
        return float('nan')  # Avoid infinite values

    return d   

def get_d(ctrl, test, varName="z", test_name="grp", saveN = False):
    """
    Accept 2 2D matrices of size [n_ctrl x n_vertices] and [n_test x n_vertices]. Note n_vertices must be equivalent but n_ctrl need not = n_test.
    Calculates d with test - ctrl (i.e. positive d means test > ctrl)
    Return:
        Matrix of size 5 x n_vertices with the following rows:
            m_ctrl: mean of ctrl at this vertex
            std_ctrl: st.dev of ctrl at this vertex
            m_{test}
            std_{test}
            d_{test}: cohen's d between both distributions at this vertex
    """
    import numpy as np
    import pandas as pd

    # get means and stds
    n_ctrl  = ctrl.shape[0]
    m_ctrl = np.mean(ctrl, axis=0)
    std_ctrl = np.std(ctrl, axis=0)
    
    n_test = test.shape[0]
    m_test = np.mean(test, axis=0)
    std_test = np.std(test, axis=0)

    pooled_std = (((n_ctrl - 1) * std_ctrl**2 + (n_test - 1) * std_test**2) / (n_ctrl + n_test - 2))**0.5
    # Find any vertices where pooled_std is 0 and set to nan to avoid division by zero
    if np.any(pooled_std == 0):
        print(f"[get_d] WARNING: {np.sum(pooled_std == 0)} vertices have pooled_std == 0. Setting d to NaN for these vertices.")
        pooled_std = np.where(pooled_std == 0, np.nan, pooled_std)
    
    d = (m_test - m_ctrl) / pooled_std

    # create output df
    if saveN:
        n_vertices = test.shape[1] 
        n_ctrl_arr = np.full(n_vertices, n_ctrl) # Extend all arrays to the length of test[1] (number of vertices)
        n_test_arr = np.full(n_vertices, n_test)
        out = np.vstack([n_ctrl_arr, m_ctrl, std_ctrl, n_test_arr, m_test, std_test, d])
        row_names = [f'n_{varName}_ctrl',f'm_{varName}_ctrl', f'std_{varName}_ctrl', f'n_{varName}_{test_name}', f'm_{varName}_{test_name}', f'std_{varName}_{test_name}', f'd_{varName}_{test_name}']
    else:
        out = np.vstack([m_ctrl, std_ctrl, m_test, std_test, d])
        row_names = [f'm_{varName}_ctrl', f'std_{varName}_ctrl', f'm_{test_name}', f'std_{test_name}', f'd_{test_name}']
    
    out_df = pd.DataFrame(out, index=row_names, columns=test.columns)
    
    return out_df


def get_z(x, ctrl):
    """
    Calculate z-scores for a given value in a DataFrame, using the control group as the reference.
    
    inputs:
        x: value for specific subject and same column as col_ctrl
        ctrl: vector: 
            column name for control group in the DataFrame

    outputs:
        z: z-scores for the specified column
    """
    import pandas as pd
    import numpy as np

    # Ensure x and ctrl are DataFrames or numpy arrays
    x = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
    ctrl = pd.DataFrame(ctrl) if not isinstance(ctrl, pd.DataFrame) else ctrl

    ctrl_mean = ctrl.mean(axis=0)
    ctrl_std = ctrl.std(axis=0)

    # Broadcast subtraction and division for all columns
    z = (x - ctrl_mean) / ctrl_std

    return z

def get_w(map_ctrl, demo_ctrl, map_test, demo_test, covars, verbose=False):
    """
    Efficiently compute W-scores for patient maps based on control maps and demographics.
    Supports 2D DataFrames for map_ctrl and map (n_subjects x n_vertices).

    Input:
        map_ctrl: DataFrame (n_controls x n_vertices)
        demo_ctrl: DataFrame (n_controls x covariates)
        map_test: DataFrame (n_subjects x n_vertices)
        demo_test: DataFrame (n_subjects x covariates)
        covars: list of covariate column names
        verbose: bool, whether to print progress messages

    Output:
        w: DataFrame (n_subjects x n_vertices)
        model: DataFrame (regression coefficients and residual std per vertex)
    """
    import numpy as np
    import pandas as pd
    
    # Prepare covariate matrices
    X_ctrl = demo_ctrl[covars].values.astype(float)
    X_ctrl = np.hstack([np.ones((X_ctrl.shape[0], 1)), X_ctrl])  # add intercept column

    X_test = demo_test[covars].values.astype(float)
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])  # add intercept column

    # Prepare output containers
    models = pd.DataFrame(
        index=['intercept'] + [str(c) for c in covars] + ['resid_std'], # creates n_covar + 2 rows
        columns=map_test.columns # n_vertices columns
    )
    w = pd.DataFrame(index=map_test.index, columns=map_test.columns)

    # Convert map_ctrl and map to numpy arrays
    Y_ctrl = map_ctrl.values.astype(float)  # shape: n_controls x n_vertices
    Y_test = map_test.values.astype(float)  # shape: n_subjects x n_vertices

    # Efficient batch regression for each vertex
    for i, col in enumerate(map_test.columns):
        if verbose and i % 200 == 0:
            print(f"\r\t\t {(100*(i+1) / len(map_test.columns)):.0f}%\n", end="")

        y_ctrl = Y_ctrl[:, i] # extract col i
        if np.all(y_ctrl == 0):
            if verbose:
                print(f"{col} fully 0 in control map. skipping.")
            w[col] = np.nan
            models[col] = np.nan
            continue

        # Fit linear regression
        coef, _, _, _ = np.linalg.lstsq(X_ctrl, y_ctrl, rcond=None)
        predicted_ctrl = X_ctrl @ coef # shape: n_controls
        resid = y_ctrl - predicted_ctrl # shape: n_controls
        resid_std = np.std(resid) # shape: 1

        # Store model
        models.loc[models.index[:-1], col] = coef
        models.loc['resid_std', col] = resid_std

        # Predict expected values for all subjects
        expected = X_test @ coef
        w[col] = (Y_test[:, i] - expected) / resid_std # CAREFUL: may not be appropriate to divide by residual_std if fat tails

    return w, models

def catToDummy(df, exclude_cols=None, verbose = False):
    """
    Convert categorical (string) columns to dummy codes.
    
    Parameters:
    df: DataFrame to process
    exclude_cols: List of columns to exclude from conversion
    
    Returns:
    df_converted: DataFrame with categorical variables converted to dummy codes
    conversion_log: Dictionary logging all conversions made
    """
    import pandas as pd

    if exclude_cols is None:
        exclude_cols = []
    
    df_converted = df.copy()
    conversion_log = {}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        # Check if column is non-numeric (contains strings)
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            try:
                # Try to convert to numeric first
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                # Column contains non-numeric values, convert to dummy codes
                unique_vals = df[col].dropna().unique()
                
                if len(unique_vals) == 2:
                    # Binary variable - simple 0/1 encoding
                    val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                    df_converted[col] = df[col].map(val_map)
                    conversion_log[col] = {
                        'type': 'binary',
                        'mapping': val_map,
                        'original_values': list(unique_vals)
                    }
                    if verbose:
                        print(f"[catToDummy] Binary conversion for '{col}': {val_map}")
                    
                elif len(unique_vals) > 2:
                    # Multi-category variable - one-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                    
                    # Drop original column and add dummy columns
                    df_converted = df_converted.drop(columns=[col])
                    df_converted = pd.concat([df_converted, dummies], axis=1)
                    
                    conversion_log[col] = {
                        'type': 'one_hot',
                        'new_columns': list(dummies.columns),
                        'original_values': list(unique_vals)
                    }
                    print(f"[convert_categorical] One-hot encoding for '{col}': {list(unique_vals)} -> {list(dummies.columns)}")
    
    return df_converted, conversion_log

def printItemMetadata(item, return_txt = False, idx=None, clean = False):
    """
    Print metadata of a dictionary item in a readable format.
    
    Parameters:
    item: Dictionary containing metadata
    
    Returns:
    None
    """
    import pandas as pd
    
    study = item.get('study', None)
    if study is None:
        study = item.get('studies', None)
    region = item.get('region', None)
    feature = item.get('feature', None)
    surf = item.get('surf', None)
    label = item.get('label', None)
    smth = item.get('smth', None)
    if idx is not None:
        txt = f"\t[{study}] - {region}: {feature}, {surf}, {label}, {smth}mm (idx {idx})"
    else:
        txt = f"\t[{study}] - {region}: {feature}, {surf}, {label}, {smth}mm"
    
    if clean:
        txt = txt.replace('\t', '')
        # Remove other problematic characters
        txt = txt.replace('\x09', '')  # Another way tabs might appear
        # Replace newlines with spaces if needed
        txt = txt.replace('\n', '')
    
    if return_txt:
        return txt
    else:
        print(txt)
        
def relabel_vertex_cols(df, ipsiTo=None, n_vertices=32492):
    """
    Take df with columns '{idx}_{hemi}' and rename to just contain an index. By convention, L hemi then R hemi. 
    ipsiTo provides correspondence between ipsi suffix and hemisphere. ipsiTo also indicates if the columns are ipsi/contra flipped.

    Input:
        df: vertex-wise dataframe with vertex in columns, pts in rows. All vertices from both hemispheres should be present.
            Number of columns per hemisphere should be 32492 for fsLR-32k
        n_vertices: number of vertices per hemisphere (default is 32492 for fsLR-32k)
        ipsiTo: if provided, searches for columns ending with '_ipsi' and '_contra' and maps '_ipsi' indices to  
    """
      
    new_cols = []
    for col in df.columns:
        if ipsiTo is not None:
            if col.endswith('_ipsi'):
                idx = int(col.replace('_ipsi', ''))
                if ipsiTo == 'L':
                    new_cols.append(idx)
                elif ipsiTo == 'R':
                    new_cols.append(idx + n_vertices)
            elif col.endswith('_contra'):
                idx = int(col.replace('_contra', ''))
                if ipsiTo == 'L':
                    new_cols.append(idx + n_vertices)
                elif ipsiTo == 'R':
                    new_cols.append(idx)
        
        else: # columns are in the format '{idx}_L' and '{idx}_R'
            if col.endswith('_L'):
                idx = int(col.replace('_L', ''))
                new_cols.append(idx)
            elif col.endswith('_R'):
                idx = int(col.replace('_R', '')) + n_vertices
                new_cols.append(idx)
            else:
                new_cols.append(col)  # keep as is if not a vertex column
    
    # Create a mapping of old to new columns
    col_map = dict(zip(df.columns, new_cols))
    
    # Sort column
    sorted_cols = sorted([c for c in new_cols if isinstance(c, int)]) + [c for c in new_cols if not isinstance(c, int)]
    
    # Reindex dataframe columns
    df_renamed = df.rename(columns=col_map)
    df_renamed = df_renamed[sorted_cols]
    return df_renamed

def apply_glasser(df, ipsiTo=None, labelType='glasser_int'):
    """
    Input:
        df: vertex-wise dataframe with vertex in columns, pts in rows. All vertices from both hemispheres should be present.
            Number of columns per hemisphere should be 32492 for fsLR-32k
        labelType: final label to return. options:
            - 'glasser_int': integer [0:360] indicating glasser region
            - 'glasser_str': string with glasser region name
            - 'glasser_long': string with long glasser region name (e.g. 'V1d', 'V1v', etc)
            - 'lobe': string with lobe name (e.g. 'frontal', 'parietal', etc)
    Returns:
        df_glasser: mean values per region for the glasser atlas.
       
    """
    import pandas as pd

    glasser_df = pd.read_csv("/host/verges/tank/data/daniel/parcellations/glasser-360_conte69.csv", header=None, names=["glasser"]) # index is vertex num, value is region number
    df_relbl = relabel_vertex_cols(df, ipsiTo) # remove '_L' or '_R'/'_ipsi' or '_contra' suffixes from column names, and convert to integer indices
    
    df_relbl.columns = glasser_df['glasser'].values[df_relbl.columns.astype(int)]

    if labelType != 'glasser_int':
        glasser_details = pd.read_csv("/host/verges/tank/data/daniel/parcellations/glasser_details.csv")
        if labelType == 'glasser_str':
            gd_col  = 'RegionName'
        elif labelType == 'glasser_long':
            gd_col = 'regionLongName'
        elif labelType == 'lobe':
            gd_col = 'Lobe'
        elif labelType == 'cortex':
            gd_col = 'cortex'
        elif labelType == 'LobeLong':
            gd_col = 'LobeLong'
        elif labelType == 'Lobe_hemi':
            gd_col = 'Lobe_hemi'
    
        df_relbl.columns = df_relbl.columns.map(
            lambda x: glasser_details.loc[glasser_details['regionID'] == x, gd_col].values[0]
            if x in glasser_details['regionID'].values else x
        )

    return df_relbl
            
def glasser_mean(df_glasserLbl):
    """
    Calculate the mean values per region for the glasser atlas.
    Input:
        glasser_df: vertex-wise dataframe with vertex in columns, pts in rows.

    Returns:
        df_glasser_mean: dataframe with mean values per region for the glasser atlas.    
    """
    df_glasser_mean = df_glasserLbl.groupby(df_glasserLbl.columns, axis=1).mean().reset_index(drop=True)


######################### VISUALIZATION FUNCTIONS #########################

def plotMatrices(dl, key, name_append=None, show=False, save_pth=None, test=False):
    """
    Plot matrix visualizations for map values from corresponding study

    dl: 
        dictionary list with paired items from different studies
    key:
        key in the dictionary items to plot (e.g., 'map_smth')
    name_append:
        if provided, append this string to the filename when saving
    show:
        if True, show the plots interactively
    save_pth:
        if provided, save the plots to this path instead of showing them interactively
    """
    import matplotlib.pyplot as plot
    from matplotlib import gridspec
    import numpy as np
    import datetime

    skip_idx = []
    counter = 0

    print(f"Plotting matrices for {key}...")
    
    if test:
        print("TEST MODE: Randomly choosing 2 pairs to plot")
        import random
        # randomly reorder dl items
        random.shuffle(dl)

    for idx, item in enumerate(dl):
        if idx in skip_idx:
            continue
        else:
            skip_idx.append(idx)
    
        counter = counter+1
        
        idx_other = get_pair(dl, idx = idx, mtch=['region', 'surf', 'label', 'feature', 'smth'], skip_idx=skip_idx)
        if idx_other is None:
            print(f"\tWARNING. No matching index found for: {printItemMetadata(item, idx=idx)}.\nSkipping.")
            continue
        skip_idx.append(idx_other)
        
        item_other = dl[idx_other]
        if item_other is None:
            print(f"\tWARNING. Item other is None: {printItemMetadata(item, idx=idx)}.\nSkipping.")
            continue

        study = item['study']
        if study == 'MICs':
            idx_tT = idx
            idx_sT = idx_other

            item_tT = item
            item_sT = item_other
        else:
            idx_tT = idx_other
            idx_sT = idx

            item_tT = item_other
            item_sT = item
        
        if item_tT is None and item_sT is None:
            print(f"\tWARNING. Both items are None (3T: {printItemMetadata(item_tT, idx=idx)}, 7T: {printItemMetadata(item_sT, idx=idx)}).\nSkipping.")
            continue
        elif item_tT is None:
            print(f"\tWARNING. Item_tT is None: {printItemMetadata(item_tT, idx=idx)}.\nSkipping.")
            continue
        elif item_sT is None:
            print(f"\tWARNING. Item_sT is None: {printItemMetadata(item_sT, idx=idx)}.\nSkipping.")
            continue

        title_tT = f"{key} {item_tT['study']} [idx: {idx_tT}]"
        title_sT = f"{key} {item_sT['study']} [idx: {idx_sT}]"
        
        feature_tT = item_tT['feature']
        feature_sT = item_sT['feature']

        df_tT = item_tT.get(key, None)
        df_sT = item_sT.get(key, None)
        
        if df_tT is None and df_sT is None:
            print(f"\tWARNING. Missing key '{key}'. Skipping {printItemMetadata(item_tT, clean=True)} and {printItemMetadata(item_sT, clean=True)}\n")
            continue
        elif df_tT is None:
            print(f"\tWARNING. Missing key '{key}' for {printItemMetadata(item_tT, clean=True)}. Skipping.\n")
            continue
        elif df_sT is None:
            print(f"\tWARNING. Missing key '{key}' for {printItemMetadata(item_sT, clean=True)}. Skipping.\n")
            continue

        # determine min and max values across both matrices for consistent color scaling
        assert feature_tT == feature_sT, f"Features do not match: {feature_tT}, {feature_sT}"
        assert item_tT['region'] == item_sT['region'], f"Regions do not match: {item_tT['region']}, {item_sT['region']}"
        assert item_tT['surf'] == item_sT['surf'], f"Surfaces do not match: {item_tT['surf']}, {item_sT['surf']}"
        assert item_tT['label'] == item_sT['label'], f"Labels do not match: {item_tT['label']}, {item_sT['label']}"
        assert item_tT['smth'] == item_sT['smth'], f"Smoothing kernels do not match: {item_tT['smth']}, {item_sT['smth']}"
        
        if key == "df_z" or key == "df_w":
            cmap = "seismic"
            min_val = -3
            max_val = 3
        else:
            cmap = 'inferno'
            if feature_tT.lower() == "thickness":
                min_val = 0
                max_val = 4
                cmap = 'Blues'
            elif feature_tT.lower() == "flair":
                min_val = -500
                max_val = 500
                cmap = "seismic"
            elif feature_tT.lower() == "t1map":
                min_val = 1000
                max_val = 2800
                cmap = "inferno"
            elif feature_tT.lower() == "fa":
                min_val = 0
                max_val = 1
                cmap="Blues"
            elif feature_tT.lower() == "adc": # units: mm2/s
                min_val = 0
                max_val = 0.0025
                cmap = "Blues"
            else:
                min_val = min(np.percentile(df_sT.values, 95), np.percentile(df_tT.values, 95))
                max_val = max(np.percentile(df_sT.values, 5), np.percentile(df_tT.values, 5))

        # Create a grid layout with space for the colorbar
        fig = plot.figure(figsize=(30, 25))
        spec = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.43)

        # Create subplots
        ax1 = fig.add_subplot(spec[0])
        ax2 = fig.add_subplot(spec[2])

        # Plot the matrices
        visMatrix(df_tT, feature=feature_tT, title=title_tT, 
                  return_fig=False, show_index=True, ax=ax1, min_val=min_val, max_val=max_val, cmap=cmap, nan_side="left")
        visMatrix(df_sT, feature=feature_sT, title=title_sT, 
                  return_fig=False, show_index=True, ax=ax2, min_val=min_val, max_val=max_val, cmap=cmap, nan_side="right")

        # Add a colorbar between the plots
        if feature_tT.upper() == "ADC":
            cmap_title = "ADC (mm²/s)"
        elif feature_tT.upper() == "T1MAP":
            cmap_title = "T1 (ms)"
        else:
            cmap_title = feature_tT

        if key== "df_z":
            cmap_title = f"Z-score [{cmap_title}]"
        elif key == "df_w":
            cmap_title = f"W-score [{cmap_title}]"

        cbar_ax = fig.add_subplot(spec[1])
        norm = plot.Normalize(vmin=min_val, vmax=max_val)
        cbar = plot.colorbar(plot.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar.set_label(cmap_title, fontsize=20, labelpad=0)
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(axis='x', direction='in', labelsize=20)

        # Add a common title
        region = item_tT['region']
        surface = item_tT['surf']
        label = item_tT['label']
        smth = item_tT['smth']
        fig.suptitle(f"{region}: {feature_tT}, {surface}, {label}, {smth}mm", fontsize=30, y=0.9)

        if show:
            plot.show()

        if save_pth is not None:
            if name_append is not None:
                save_name = f"{region}_{feature_tT}_{surface}_{label}_smth-{smth}mm_{name_append}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
            else:
                save_name = f"{region}_{feature_tT}_{surface}_{label}_smth-{smth}mm_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
            fig.savefig(f"{save_pth}/{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"\tSaved: {save_pth}/{save_name}.png")
            plot.close(fig)
        
        if test and counter >= 2:
            break

def visMatrix(df, feature="Map Value", title=None, min_val=None, max_val=None, 
              cmap='seismic', return_fig=True, show_index=False, ax=None, nan_color='green', nan_side="right"):
    """
    Visualizes a matrix from a pandas DataFrame using matplotlib's imshow, with options for colormap, value range, and axis customization.
    Parameters
    ----------
    df : pandas.DataFrame
        The data to visualize, where rows are indices and columns are vertices.
    feature : str, optional
        Label for the colorbar (default is "Map Value").
    title : str, optional
        Title for the plot.
    min_val : float, optional
        Minimum value for colormap scaling. If None, uses the minimum of the data.
    max_val : float, optional
        Maximum value for colormap scaling. If None, uses the maximum of the data.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for visualization (default is 'seismic').
    return_fig : bool, optional
        If True, returns the matplotlib Figure object; otherwise, returns the Axes object.
    show_index : bool, optional
        If True, displays DataFrame index labels on the y-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure and axes.
    
    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes containing the visualization, depending on `return_fig`.
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        print("[visMatrix] WARNING: DataFrame is empty or None. Skipping visualization.")
        return None
    
    # Convert DataFrame to numpy array, mask NaNs
    data = df.values
    mask = np.isnan(data)

    # Use provided cmap parameter
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    
    cmap_obj.set_bad(color=nan_color) # Color for NaN values

    if min_val is None:
        min_val = np.nanmin(data)
    if max_val is None:
        max_val = np.nanmax(data)
    
    # Use provided axes or create new figure
    anyNaN = np.isnan(data).any()
    if ax is None:
        fig_length = max(6, min(30, 0.1 * data.shape[0]))
        fig_width = 10
        if anyNaN:  # Increase width if NaN annotations are present
            fig_width += 2
        fig, ax = plt.subplots(figsize=(fig_width, fig_length))
        create_colorbar = True
    else:
        fig = ax.get_figure()
        create_colorbar = False  # Let caller handle colorbar
    
    im = ax.imshow(data, aspect='auto', cmap=cmap_obj, vmin=min_val, vmax=max_val, interpolation='none')
    im.set_array(np.ma.masked_where(mask, data))

    if anyNaN:
        print(f"\tNaN values present [{title}: {feature}]")
        for i, row in enumerate(data):
            nan_count = np.isnan(row).sum()
            if nan_count > 0:  # Annotate next to the row
                if nan_side == "right":
                    ax.annotate(f"NAN: {nan_count}", 
                                xy=(data.shape[1], i), 
                                xytext=(data.shape[1] + 1, i),  # Place outside the plot
                                va='center', ha='left', fontsize=9, color='black')
                elif nan_side == "left":
                    ax.annotate(f"NAN: {nan_count}", 
                                xy=(-1, i), 
                                xytext=(-2, i),  # Place outside the plot
                                va='center', ha='right', fontsize=9, color='black')
            
    if title:
        ax.set_title(title, fontsize=23)

    ax.set_xlabel("Vertex", fontsize=15)
    ax.tick_params(axis='x', labelsize=10)
    
    if create_colorbar:
        if feature.upper == "ADC":
            feature = "ADC (mm²/s)"
        elif feature.upper == "T1MAP":
            feature = "T1 (ms)"
        cbar = plt.colorbar(im, ax=ax, label=feature, shrink=0.25)
        cbar.ax.tick_params(axis='y', direction='in', length=5)  # Place ticks inside the color bar)

    if show_index:
        if nan_side == "left":# nan_side and index side should be different
            ax.yaxis.set_label_position("right") 
            ax.yaxis.tick_right()
        else:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index.astype(str), fontsize=8)
        ax.tick_params(axis='y', pad=5, labelsize=14)

    if return_fig:
        return fig
    else:
        if ax is None:
            plt.show()
        return ax
    
def pngs2pdf(fig_dir, output=None, verbose=False):
    """
    Combine multiple pngs held in same folder to a single pdf.
    Groups pngs with variation only in "_smth-*" or "_stat-*" values.

    Input:
        fig_dir: Directory containing png files.
        output: Directory to save output pdf files. If None, saves in fig_dir.
    Output:
        PDF files saved in output directory.
    """
    
    import os
    import re
    from PIL import Image
    
    if output is None:
        output = fig_dir
    else:
        if not os.path.exists(output):
            os.makedirs(output)
            print(f"Created output directory: {output}")
    
    # Find all PNG files in the directory
    files = [f for f in os.listdir(fig_dir) if os.path.isfile(os.path.join(fig_dir, f)) and f.endswith('.png')]

    # Extract base names (excluding smoothing pattern)
    def get_base_name(filename):
        filename = re.sub(r"_smth-(\d+|NA)mm", "", filename)  # Remove smoothing pattern
        filename = re.sub(r"_stat-(\w+)_", "_", filename)  # Remove stat pattern
        filename = re.sub(r"-(\d{6})(?=\.\w+$)", "", filename)  # Remove time pattern
        return filename
    
    # Group files by their base name
    file_groups = {}
    for f in files:
        base_name = get_base_name(f)
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(f)
    if verbose:
        print(f"Creating {len(file_groups)} PDFs...")

    # Create a PDF for each group
    for base_name, group_files in file_groups.items():
        # sort files
        group_files = sorted(group_files, key=lambda f: re.search(r"_stat-(\w+)_", f).group(1) if re.search(r"_stat-(\w+)_", f) else "") # alphabetically by stat name
        group_files = sorted(group_files, key=lambda f: int(re.search(r"_smth-(\d+)mm", f).group(1)) if re.search(r"_smth-(\d+)mm", f) else -1) # then by increasing smoothing kernel

        # Output PDF path
        base_name = re.sub(r"\.png$", "", base_name)  # Remove the .png extension using re
        output_pdf = os.path.join(output, f"{base_name}.pdf")

        # Open images and save them directly to a PDF
        images = []
        for file in group_files:
            file_path = os.path.join(fig_dir, file)
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")  # Convert to RGB if necessary
            images.append(img)

        # Save all images to a single PDF
        if images:
            images[0].save(output_pdf, save_all=True, append_images=images[1:])
            if verbose:
                print(f"\tPDF created: {output_pdf}")

def pairedItems(item, dictlist, mtch=['grp', 'lbl']):
    """
    Given a dict item and a list of dicts, return a list of indices in dictlist
    where all mtch keys match the values in item.

    Args:
        item (dict): The reference dictionary.
        dictlist (list): List of dictionaries to search.
        mtch (list): List of keys to match on.

    Returns:
        list: List of indices in dictlist where all mtch keys match item.
    """
    matched_indices = []
    for i, d in enumerate(dictlist):
        if all(item.get(key) == d.get(key) for key in mtch):
            matched_indices.append(i)
    return matched_indices

def h_bar(item, df_name, metric, ipsiTo=None, title=False):
    """
    Plot horizontal bar chart showing mean statistic by parcellated region.

    Input:
        item: dictionary item containing 'grp', 'label', 'study', and DataFrame with parcellated regions.
        df_name: name of the DataFrame key in item to use for plotting.
        ipsiTo: hemisphere to use for ipsi/contra labeling ('L' or 'R'). If None, uses both hemispheres.
        title: if True, adds a title to the plot.

    Output:
        figure object
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Use correct DataFrame
    df = item[df_name].loc[[metric]]
    
    if df is None or df.empty:
        print(f"[h_bar] WARNING: No data found for {df_name} in item {item['label']}. Skipping horizontal bar chart.")
        return

    lbl = item['label']
    study = item.get('study', "comp")

    # Prepare data
    df_long = df.melt(var_name='Parcel', value_name='MeanZ')
    df_long['Hemisphere'] = df_long['Parcel'].apply(lambda x: 'L' if str(x).endswith('_left') else ('R' if str(x).endswith('_right') else ''))
    df_long['Lobe'] = df_long['Parcel'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else str(x))

    # Split into L and R
    df_L = df_long[df_long['Hemisphere'] == 'L'].copy()
    df_R = df_long[df_long['Hemisphere'] == 'R'].copy()
    df_L['Lobe_clean'] = df_L['Parcel'].str.replace('_left$', '', regex=True)
    df_R['Lobe_clean'] = df_R['Parcel'].str.replace('_right$', '', regex=True)
    df_merged = pd.merge(df_L, df_R, on='Lobe_clean', suffixes=('_L', '_R'))
    df_merged['absmax'] = df_merged[['MeanZ_L', 'MeanZ_R']].abs().max(axis=1)
    df_merged = df_merged.sort_values('absmax', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_merged))

    # Find the largest absolute value for xlim and color scaling
    max_abs = np.nanmax(df_merged[['MeanZ_L', 'MeanZ_R']].abs().values)
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 1

    norm = plt.Normalize(-max_abs, max_abs)
    cmap = plt.cm.seismic
    colors_L = ['#fca9a9' if v > 0 else '#a9c6fc' for v in df_merged['MeanZ_L']]
    colors_R = ['#fca9a9' if v > 0 else '#a9c6fc' for v in df_merged['MeanZ_R']]

    fig, ax = plt.subplots(figsize=(13, max(6, int(len(df_merged)/2.5))))

    # Plot L as negative, R as positive
    bars_L = ax.barh(
        y=y_pos,
        width=-abs(df_merged['MeanZ_L']),
        color=colors_L,
        edgecolor='k',
        align='center',
        label='Left'
    )
    bars_R = ax.barh(
        y=y_pos,
        width=abs(df_merged['MeanZ_R']),
        color=colors_R,
        edgecolor='k',
        align='center',
        label='Right'
    )

    # Annotate bars: label inside, value outside with color
    for bar, label, val in zip(bars_L, df_merged['Lobe_clean'], df_merged['MeanZ_L']):
        xpos_label = bar.get_x() + bar.get_width() / 2
        ax.text(xpos_label, bar.get_y() + bar.get_height()/2, f"L {label.upper()}",
                va='center', ha='center', color='black', fontsize=14)
        xpos_val = bar.get_x() + bar.get_width() - 0.02 * max_abs
        ax.text(xpos_val, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                va='center', ha='right', color='black', fontsize=14, fontweight='bold')

    for bar, label, val in zip(bars_R, df_merged['Lobe_clean'], df_merged['MeanZ_R']):
        xpos_label = bar.get_x() + bar.get_width() / 2
        ax.text(xpos_label, bar.get_y() + bar.get_height()/2, f"R {label.upper()}",
                va='center', ha='center', color='black', fontsize=14)
        xpos_val = bar.get_x() + bar.get_width() + 0.02 * max_abs
        ax.text(xpos_val, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                va='center', ha='left', color='black', fontsize=14, fontweight='bold')
    
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    ax.axvline(0, color='k', linewidth=1)
    # Set x-axis to -max_abs to +max_abs
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)
    xticks = np.linspace(-max_abs, max_abs, num=5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.1f}" if x != 0 else "0" for x in xticks], fontsize=16)
    # Label axis as absolute value
    if study != "comp":
        ax.set_xlabel('mean z-score', fontsize=16)
    elif study == "comp":
        ax.set_xlabel('z dif (7T-3T/3T)', fontsize=16)
    if title:
        ax.set_title(f'$\mathbf{{[{study}]\, {lbl}\ mean\ z}}$', fontsize=20)

    # Annotate hemispheres
    mid_y = len(y_pos) // 2
    fig.tight_layout()
    ax.annotate("L", xy=(ax.get_xlim()[0], mid_y), xytext=(ax.get_xlim()[0] - 0.05*max_abs, mid_y),
                 ha='left', va='center', fontsize=22, fontweight='bold', color='black')
    ax.annotate("R", xy=(ax.get_xlim()[1], mid_y), xytext=(ax.get_xlim()[1] + 0.05*max_abs, mid_y),
                 ha='right', va='center', fontsize=22, fontweight='bold', color='black')

    ax.legend(loc='lower right')
    # Do not show the plot, just return the figure object
    return fig

def visMean(dl, df_name='comps_df_d_ic', df_metric=None, dl_indices=None, ipsiTo="L", title=None, save_name=None, save_path=None):
    """
    Create brain figures from a list of dictionary items with vertex-wise dataframes.
    Input:
        dl: list of dictionary items with keys 'study', 'grp', 'label', 'feature', 'region' {df_name}
        df_name: name of the dataframe key to use for visualization (default is 'df_z_mean')
        indices: list of indices to visualize. If None, visualize all items in the list.
        ipsiTo: hemisphere to use for ipsilateral visualization ('L' or 'R').
    """
    import pandas as pd
    from IPython.display import display

    for i, item in enumerate(dl):
        region = item.get('region', 'ERROR')
        feature = item.get('feature', 'ERROR')
        label = item.get('label', 'ERROR')
        surface = item.get('surf', 'ERROR')
        smth = item.get('smth', 'ERROR')
        
        print(f"[visMean] [{i}] ([{item.get('studies','NA')}] {region} {feature} {label} {surface} {smth}mm)")
        
        if dl_indices is not None and i not in dl_indices:
            continue
        if df_name not in item:
            print(f"[visMean] WARNING: {df_name} not found in item {i}. Skipping.")
            continue
        
        df = item[df_name]
        if df_metric is not None:
            df = df.loc[[df_metric]]

        #print(f"\tdf of interest: {df.shape}")

        # remove SES or ID columns if they exist
        if isinstance(df, pd.Series):
            df = df.to_frame().T  # Convert Series to single-row DataFrame
        df = df.drop(columns=[col for col in df.columns if col in ['SES', 'ID', 'MICS_ID', 'PNI_ID']], errors='ignore')
        #print(f"\tdf after removing ID/SES: {df.shape}")
        
        # surface from size of df
        if ipsiTo is not None:
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

        lh = df[lh_cols]
        rh = df[rh_cols]
        #print(f"\tL: {lh.shape}, R: {rh.shape}")
        fig = showBrain(lh, rh, surface, ipsiTo=ipsiTo, save_name=save_name, save_pth=save_path, title=title, min=-2, max=2, inflated=True)

        return fig

def itmToVisual(item, df_name='comps_df_d_ic', metric = 'd_df_z_TLE_ic_ipsiTo-L_Δd',
                region = None, feature = None, 
                ipsiTo="L", 
                save_name=None, save_pth=None, title=None, 
                min_val = None, max_val=None):
    """
    Convert a dictionary item to format to visualize.
    
    Input:
        dict: dictionary item with keys 'study', 'grp', 'label', 'feature', 'df_z_mean'
        df_name: name of the dataframe key to use for visualization (default is 'df_z_mean')
        ipsiTo: only define if TLE_ic: hemisphere to use for ipsilateral visualization ('L' or 'R').
        save_name: name to save the figure (default is None)
        save_pth: path to save the figure (default is None)
        title: title for the plot (default is None)
        max_val: maximum value for the color scale (default is 2)
    Output:
        fig: figure object for visualization
    """
    import pandas as pd

    region = item.get('region', region)
    surface = item.get('surf', 'fsLR-5k')
    try:
        # If metric is numeric (int or float or string of digits), convert to int
        if isinstance(metric, (int, float)) or (isinstance(metric, str) and metric.isdigit()):
            df = item[df_name].loc[[int(metric)]]
        else:
            df = item[df_name].loc[[metric]]
    except KeyError:
        print(f"[itmToVisual] WARNING: {metric} not found in item. Skipping.")
        return None
    #print(f"\tdf of interest: {df.shape}")

    # remove SES or ID columns if they exist
    if isinstance(df, pd.Series):
        df = df.to_frame().T  # Convert Series to single-row DataFrame
    df = df.drop(columns=[col for col in df.columns if col in ['SES', 'ID', 'MICS_ID', 'PNI_ID']], errors='ignore')
    #print(f"\tdf after removing ID/SES: {df.shape}")
    
    if ipsiTo is not None:
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

    lh = df[lh_cols]
    rh = df[rh_cols]
    #print(f"\tL: {lh.shape}, R: {rh.shape}")
    # ensure all numeric data
    lh = lh.apply(pd.to_numeric, errors='coerce')
    rh = rh.apply(pd.to_numeric, errors='coerce')
    
    title = title or f"{item.get('study', '3T-7T comp')} {item['label']}"
    
    if max_val is None:
        # Calculate min and max from both lh and rh maps
        max_val = max(lh.max().max(), rh.max().max())
    
    if min_val is None:
        min_val = min(lh.min().min(), rh.min().min())

    if feature is None:
        feature = item.get('feature', '')
    if region is None:
        region = item.get('region', 'ctx') # asign default value to 'ctx'

    fig = showBrain(lh, rh, region,
                    surface, feature_lbl = feature,
                    ipsiTo=ipsiTo, title=title, inflated=True,
                    save_name=save_name, save_pth=save_pth,
                    min = min_val, max = max_val
                    )

    return fig

def showBrain(lh, rh, region = "ctx",
               surface='fsLR-5k', feature_lbl=None, 
               ipsiTo=None, title=None, 
               min=-2.5, max=2.5, inflated=True, 
               save_name=None, save_pth=None, cmap="seismic"
               ):
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

    if region == 'ctx' or region == 'cortex':
        micapipe=os.popen("echo $MICAPIPE").read()[:-1]
        if micapipe == "":
            micapipe = "/data_/mica1/01_programs/micapipe-v0.2.0"
            print(f"[showBrains] WARNING: MICAPIPE environment variable not set. Using hard-coded path {micapipe}")
    elif region == 'hipp' or region == 'hippocampus':
        hipp_surfaces = "/host/verges/tank/data/daniel/3T7T/z/code/analyses/resources/"

    # set wd to save_pth
    if save_pth is not None:
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        os.chdir(save_pth)

    if region == "ctx" or region == "cortex":
        if surface == 'fsLR-5k':
            if inflated == True:
                # Load fsLR 5k inflated
                surf_lh = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
                surf_rh = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
            else:
                # Load fsLR 5k
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
        
    elif region == "hipp" or region == "hippocampus": # only have templates for midthickness
        if surface == '0p5mm': # use hippunfold toolbox
            
            surf_lh = read_surface(hipp_surfaces + 'tpl-avg_space-canonical_den-0p5mm_label-hipp_midthickness_L.surf.gii', itype='gii')
            surf_rh = read_surface(hipp_surfaces + 'tpl-avg_space-canonical_den-0p5mm_label-hipp_midthickness_R.surf.gii', itype='gii')

            # unfolded
            surf_lh_unf = read_surface(hipp_surfaces + 'tpl-avg_space-unfold_den-0p5mm_label-hipp_midthickness.surf.gii', itype='gii')
            surf_rh_unf = read_surface(hipp_surfaces + 'tpl-avg_space-unfold_den-0p5mm_label-hipp_midthickness.surf.gii', itype='gii')
        
        elif surface == '2mm':
            raise NotImplementedError("2mm hippocampal surfaces not implemented yet. Missing templates.")
            surf_lh = read_surface(hipp_surfaces + 'tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii', itype='gii')
            surf_rh = read_surface(hipp_surfaces + 'tpl-avg_space-canonical_den-2mm_label-hipp_midthickness.surf.gii', itype='gii')

            # unfolded
            surf_lh_unf = read_surface(hipp_surfaces + 'tpl-avg_space-unfold_den-2mm_label-hipp_midthickness.surf.gii', itype='gii')
            surf_rh_unf = read_surface(hipp_surfaces + 'tpl-avg_space-unfold_den-2mm_label-hipp_midthickness.surf.gii', itype='gii')
        
        else:
            raise ValueError(f"Surface {surface} not recognized. Use '0p5mm' or '2mm'.")
    else:
        raise ValueError(f"Region {region} not recognized. Use 'ctx' or 'hipp'.")

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
        fig = plot_hemispheres(
            surf_lh, surf_rh, array_name=data, 
            size=(900, 250), color_bar='bottom', zoom=1.25, embed_nb=True, interactive=False, share='both',
            nan_color=(0, 0, 0, 1), color_range=(min,max), cmap=cmap, transparent_bg=False, 
            #, label_text = lbl_text
        )

    return fig

def vis_item(item, metric, ipsiTo=None, save_pth=None):
    """ 
    Visualize outputs of an item.
        [1] hippocampal map - folded and unfolded < TO COME >
        [2] mean z-score (or z-score difference if item is 3T-7T comparison)
        [3] bar graph: mean z-score by lobe

    Input:
        item: dictionary item with keys 'study', 'label', and relevant dataframes
        metric: name of metric to visualize (this metric name should correspond to index name of dfs) 
        ipsiTo: if ipsi/contra data, default is None
        save_pth: path to save the figure, if None, will not save
    Output:
        figure object with all three figures
    """
    import datetime
    from IPython.display import Image
    from PIL import Image as PILImage
    import io
    import matplotlib.pyplot as plt
    import numpy as np

    if metric == 'dD':
        metric_lbl = 'ΔD (7T - 3T)'
    elif metric == 'dD_by3T':
        metric_lbl = 'ΔD [(7T - 3T) / 3T]'
    elif metric == 'dD_by7T':
        metric_lbl = 'ΔD [(7T - 3T) / 7T]'
    else:
        metric_lbl = metric

    if item.get('study', 'comp') == "comp":
        title = f"7T-3T comparison: ({item['label']})\n{metric_lbl}"
        df_crtx_plt = "comps_crtx"
        df_barplot = "comps_crtx_glsr_Lobe_hemi"
        df_hipp = "comps_hipp"
        if ipsiTo is not None:
            title += f"\n(ipsi to {ipsiTo} hemi)"

        if save_pth is not None: 
            save_name = f"{save_pth}/crtxParc_3T7T_{item['grp']}_{item['label']}_zDif"
    else:
        title = f"{item['study']} ({item['label']})\n{metric_lbl}"
        df_crtx_plt = "df_d_crtx_ic"
        df_barplot = "df_d_crtx_ic_glsr_Lobe_hemi"
        df_hipp = "df_d_hipp_ic"
        if ipsiTo is not None:
            title += f"\n(ipsi to {ipsiTo} hemi)"
        if save_pth is not None: 
            save_name = f"{save_pth}/crtxParc_{item['study']}_{item['label']}"

    # hippocampus visual -- TO COME
    # 

    # Cortex visual
    crtx_img = itmToVisual(item, df_name=df_crtx_plt, metric=metric, feature = metric_lbl, ipsiTo=ipsiTo)
    #print(type(crtx_img))
    img_bytes = crtx_img.data  # This is the raw PNG bytes
    img = PILImage.open(io.BytesIO(img_bytes))
    img_arr = np.array(img)

    # --- Barplot as image ---
    barplot_fig = h_bar(item, df_name=df_barplot, metric=metric, ipsiTo=ipsiTo)
    buf = io.BytesIO()
    barplot_fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    bar_img = PILImage.open(buf)
    bar_img_arr = np.array(bar_img)

    # Now, create a new matplotlib figure and add both the image and your other figure
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Show the image in the first subplot
    axs[0].imshow(img_arr)
    axs[0].axis('off')

    # Show the matplotlib Figure as an image in the second subplot
    axs[1].imshow(bar_img_arr)
    axs[1].axis('off')

    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_pth is not None:
        date = datetime.datetime.now().strftime("%d%b%Y-%H%M")
        fig.savefig(f"{save_name}_{date}.png", dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {save_name}_{date}.png")

    return fig



def plot_ridgeplot(matrix, matrix_df=None, Cmap='rocket', Range=(0.5, 2), Xlab="Map value", save_path=None, title=None, Vline=None, VlineCol='darkred'):
    """
    Parameters:
    - matrix: numpy array or pandas DataFrame
        Each row is an individual, each column is a vertex.
    - matrix_df: pandas DataFrame, optional
        DataFrame with info about individuals (rows).
    - Cmap: str, optional
        Colormap for ridgeplot.
    - Range: tuple, optional
        x-axis range.
    - Xlab: str, optional
        x-axis label.
    - save_path: str, optional
        Path to save plot.
    - title: str, optional
        Plot title.
    - Vline: float, optional
        Value for vertical line.
    - VlineCol: str, optional
        Color for vertical line.
    Returns:
    None

    Plots a ridgeplot: each row is an individual, each column is a vertex.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib as mpl    
    import numpy as np
    import pandas as pd
    # If input is DataFrame, convert to numpy array
    if hasattr(matrix, "values"):
        matrix_np = matrix.values
    else:
        matrix_np = matrix

    n_individuals = matrix_np.shape[0]
    n_vertices = matrix_np.shape[1]

    if matrix_df is None:
        matrix_df = pd.DataFrame({'id': [f'{i+1}' for i in range(n_individuals)]})
        print_labels = False
    else:
        print_labels = True

    mean_row_values = np.mean(matrix_np, axis=1)
    sorted_indices = np.argsort(mean_row_values)
    sorted_matrix = matrix_np[sorted_indices]
    sorted_id_x = matrix_df['id'].values[sorted_indices]

    # Flatten matrix for plotting
    ai = sorted_matrix.flatten()
    subject = np.repeat(np.arange(1, n_individuals+1), n_vertices)
    id_x = np.repeat(sorted_id_x, n_vertices)

    d = {'feature': ai,
         'subject': subject,
         'id_x': id_x
        }
    df = pd.DataFrame(d)

    f, axs = plt.subplots(nrows=n_individuals, figsize=(3.468504*2.5, 2.220472*3.5), sharex=True, sharey=True)
    f.set_facecolor('none')

    x = np.linspace(Range[0], Range[1], 100)

    for i, ax in enumerate(axs, 1):
        sns.kdeplot(df[df["subject"]==i]['feature'],
                    fill=True,
                    color="w",
                    alpha=0.25,
                    linewidth=1.5,
                    legend=False,
                    ax=ax)
        
        ax.set_xlim(Range[0], Range[1])
        
        im = ax.imshow(np.vstack([x, x]),
                       cmap=Cmap,
                       aspect="auto",
                       extent=[*ax.get_xlim(), *ax.get_ylim()]
                      )
        ax.collections
        path = ax.collections[0].get_paths()[0]
        patch = mpl.patches.PathPatch(path, transform=ax.transData)
        im.set_clip_path(patch)
           
        ax.spines[['left','right','bottom','top']].set_visible(False)
        
        if i != n_individuals:
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel(Xlab)
            
        ax.set_yticks([])
        ax.set_ylabel("")
        
        ax.axhline(0, color="black")

        ax.set_facecolor("none")

    for i, ax in enumerate(axs):
        if i == n_individuals - 1:
            ax.set_xticks([Range[0], Range[1]])
        else:
            ax.set_xticks([])
        if print_labels:
            ax.text(0.05, 0.01, sorted_id_x[i], transform=ax.transAxes, fontsize=10, color='black', ha='left', va='bottom')

    if Vline is not None:
        for ax in axs:
            ax.axvline(x=Vline, linestyle='dashed', color=VlineCol)

    plt.subplots_adjust(hspace=-0.8)
    
    if title:
        plt.suptitle(title, y=0.99, fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
