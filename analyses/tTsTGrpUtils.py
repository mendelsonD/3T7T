################### OVERALL UTILS ####################################
def loadPickle(pth, verbose=True, dlPrint=False):
    """
    Load a pickle file from the specified path.
    
    Parameters:
    - pth: str
        Path to the pickle file.
    
    Returns:
    - obj: object
        The object loaded from the pickle file.
    """
    import pickle    
    import os
    with open(pth, "rb") as f:
            obj = pickle.load(f)

    file_size = os.path.getsize(pth) / (1024 * 1024)
    if verbose:
        print(f"[loadPickle] Loaded object ({file_size:0.1f} MB): {pth}")
    if dlPrint:
        print('-'*100)
        print_dict(obj)
        print('='*100)
    return obj

def savePickle(obj, root, name, timeStamp = True, append=None, test=False, verbose = True, rtn_txt = False):
    """
    Save an object to a pickle file.
    
    Parameters:
    - obj: object
        The object to be saved.
    - root: str
        Directory where the pickle file will be saved.
    - name: str
        Name of the pickle file (without extension).
    - timeStamp: bool
        If True, appends a timestamp to the filename.
    - append: str or None
        If provided, appends this string to the filename (before timeStamp)
    - test: bool
        If True, appends 'TEST_' to the filename.
    - verbose: bool
        If True, prints status messages.
    - rtn_txt: bool
        If True, returns the path and a status message.

    Output:
        Saves file.
    Returns:
        - pth: str
            Path to the saved pickle file.
    """
    import pickle
    import os
    import datetime
    
    # Ensure root directory exists
    if not os.path.exists(root):
        os.makedirs(root)
        if verbose:
            print(f"\t[savePickle] Created directory: {root}")

    if test: 
        name = f"TEST_{name}"
    if append is not None:
        name = f"{name}_{append}"
    if timeStamp:
        name = f"{name}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"

    pth = os.path.join(root, f"{name}.pkl")
    
    with open(pth, "wb") as f:
        pickle.dump(obj, f)
    
    file_size = os.path.getsize(pth) / (1024 * 1024)  # size in MB
    print_txt = f"[pkl_save] Saved object ({file_size:0.1f} MB) to {pth}"

    if verbose:
        print(print_txt)
    
    if rtn_txt:
        return pth, print_txt
    else:
        return pth

def _get_file_logger(name, log_file_path):
    """
    Return a logger configured to write to log_file_path.
    Handlers are only added once per logger instance to avoid duplicate logs.
    """
    import logging, os

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Ensure directory exists
    log_dir = os.path.dirname(log_file_path) or '.'
    os.makedirs(log_dir, exist_ok=True)
    abs_path = os.path.abspath(log_file_path)

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if getattr(h, "baseFilename", None) == abs_path:
                    logger.propagate = False
                    logger._tTsT_configured = abs_path
                    return logger
            except Exception:
                continue

    # Remove old handlers (close them)
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
            h.close()
        except Exception:
            pass

    # Info handler (general messages)
    info_handler = logging.FileHandler(log_file_path)
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter("%(message)s")
    info_handler.setFormatter(info_formatter)

    # Warning handler (different formatter with timestamp)
    warn_handler = logging.FileHandler(log_file_path)
    warn_handler.setLevel(logging.WARNING)
    warn_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%d %b %Y-%H:%M:%S")
    warn_handler.setFormatter(warn_formatter)

    logger.addHandler(info_handler)
    logger.addHandler(warn_handler)

    # Prevent propagation to root logger to avoid duplicate console output
    logger.propagate = False

    # mark as configured
    logger._tTsT_configured = True

    return logger

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

def get_huDice(root, sub, ses, desc = "unetf3d", rtn_ERR = False):
    """
    Get dice score for hippocampal surface segmeetnation. 
    One value per hemi. Values >0.7 are very likely to be good segmentations

    Input:
        root: root directory to hippunfold output
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)

        desc: 'desc' field of file name. Default: "unetf3d"
        rtn_ERR: if should return error statement as string. Default: False

    Output:
        dice: list of dice scores for left and right hemispheres [L, R]
    """
    import pandas as pd
    
    pth_L = f"{root}/sub-{sub}/ses-{ses}/qc/sub-{sub}_ses-{ses}_hemi-L_desc-{desc}_dice.tsv"
    pth_R = f"{root}/sub-{sub}/ses-{ses}/qc/sub-{sub}_ses-{ses}_hemi-R_desc-{desc}_dice.tsv"
    
    try:
        dL = float(pd.read_csv(pth_L, sep="\t").columns[0])
        dR = float(pd.read_csv(pth_R, sep="\t").columns[0])
    except Exception as e:
        ERR_txt = f"\t[get_huDice] WARNING: Could not read dice score files. Setting to None.\nError: {e}"
        if rtn_ERR:
            return ERR_txt, None
        else:
            print(ERR_txt)
            dL, dR = None, None
    
    return dL, dR

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
            lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-{space}_{surf}_label-hipp_thickness.shape.gii"
            rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-{space}_{surf}_label-hipp_thickness.shape.gii"
        else:
            lh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-L_space-{space}_{surf}_label-hipp_{lbl}.surf.gii"
            rh = f"{root}/sub-{sub}/ses-{ses}/surf/sub-{sub}_ses-{ses}_hemi-R_space-{space}_{surf}_label-hipp_{lbl}.surf.gii"
    else:
        raise ValueError("[get_surf_path] Unknown root directory. Choose from 'micapipe' or 'hippunfold'.")

    return lh, rh 

def get_mapVol_pth(root, sub, ses, study, feature, raw=False, space="nativepro"):
    """
    Get path to map volume.
    
    input:
        root: root directory to micapipe output
        sub: subject ID (no `sub-` prefix)
        ses: session ID (with leading zero if applicable; no `ses-` prefix)
        study: study name (e.g., "3T", "7T")
        feature: type of map to retrieve (e.g., "T1map", "FLAIR", "ADC", "FA")
            File naming pattern:
                T1map: map-T1
                FLAIR: map-flair
                ADC: DTI_map-ADC
                FA: DTI_map-FA
        raw: whether to get raw volume. If false, then get micapipe generated map
        space: space of the map (default is "nativepro")

    output:
        Path to the map file in nativepro space.
    """
    feature = feature.upper()
    
    if raw:
        if feature == "T1MAP":           
            if study == "7T":
                img = "acq-05mm_T1map"
            elif study == "3T":
                img = 'T1map'
            else:
                img = 'T1map'
        elif feature == "FLAIR": 
            img = "FLAIR"
        elif feature == "THICKNESS" or feature == "T1W": 
            if study == "7T":
                img = "UNIT1"
            elif study == "3T":
                img = 'T1w'
            else:
                img = 'T1w'
        if feature == "ADC" or feature == "FA":
            print("[get_mapVol_pth] WARNING: Diffusion raw volumes not yet implemented. Skipping.")
            return None
        pth = f"{root}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_{img}.nii.gz"
    else:
        if feature == 'T1W':
            pth = f"{root}/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_space-{space}_T1w.nii.gz"
        else:
            if feature == "T1MAP": mtrc = "map-T1map"
            elif feature == "FLAIR": mtrc = "map-flair"
            elif feature == "ADC": mtrc = "model-DTI_map-ADC"
            elif feature == "FA": mtrc = "model-DTI_map-FA"
            else:
                raise ValueError(f"[get_mapVol_pth] Invalid metric: {feature}. Choose from 'T1map', 'FLAIR', 'ADC', or 'FA'.")
            pth = f"{root}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_space-{space}_{mtrc}.nii.gz"

    return pth


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

def get_RawVolumeNames(features):
    """
    from list of features, determine the raw volumes to check for QC
    NOTE. To implement diffusion scans

    Input:
        features (list): list of features to be included
    
    Output:
        vol_names (list): list of raw volume names to check for QC
    """
    vol_names = []
    if 'thickness' in features:
        vol_names = ['T1w']
    if 'T1map' in features:
        vol_names.append('T1map')
    if 'flair' in features:
        vol_names.append('flair')
    if 'FA' in features or 'ADC' in features:
        print("[get_RawVolumeNames] WARNING: Diffusion scans not yet implemented in get_RawVolumeNames function. Not including in volume names.")

    return list(set(vol_names)) # return unique names only

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

def appendSeries(series, df, match_on):
            """
            Append a series to a df matching on provided keys.

            Input:
                series: pd.Series to append
                df: pd.DataFrame to append to
                
                One of either below must be defined:
                match_on: list of columns to match on
                idx_seriesCol: optional, index of the series to match on if not the series index
            """
            import pandas as pd

            df_s = pd.DataFrame([series])
            df_out = df.merge(df_s, on=match_on, how='outer', suffixes=('', '_from_df_s'))

            for col in df_s.columns: # combines values from repeated columns
                if col not in match_on:  # Skip columns used for matching
                    if col in df_out.columns and f"{col}_from_df_s" in df_out.columns:
                        # Combine values from df_s into the original column
                        df_out[col] = df_out[f"{col}_from_df_s"].combine_first(df_out[col])
                        # Drop the temporary column
                        df_out.drop(columns=[f"{col}_from_df_s"], inplace=True)
            
            return df_out

def addToSeries(col, val, series):
    """
    Add a value to a series at a specified column.
    If the col name already exists, then replace the value in this key. 
    Col and val can be lists or strings.

    Input:
        col: column name to add value to
        val: value to add
        series: pd.Series to add value to

    Output:
        series: updated pd.Series with new value added
    """
    import pandas as pd
    
    def addSingleCol(c,v,series):
        if c in series.index:
            series[c] = v # overrides previous value
        else:
            series[c] = v
        return series
    #print(f"{type(col)}, {type(val)}: {col}, {val}")
    if type(col) != list:
        col = [col]
    if type(val) != list:
        val = [val]
    if len(col) != len(val):
        raise ValueError("[addToSeries] col and val must be of same length if both are lists.")
    #print(f"{type(col)}, {type(val)}: {col}, {val}")

    for c, v in zip(col, val):
        series = addSingleCol(c,v,series)
    
    return series
        

def idToMap(df_demo, studies, dict_demo, specs, 
            save=True, save_pth=None, save_name="02a_mapPths", test=False, test_frac = 0.1,
            verbose=False):
    """
    TODO. SAVE df intermittently (robust against interruptions). Save with time stamp. 
        Then, at the end save the final version, find all files matching the naming pattern with times greater than the start time and less than current time and delete these intermediate files.
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
    import datetime
    import logging
    

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.txt")
    print(f"[idToMap] Saving log to: {log_file_path}")
    
    # Configure module logger (handlers added per-file by _get_file_logger)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info("Log started for idToMap function.")

    out_pth = ""
    match_on = ['UID', 'study', 'SES']

    try:
        logger.info(f"[idToMap] Saving log to: {log_file_path}")
        start_time = datetime.datetime.now()
        print(f"\tStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}.")
        logger.info(f"Finding/computing smoothed maps for provided surface, label, feature and smoothing combinations. Adding paths to dataframe.\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\tParameters:\tspecs:{specs}")
        logger.info(f"\tNumber of rows to process: {df_demo.shape[0]}")
        
        def ctx_maps(out_dir, study, df, idx, sub, ses, uid, surf, ft, smth, lbl, verbose=False):
            """
            Get or compute cortical smoothed maps for a given subject and session, surface, label, feature, and smoothing kernel size.
            """
            import os
            import pandas as pd

            logger.info(f"\t{ft}, {lbl}, {surf}, smth-{smth}mm")
            root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
            study_code = study['study']

            skip_L, skip_R = False, False
           
            pths_series = pd.Series({'UID': uid,
                                    'study': study_code,
                                    'SES': ses}, dtype="object") # Series object to hold paths. Prevents excessive editing of df which leads to defragmentation.

            # Ø. Declare output names and final file of interest paths
            if ft == "thickness":
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{ft}"
            else:
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{lbl}_{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{lbl}_{ft}"
                                            
            pth_map_unsmth_L = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_L_filename}.func.gii"
            pth_map_unsmth_R = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_R_filename}.func.gii"
            
            pth_map_smth_L  = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_ctx_{out_pth_L_filename}_smth-{smth}mm.func.gii",
            )

            pth_map_smth_R = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_ctx_{out_pth_R_filename}_smth-{smth}mm.func.gii",
            )
            
            # Determine appropriate col name
            col_smth_L = os.path.basename(pth_map_smth_L).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            col_smth_R = os.path.basename(pth_map_smth_R).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            
            # Remove smoothing suffix for path to unsmoothed maps
            col_unsmth_L = col_smth_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = col_smth_R.replace(f"_smth-{smth}mm", "_unsmth")

            if chk_pth(pth_map_smth_L) and chk_pth(pth_map_smth_R): # If smoothed map already exists, do not recompute. Simply add path to df.
                if verbose:
                    print(f"\t\tSmoothed maps exists: {pth_map_smth_L}\t{pth_map_smth_R}\n")
                
                pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R], 
                                          val = [pth_map_unsmth_L, pth_map_unsmth_R, pth_map_smth_L, pth_map_smth_R], 
                                          series = pths_series)
                df_out = appendSeries(pths_series, df, match_on=match_on)
                return df_out
            
            elif chk_pth(pth_map_smth_L):
                if verbose:
                    logger.info(f"\t\tSmoothed L map exists :\t\t{pth_map_smth_L}")

                pths_series = pd.concat([pths_series,
                                        pd.Series({col_unsmth_L: pth_map_unsmth_L, 
                                                   col_smth_L: pth_map_smth_L})])
                skip_L = True
            elif chk_pth(pth_map_smth_R):
                if verbose:
                    logger.info(f"\t\tSmoothed R map exists, adding path to df:\t\t{pth_map_smth_R}")

                pths_series = addToSeries(col = [col_unsmth_R, col_smth_R], 
                                         val = [pth_map_unsmth_R, pth_map_smth_R], 
                                         series = pths_series)
                
                skip_R = True
            else: # perform smoothing with steps below
                pass

            # A. Search for unsmoothed map
            if not chk_pth(pth_map_unsmth_L) and not chk_pth(pth_map_unsmth_R): # both unsmoothed maps mising
                
                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    
                    dir_raw = f" {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} "
                    pth_error = "NA: NO RAWDATA"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Unsmoothed maps missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                   
                else: # If raw data exists and unsmoothed maps don't exist, must be processing problem (eg., not yet micapipe processed or failed processing) 
                    
                    dir_surf = os.path.commonpath([pth_map_unsmth_L, pth_map_unsmth_R]) if pth_map_unsmth_L and pth_map_unsmth_R else "" # Find the common directory of both pth_map_unsmth_L and pth_map_unsmth_R
                    pth_error = "NA: MISSING MP PROCESSING (unsmoothed map)"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Unsmoothed maps MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")

                pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                          val = [pth_error, pth_error, pth_error, pth_error],
                                          series = pths_series)
                
                df_out = appendSeries(pths_series, df, match_on=match_on)
                return df_out
            
            elif not chk_pth(pth_map_unsmth_L): # missing L hemi only
                
                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    
                    dir_raw = f" {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    pth_error = "NA: NO RAWDATA"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Hemi-L unsmoothed map due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                
                else: # Must be micapipe problem if raw data exists

                    dir_surf = os.path.basename(pth_map_unsmth_L)
                    pth_error = "NA: MISSING MP PROCESSING (unsmoothed map)"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses}  ({ft}, {lbl}, {surf}): Hemi-L unsmoothed map MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")

                skip_L = True
                pths_series = addToSeries(col = [col_unsmth_L, col_smth_L],
                                          val = [pth_error, pth_error],
                                          series = pths_series)

            elif not chk_pth(pth_map_unsmth_R): # missing R hemi only

                if checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft) == False: # check if raw data problem
                    dir_raw = f" {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} "
                    pth_error = "NA: NO RAWDATA"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Hemi-R unsmoothed map due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe once resolved.\n")
                    
                else: # Must be micapipe problem if raw data exists
                    dir_surf = os.path.basename(pth_map_unsmth_R)
                    pth_error = "NA: MISSING MP PROCESSING (unsmoothed map)"
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses}  ({ft}, {lbl}, {surf}): Hemi-R unsmoothed map MISSING in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_surf} ).\n")

                skip_R = True
                pths_series = addToSeries(col = [col_unsmth_R, col_smth_R],
                            val = [pth_error, pth_error],
                            series = pths_series)
            
            else: # unsmoothed maps exist for both
                if verbose:
                    logger.info(f"\t\tUnsmoothed maps:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
                
                pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R],
                                          val = [pth_map_unsmth_L, pth_map_unsmth_R],
                                          series = pths_series)
            
            # add unsmoothed maps paths to df
            if not skip_L:
                logger.info(f"\t\tAdded L unsmoothed path: {pth_map_unsmth_L}")
                pths_series = addToSeries(col = [col_unsmth_L],
                                            val = [pth_map_unsmth_L],
                                            series = pths_series)
            if not skip_R:
                logger.info(f"\t\tAdded L unsmoothed path: {pth_map_unsmth_L}")
                pths_series = addToSeries(col = [col_unsmth_R],
                                            val = [pth_map_unsmth_R],
                                            series = pths_series)

            # B. Smooth map and save to project directory
            surf_L, surf_R = get_surf_pth( # Get surface .func files
                root=root_mp,
                sub=sub,
                ses=ses,
                surf=surf,
                lbl=lbl
            )

            logger.warning(f"VALUE OF `pths_series` BEFORE CHECKING SURFACES:\n{pths_series}")
            if not chk_pth(surf_L) and not chk_pth(surf_R) and not skip_L and not skip_R: # check that surfaces exist
                dir_surf = os.path.commonpath([surf_L, surf_R]) # Find the common directory of both surf_L and surf_R
                logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface not found. Skipping. Check micapipe processing ( {dir_surf} ). Missing: {surf_L}\t{surf_R}\n")
                surf_error = "NA: MISSING MP PROCESSING (surface)"
                
                pths_series = addToSeries(col = [col_smth_L, col_smth_R],
                                          val = [surf_error, surf_error],
                                          series = pths_series)

                df_out = appendSeries(pths_series, df, match_on=match_on)
                return df_out
            
            elif not chk_pth(surf_L):
                logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface missing for hemi-L. Check micapipe processing ( {os.path.dirname(surf_L)} ). Skipping smoothing for this hemi. Expected: {surf_L}")
                surf_error = "NA: MISSING MP PROCESSING (surface)"    
                skip_L = True
                pths_series = addToSeries(col = [col_unsmth_L, col_smth_L],
                                          val = [surf_error, surf_error],
                                          series = pths_series)
                
            elif not chk_pth(surf_R):
                logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Cortical nativepro surface missing for hemi-R. Check micapipe processing ( {os.path.dirname(surf_R)} ). Skipping smoothing for this hemi. Expected: {surf_R}")
                surf_error = "NA: MISSING MP PROCESSING (surface)"
                skip_R = True
                pths_series = addToSeries(col = [col_unsmth_R, col_smth_R],
                                          val = [surf_error, surf_error],
                                          series = pths_series)
                
            else:
                if verbose:
                    logger.info(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")

            # ii. Smooth
            if not skip_L:
                pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, pth_map_smth_L, kernel=smth, verbose=False)

                if pth_map_smth_L is None:
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Smoothing failed for hemi-L. Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}\n")
                    surf_smooth_error = f"NA: SMOOTHING FAILED. Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}"
                    pths_series = addToSeries(col = [col_smth_L],
                                              val = [surf_smooth_error],
                                              series = pths_series)
                else:
                    pths_series = addToSeries(col = [col_smth_L],
                                              val = [pth_map_smth_L],
                                              series = pths_series)
                    if verbose:
                        logger.info(f"\t\tSmooth map L: {pth_map_smth_L}")
            
            if not skip_R:
                pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_R, pth_map_smth_R, kernel=smth, verbose=False)
                
                if pth_map_smth_R is None:
                    logger.warning(f"\t\t[ctx_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Smoothing failed for hemi-R. Surf: {surf_R}, Unsmoothed: {pth_map_unsmth_R}, kernel: {smth}\n")
                    surf_smooth_error = f"NA: SMOOTHING FAILED. Surf: {surf_R}, Unsmoothed: {pth_map_unsmth_R}, kernel: {smth}"
                    pths_series = addToSeries(col = [col_smth_R],
                                              val = [surf_smooth_error],
                                              series = pths_series)
                else:
                    pths_series = addToSeries(col = [col_smth_R],
                                            val = [pth_map_smth_R],
                                            series = pths_series)
                    if verbose:
                        logger.info(f"\t\tSmooth map R: {pth_map_smth_R}")
            
            df_out = appendSeries(pths_series, df, match_on=match_on)
            return df_out

        def hipp_maps(out_dir, study, df, idx, sub, ses, uid, surf, ft, smth, lbl, verbose=False):
            """
            Get or compute hippocampal smoothed maps for a given subject and session, surface, label, feature, and smoothing kernel size.
            """
            import os 
            import pandas as pd
            
            logger.info(f"\t{ft}, {lbl}, {surf}, smth-{smth}mm")
            root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
            root_hu = f"{study['dir_root']}{study['dir_deriv']}{study['dir_hu']}"
            study_code = study['study']

            skip_L, skip_R = False, False

            pths_series = pd.Series({'UID': uid,
                                    'study': study_code,
                                    'SES': ses}, dtype="object") # Series object to hold paths. Prevents excessive editing of df which leads to defragmentation.

            # Ø. Declare output names and final file of interest paths                 
            if ft == "thickness":
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{ft}"
            else:
                out_pth_L_filename = f"hemi-L_surf-{surf}_label-{lbl}_{ft}"
                out_pth_R_filename = f"hemi-R_surf-{surf}_label-{lbl}_{ft}"

            pth_map_smth_L  = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-{smth}mm.func.gii",
            )

            pth_map_smth_R = os.path.join(
                out_dir,
                f"sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-{smth}mm.func.gii",
            )
            
            col_smth_L = os.path.basename(pth_map_smth_L).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            col_smth_R = os.path.basename(pth_map_smth_R).replace('.func.gii', '').replace(f"sub-{sub}_", '').replace(f"ses-{ses}_", '')
            
            col_unsmth_L = col_smth_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = col_smth_R.replace(f"_smth-{smth}mm", "_unsmth")

            if chk_pth(pth_map_smth_L) and chk_pth(pth_map_smth_R): # If smoothed map already exists, do not recompute. Simply add path to df.
                if verbose:
                    logger.info(f"\t\tSmoothed maps exist, adding paths to df:\t{pth_map_smth_L}\t{pth_map_smth_R}\n")
                
                if ft == "thickness":
                    pth_map_unsmth_L, pth_map_unsmth_R = get_surf_pth(root=root_hu, sub=sub, ses=ses, lbl="thickness", surf=surf, space="T1w")
                else:
                    pth_map_unsmth_L = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-NA.func.gii"
                    pth_map_unsmth_R = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-NA.func.gii"
                
                pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                          val = [pth_map_unsmth_L, pth_map_unsmth_R, pth_map_smth_L, pth_map_smth_R],
                                          series = pths_series)
                
                df_out = appendSeries(pths_series, df, match_on=match_on)
                return df_out
            
            elif chk_pth(pth_map_smth_L):
                if verbose:
                    logger.info(f"\t\tSmoothed L map exists, adding path to df:\t\t{pth_map_smth_L}")
                pths_series = addToSeries(col = [col_unsmth_L, col_smth_L],
                                         val = [pth_map_unsmth_L, pth_map_smth_L],
                                         series = pths_series)
                skip_L = True

            elif chk_pth(pth_map_smth_R):
                if verbose:
                    logger.info(f"\t\tSmoothed R map exists, adding path to df:\t\t{pth_map_smth_R}")
                df.loc[idx, col_smth_R] = pth_map_smth_R
                skip_R = True
                pths_series = addToSeries(col = [col_unsmth_R, col_smth_R],
                                            val = [pth_map_unsmth_R, pth_map_smth_R],
                                            series = pths_series)
            else:
                pass
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
                    dir_raw = f" {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} "
                    surf_error = "NA: NO RAWDATA"
                    logger.warning(f"\t\t[hipp_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe and hippunfold once resolved.\n")

                elif not chk_pth(pth = T1w_pth): # Check T1w from Micapipe outputs
                    dir_t1w = os.path.dirname(T1w_pth)
                    surf_error = "NA: MISSING MP PROCESSING (nativepro T1w)"
                    logger.warning(f"\t\t[hipp_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING Nativepro T1w in MICAPIPE OUTPUTS. Check micapipe outputs ( {dir_t1w} ).\n")                    
                
                else: # hippunfold processing error
                    dir_surf =  os.path.commonpath([surf_L, surf_R]) if surf_L and surf_R else "" # Find the common directory of both surf_L and surf_R
                    surf_error = "NA: MISSING HU PROCESSING (surf)"
                    logger.warning(f"\t\t[hipp_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface MISSING due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs ( {dir_surf} ).\n") # could also check that the dir exists and further specify
                
                pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                          val = [surf_error, surf_error, surf_error, surf_error],
                                          series = pths_series)
                df_out = appendSeries(pths_series, df, match_on=match_on)
                return df_out

            elif not chk_pth(surf_L) and not skip_L: # Must be hippunfold processing error (e.g., segmentation failed). Rawdata of micapipe processing problem would affect both hemis.                 
            
                dir_surf = os.path.dirname(surf_L)
                surf_error = "NA: MISSING HU PROCESSING (surf)"
                logger.warning(f"\t\t[hipp_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): L-hemi hippocampal surface missing due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs ( {dir_surf} ).\n") # could also check that the dir exists and further specify
                pths_series = addToSeries(col = [col_unsmth_L, col_smth_L],
                                          val = [surf_error, surf_error],
                                          series = pths_series)

                if skip_R: 
                    df = appendSeries(pths_series, df, match_on=match_on)
                    return df
                else:
                    skip_L = True # cannot continue to smoothing for this hemi
                    if verbose: print(f"\t\tSurface (R only):\t{surf_R}")
 
            elif not chk_pth(surf_R) and not skip_R:

                dir_surf = os.path.dirname(surf_R)
                surf_error = "NA: MISSING HU PROCESSING (surf)"
                logger.warning(f"\t\t[hipp_maps] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): R-hemi hippocampal surface missing due to HIPPUNFOLD OUTPUTS. Check hippunfold outputs ( {dir_surf} ).\n") # could also check that the dir exists and further specify
                pths_series = addToSeries(col=  [col_unsmth_R, col_smth_R],
                                        vol = [surf_error, surf_error],
                                        series = pths_series)
                
                if skip_L: 
                    df = appendSeries(pths_series, df, match_on=match_on)
                    return df
                else:
                    skip_R = True # cannot continue to smoothing for this hemi
                    if verbose: logger.info(f"\t\tSurface (L only):\t{surf_L}")

            else:
                if verbose:
                    logger.info(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")
            

            # A. Generate unsmoothed maps
            if ft == "thickness": # get path to unsmoothed map from hippunfold outputs
                pth_map_unsmth_L, pth_map_unsmth_R = get_surf_pth(root=root_hu, sub=sub, ses=ses, lbl="thickness", surf=surf, space="T1w")
            
            else: # generate feature map from volume and hippunfold surface
                vol_pth = get_mapVol_pth(root=root_mp, sub=sub, ses=ses, study = study_code, feature=ft) # get the volume path from micapipe outputs

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
                        logger.warning(f"\t\t[hipp_maps] ERROR. {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}) Unsmoothed map could not compute.\n")
                        map_error = "NA: SCRIPT ERROR (unsmoothed feature map computation)"
                        pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                                 val = [map_error, map_error, map_error, map_error],
                                                 series = pths_series)
                        df = appendSeries(pths_series, df, match_on=match_on)
                        return df
                    
                    elif not chk_pth(pth_map_unsmth_L) and not skip_L:
                    
                        logger.warning(f"\t\t[hipp_maps] ERROR. {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}). Could not compute unsmoothed map for L hemi.")
                        map_error = "NA: PROCESSING ERROR (unsmoothed feature map computation)"
                        
                        pths_series = addToSeries(col = [col_unsmth_L, col_smth_L],
                                                 val = [map_error, map_error],
                                                 series = pths_series)
                        if skip_R: 
                            df = appendSeries(pths_series, df, match_on=match_on)
                            return df
                        else:
                            skip_L = True
                            pths_series = pd.Series() # reset series to avoid duplicating entries
                    
                    elif not chk_pth(pth_map_unsmth_R) and not skip_R:
                        logger.warning(f"\t\t[hipp_maps] ERROR. {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}). Could not compute unsmoothed map for R hemi.")
                        map_error = "NA: PROCESSING ERROR (unsmoothed feature map computation)"
                        
                        pths_series = addToSeries(col = [col_unsmth_R, col_smth_R],
                                                 val = [map_error, map_error],
                                                 series = pths_series)
                        if skip_L:
                            df = appendSeries(pths_series, df, match_on=match_on)
                            return df
                        else:
                            skip_R = True
                    else:
                        if verbose:
                            logger.info(f"\t\tUnsmoothed map paths:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
                    
                elif not checkRawPth(root = study['dir_root'] + study['dir_raw'], sub = sub, ses = ses, ft = ft): # Unsmoothed map doesn't exist. Check raw data.
                    dir_raw = f" ( {study['dir_root']}{study['dir_raw']}/sub-{sub}/ses-{ses} ) "
                    data_error = "NA: NO RAWDATA"
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Surface missing due to MISSING RAW DATA. Check raw data ( {dir_raw} ). Process with micapipe and hippunfold once resolved.\n")
                    
                    pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                             val = [data_error, data_error, data_error, data_error],
                                             series = pths_series)
                    df_out = appendSeries(pths_series, df, match_on=match_on)
                    return df_out
                
                else: # Must be due to micapipe error
                    print(f"\t\t[WARNING] {study_code} {sub}-{ses} ({ft}, {lbl}, {surf}): Feature volume not found. Check micapipe processing ( {os.path.dirname(vol_pth)} ). Skipping.\n")
                    proc_error = "NA: MISSING MP PROCESSING (volume ft map)"
                    pths_series = addToSeries(col = [col_unsmth_L, col_unsmth_R, col_smth_L, col_smth_R],
                                             val = [proc_error, proc_error, proc_error, proc_error],
                                             series = pths_series)
                    
                    df_out = appendSeries(pths_series, df, match_on=match_on)
                    return df_out
            
            # Add unsmoothed map column
            # Create col name
            col_smth_L = os.path.basename(pth_map_smth_L).replace('.func.gii', '')
            col_smth_R = os.path.basename(pth_map_smth_R).replace('.func.gii', '')
            col_smth_L = col_smth_L.replace(f"sub-{sub}_ses-{ses}_", '')
            col_smth_R = col_smth_R.replace(f"sub-{sub}_ses-{ses}_", '')
            col_unsmth_L = col_smth_L.replace(f"_smth-{smth}mm", "_unsmth")
            col_unsmth_R = col_smth_R.replace(f"_smth-{smth}mm", "_unsmth")

            logger.info(f"\t\tUnsmoothed map cols:\t{col_unsmth_L}\t{col_unsmth_R}") 
            
            if not skip_L:
                logger.info(f"\t\t L unsmoothed path: {pth_map_unsmth_L}")
                pths_series = addToSeries(col = [col_unsmth_L],
                                         val = [pth_map_unsmth_L],
                                         series = pths_series)
                
            if not skip_R:
                logger.info(f"\t\t R unsmoothed path: {pth_map_unsmth_R}")
                pths_series = addToSeries(col = [col_unsmth_R],
                                        val = [pth_map_unsmth_R],
                                        series = pths_series)

            # B. Smooth map
            if not skip_L:
                pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, pth_map_smth_L, kernel=smth, verbose=False)
                if not chk_pth(pth_map_smth_L):
                    pth_map_smth_L = f"NA: SCRIPT ERROR. Surf: {surf_L}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}"
                
                pths_series = addToSeries(col = [col_smth_L],
                                         val = [pth_map_smth_L],
                                         series = pths_series)
                if verbose:
                    logger.info(f"\t\tSmoothed map L: {pth_map_smth_L}")
            
            if not skip_R:
                pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_L, pth_map_smth_R, kernel=smth, verbose=False)
                if not chk_pth(pth_map_smth_R):
                    pth_map_smth_R = f"NA: SCRIPT ERROR. Surf: {surf_R}, Unsmoothed: {pth_map_unsmth_L}, kernel: {smth}"
                
                pths_series = addToSeries(col = [col_smth_R],
                                         val = [pth_map_smth_R],
                                         series = pths_series)
                if verbose:
                    logger.info(f"\t\tSmoothed map R: {pth_map_smth_R}\n")
            
            df_out = appendSeries(pths_series, df, match_on=match_on)
            return df_out

        if test:
            save_name = f"TEST_{save_name}"
            df_demo = df_demo.sample(frac=test_frac).reset_index(drop=True)
            df_demo = df_demo.dropna(axis=1, how='all') # drop empty columns
            logger.info(f"[TEST MODE] Running on random {test_frac*100}% subset of demographics ({len(df_demo)} rows).")
        
        if verbose:
            logger.info("\t Finding/computing smoothed maps for provided surface, label, feature and smoothing combinations. Adding paths to dataframe...")

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
                    logger.warning(f"[idToMap] WARNING. Unknown study code `{study_code}`. Skipping row.")
                    continue
                else:
                    if verbose:
                        logger.info(f"[idToMap] {idx} of {df_demo.shape[0]-1}...")
                    ID_col = dict_demo['ID_' + study_item['study']]
            else: # if no matches, then take first study
                logger.warning(f"[idToMap] No 'study' column provided. Defaulting to first study in studies list: {studies[0]['study']}.")
                study_item = studies[0]
                ID_col = dict_demo['ID']
            
            sub = row[ID_col]
            ses = row['SES']
            uid = row['UID']

            if idx % 10 == 0 and idx > 0: # progress statement every 10 rows
                percent_complete = 100 * idx / len(df_demo)
                logger.info(f"Progress: {percent_complete:.1f}% of rows completed ({idx}/{len(df_demo)})")

            logger.info(f"\n{study_code} sub-{sub} ses-{ses}")

            out_dir = f"{specs['prjDir_root']}{specs['prjDir_maps']}/sub-{sub}_ses-{ses}" # for saving smoothed maps
            create_dir(out_dir)

            if specs['ctx']:
                print(f"\n\tCORTICAL MAPS [{study_code} sub-{sub} ses-{ses}]...")
                
                for ft in specs['ft_ctx']:
                    for surf in specs['surf_ctx']:
                        for smth in specs['smth_ctx']:
                            for lbl in specs['lbl_ctx']:
                                df_demo = ctx_maps(out_dir=out_dir, study=study_item, df=df_demo, 
                                                   idx=idx, sub=sub, ses=ses, uid=uid,
                                                   surf=surf, ft=ft, smth=smth, lbl=lbl, verbose=verbose)

                                
            if specs['hipp']:
                logger.info(f"\n\tHIPPOCAMPAL MAPS [{study_code} sub-{sub} ses-{ses}]...")
                
                for ft in specs['ft_hipp']:
                    for surf in specs['surf_hipp']:
                        for smth in specs['smth_hipp']:
                            for lbl in specs['lbl_hipp']:
                                df_demo = hipp_maps(out_dir=out_dir, study=study_item, df=df_demo, 
                                                    idx=idx, sub=sub, ses=ses, uid=uid,
                                                    surf=surf, ft=ft, smth=smth, lbl=lbl, verbose=verbose)
                            
            print('-'*100)
        
        if save:
            date = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
            if test:
                save_name = f"TEST_{save_name}"
            
            out_pth = f"{save_pth}/{save_name}_{date}.csv"
            df_demo.to_csv(out_pth, index=False)
            print(f"\n[idToMap] DataFrame with map paths saved to {out_pth}\n")
            logger.info(f"[idToMap] DataFrame with map paths saved to {out_pth}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"[idToMap] An error occurred: {e}. Check log file for details: {log_file_path}")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    mins, secs = divmod(int(duration.total_seconds()), 60) 
    logger.info(f"\n[idToMap] idToMap completed at {end_time.strftime('%d-%b-%Y %H:%M:%S')} (run duration: {mins:02d}:{secs:02d} (mm:ss)).")
    print(f"\n[idToMap] idToMap completed at {end_time.strftime('%d-%b-%Y %H:%M:%S')} (run duration: {mins:02d}:{secs:02d} (mm:ss)).")
    
    return df_demo, out_pth, log_file_path


def get_maps(df, mapCols, col_ID='MICs_ID', col_study = None, verbose=False):
    """
    Create dict item for each, study, feature, label, smoothing pair (including hippocampal)
    Note: multiple groups should be kept in same DF. Seperate groups later on

    Input:
        df: DataFrame with columns for ID, SES, Date, and paths to left and right hemisphere maps.
            NOTE. Asusme path columns end with '_L' and '_R' for left and right hemisphere respectively.
        mapCols: List of column names in df that contain paths to the maps.
        col_ID: Column name for participant ID in the DataFrame. Default is 'UID'.
        col_study: If not none, uses values in this column in the index
    
    Output:
        df_clean: Cleaned DataFrame with only valid ID-SES combinations, and paths to left and right hemisphere maps.
    """
    import nibabel as nib
    import numpy as np
    import pandas as pd
    
    assert col_ID in df.columns, f"[get_maps] df must contain 'ID' column. Cols in df: {df.columns}"
    assert 'SES' in df.columns, f"[get_maps] df must contain 'SES' column. Cols in df: {df.columns}"
    
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
        if col_study is not None and col_study != 'UID':
            df_maps = df[['UID', col_study, col_ID, 'SES', col_L, col_R]]
        else:
            df_maps = df[['UID', col_ID, 'SES', col_L, col_R]]
    else:
        if col_study is not None:
            df_maps = df[[col_study, col_ID, 'SES', col_L, col_R]]
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

    df_maps = setIndex(df=df_maps, col_ID=col_ID, sort=True) # index: <UID_><study_>ID_SES
    
    # Keep only the vertex columns (map data columns)
    vertex_cols = [col for col in df_maps.columns if col.endswith('_L') or col.endswith('_R')]
    df_maps_clean = df_maps[vertex_cols]

    if verbose:
        print(f"\t[get_maps] Maps retrieved. Size: {df_maps_clean.shape}")
    
    return df_maps_clean

def setIndex(df, col_ID='MICs_ID', col_study = None, sort = True):
    """
    Set index of DataFrame to a combination of ID and SES.

    Input:
        df: DataFrame with columns for ID and SES.
        col_ID: Column name for participant ID in the DataFrame. Default is 'MICS_ID'.
        col_study: If not none, uses values in this column in the index

    Output:
        DataFrame with index set to 'UID_ID_SES' (if UID in df) else 'ID_SES'.
    """
    import pandas as pd
    
    assert col_ID in df.columns, f"[setIndex] df must contain 'ID' column. Cols in df: {df.columns}"
    assert 'SES' in df.columns, f"[setIndex] df must contain 'SES' column. Cols in df: {df.columns}"
    if col_study is not None:
        assert col_study in df.columns, f"[setIndex] df must contain 'col_study' column. Cols in df: {df.columns}"
    
    if 'UID' in df.columns:
        if col_study is not None:
            df['UID_STUDY_ID_SES'] = df.apply(lambda row: f"{row['UID']}_{row[col_study]}_{row[col_ID]}_{row['SES']}", axis=1)
            df = df.set_index('UID_STUDY_ID_SES')
        else:
            df['UID_ID_SES'] = df.apply(lambda row: f"{row['UID']}_{row[col_ID]}_{row['SES']}", axis=1)
            df = df.set_index('UID_ID_SES')
    else:
        if col_study is not None:
            df['STUDY_ID_SES'] = df.apply(lambda row: f"{row[col_study]}_{row[col_ID]}_{row['SES']}", axis=1)
            df = df.set_index('STUDY_ID_SES')
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
        save: str
            Path to save cleaned dataframe. If None, do not save.
        col_study: str
           TODO. Confirm role of studies col: Column name for study information. If None, choosing single session from repeated IDs 
            
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
                if verbose: print(f"\t[{study}] {len(repeated_ids_study)} IDs with multiple sessions found. Processing...")
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
    

    if split:
        if verbose:
            print(f"[get_mapCols] {len(cols_L) + len(cols_R)} map columns found: {len(cols_L)} L | {len(cols_R)} R.")
        return cols_L, cols_R
    else:
        if verbose:
            print(f"[get_mapCols] {len(cols_L) + len(cols_R)} map columns found.")
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


def extractMap(df_mapPaths, cols_L, cols_R, studies, demographics, 
               save_name, save_df_pth, log_save_pth,
               append_name = None, region=None, verbose=False, test = False):
    """
    Extract map paths from a dataframe based on specified columns and optional subset string in col name.

    TODO. Add number of surfaces with no data, print at end of extractMap

    Input:
        df_mapPaths: pd.DataFrame 
            map paths kept in column passed in cols.
        cols_L, cols_R: list of str
            column names to extract from the dataframe.
        studies:
            list of dicts  regarding studies in the analysis.
            Each dict should contain:
                'name', 'dir_root', 'study', 'dir_mp','dir_hu'
        demographics: dict  regarding demographics file.
            Required keys:
                'pth', 'ID_7T', 'ID_3T', 'SES', 'date', 'grp'
        save_df_pth: str
            Path to save extracted dataframe.

        append_name: str
            string to append to the save name of the dataframe. If None, do not append anything.
        log_save_pth: str
            path to save log file to
        region: string, optional
            specify cortex or hippocampus. If none, all columns passed will be extracted and region=None will be added to dict item.
        verbose: bool, optional
            If True, print detailed processing information.
        test: bool, optional
            If True, prepend "TEST_" to the save_name and log file name.

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
                'df_maps_unsmth': str
                    path to pickle item holding pd.Dataframe with only the unsmoothed map paths for the specific map (if applicable)
                'df_maps_smth': str
                    path to pickle item holding pd.DataFrame with only the smoothed map paths for the specific map
    """
    import datetime
    import os

    # Prepare log file path
    if log_save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        log_save_pth = os.getcwd()  # Default to current working directory
    if not os.path.exists(log_save_pth):
        os.makedirs(log_save_pth)
    
    log_name = f"04a_extractMap"
    if region is not None:
        log_name = f"{log_name}_{region}"
    if append_name is not None:
        log_name = f"{log_name}_{append_name}"
    if test:
        log_name = f"TEST_{log_name}"
    start = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(log_save_pth, f"{log_name}_log_{start}.txt")
    print(f"\n[extractMap] Saving log to: {log_file_path}")

    # Configure module logger (handlers added per-file by _get_file_logger)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info("Log started for extractMap function.")
    logger.info("Parameters:")
    logger.info(f"  region: {region}")
    logger.info(f"  append_name: {append_name}")
    logger.info(f"  test : {test}")
    logger.info(f"  save_df_pth: {save_df_pth}")
    out_dl = []
    
    try:
    
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
            logger.info(f"\nRegion {region}: {len(cols_L) + len(cols_R)} map columns found (col name pattern: {subset}).")
        else:
            logger.info(f"\n{len(cols_L) + len(cols_R)} map columns found.")

        if cols_L == [] or cols_R == []:
            logger.info("\n[extractMap] WARNING. No map columns found. Skipping.")
            return out_dl
        
        if test:
            import random
            import numpy as np
            idx_len = 2
            idx_rdm = np.random.choice(len(cols_L), size=idx_len, replace=False).tolist()  # randomly choose index
            cols_L = [cols_L[i] for i in idx_rdm]
            cols_R = [cols_R[i] for i in idx_rdm]
            logger.info(f"[extractMap]: TEST MODE. Extract maps from {idx_len} random maps: indices {idx_rdm}.")
        
        counter = 0

        for col_L, col_R in zip(cols_L, cols_R):
            counter += 1
            if counter % 10 == 0:
                print(f"\t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {counter} of {len(cols_L)}...")
            
            assert col_L.replace('hemi-L', '') == col_R.replace('hemi-R', ''), f"Left and right hemisphere columns do not match: {col_L}, {col_R}"
            
            # Find the substring after 'hemi-L' and 'hemi-R' that is common between col_L and col_R
            hemi_L_idx = col_L.find('hemi-L_') + len('hemi-L_')
            commonName = col_L[hemi_L_idx:]

            if verbose:
                logger.info(f"\n\tProcessing {commonName}... (cols: {col_L} {col_R})")
            
            df_tmp = df_mapPaths.dropna(subset=[col_L, col_R]) # remove IDs with missing values in col_L or col_R
            if verbose:
                logger.info(f"\t\t{len(df_mapPaths) - len(df_tmp)} rows removed due to missing values for these maps. [{(len(df_mapPaths))} rows before, {len(df_tmp)} rows remain]")
            
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
                    logger.info(f"\t\t{n_after} unique patients remain after removing {n_removed} IDs due to incomplete study.")
                    logger.info(f"\t\tIDs removed: {sorted(df_tmp_drop['UID'].unique())}")
            if n_after == 0:
                logger.info(f"\t\t[extractMap] WARNING. No participants remain after filtering for complete study data. Skipping this map.")
                continue
            
            for study in studies:
                study_name = study['name']
                study_code = study['study']
                
                col_ID = get_IDCol(study_name, demographics) # determine ID col name for this study
                
                df_tmp_study = df_tmp[df_tmp['study'] == study_code] # filter for rows from this study
                
                if verbose:
                    logger.info(f"\t[{study_code}] {len(df_tmp_study)} rows")

                maps = get_maps(df_tmp_study, mapCols=[col_L, col_R], col_ID = col_ID, col_study='UID', verbose=False)
                if maps.shape[0] == 0:
                    logger.info(f"\t\t[extractMap] WARNING. No maps found for study {study_code}. Skipping this study for this map.")
                    continue
                
                map_name = f"{study_code}_{commonName}"
                if append_name is not None:
                    map_name = f"{map_name}_{append_name}"
                
                if test:
                    map_name = f"TEST_{map_name}"
                
                maps_pth, save_stmt = savePickle(obj = maps, root = save_df_pth, name = map_name, 
                                                 timeStamp=False, append = start, 
                                                 verbose=False, rtn_txt = True)
                logger.info(f"\t{save_stmt}")

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
                    'df_maps': maps_pth,
                })
        
        if verbose:
            logger.info(f"[extractMap] Returning list with {len(out_dl)} dictionary items (region: {region}).")
        
    except Exception as e:
        logger.error(f"An error occurred in extractMap: {e}", exc_info=True)
        print(f"[extractMap] ERROR: An error occurred. Check log file for details.")
    
    logger.info(f"\nCompleted. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[extractMap] Completed. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return out_dl

def extractMap_SES(df_mapPaths, col_sesNum = 'ses_num', col_studyID = 'ID_study', coi = None, region=None, 
                   save = True, save_pth = None, save_name = "sesMaps_dl", 
                   test=False, verbose=False, dlPrint = False):
    """
    Extract map paths from a dataframe based on specified columns and optional subset string in col name.
    Rather than iterating over study, iterate over session number.

    Input:
        df_mapPaths: pd.DataFrame 
            map paths kept in column passed in cols.
            NOTE. Should have following columns:
                col_sesNum: session number (integer) by scan date
                ID_study: appropriate ID for the row considering study
                map paths columns: n cols defined by cols_L, cols_R
        cols_L, cols_R: list of str
            column names to extract from the dataframe.
        demographics: dict  regarding demographics file.
            Required keys:
                'pth'
                'ID_7T'
                'ID_3T'
                'SES'
                'date'
                'grp'
        col_sesNum: str
            column name in df_mapPaths that indicates session number (e.g., '1', '2', etc)
        col_studyID: str
            column name in df_mapPaths that indicates appropriate ID for the row considering study
        coi: list of str, optional
            columns of interest to keep from demographics file. If None, keep all columns.
        region: string, optional
            specify cortex or hippocampus. If none, all columns passed will be extracted and region=None will be added to dict item.

        save: bool
            if True, save the output list of dicts as a .pkl file
        save_pth: str
            path to save the .pkl file to. If None, do not save.
        save_name: str
            name of the .pkl file to save (without extension) 
        verbose: bool
            if True, print progress and warnings
            
        Returns:
        out_dl: list of dicts
            Each dict contains:
                'sesNum': session number
                'region': 'cortex' or 'hippocampus'
                'surf': surface type and resolution (e.g., 'fsLR-5k')
                'label': label type (e.g., 'thickness', 'T1', 'FA', etc)
                'feature': feature type (e.g., 'thickness', 'T1', 'FA', etc)
                'smth': smoothing level (in mm)
                'df_demo': pd.DataFrame with demographics and map paths for the specific map
                'df_maps_unsmth': pd.DataFrame with only the unsmoothed map paths for the specific map (if applicable)
                'df_maps_smth': pd.DataFrame with only the smoothed map paths for the specific map
    """
    import os
    import pandas as pd
    import pickle
    import datetime

    out_dl = []
    
    cols_L, cols_R = get_mapCols(df_mapPaths, verbose=True) # get map path columns
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
        if 'ctx' in col_L.lower():
            region = 'cortex'
        elif 'hipp' in col_L.lower():
            region = 'hippocampus'
        else:
            region = 'unknown'
        print(f"\n\tProcessing {commonName}... (cols: {col_L} {col_R}) | region: {region}")
        
        if verbose:
            print(f"\n\tProcessing {commonName}... (cols: {col_L} {col_R})")
        
        df_tmp = df_mapPaths.dropna(subset=[col_L, col_R]) # remove IDs with missing values in col_L or col_R
        if verbose:
            print(f"\t\t{len(df_mapPaths) - len(df_tmp)} rows removed due to missing values for these maps. [{(len(df_mapPaths))} rows before, {len(df_tmp)} rows remain]")
        
        # Remove participants who do not have data for more than one session in this analysis
        participant_counts = df_tmp.groupby(col_studyID)[col_sesNum].nunique()
        valid_ids = participant_counts[participant_counts > 1].index.tolist() # keep IDs with data in at least one session
        df_tmp_drop = df_tmp[~df_tmp[col_studyID].isin(valid_ids)].copy()
        df_tmp = df_tmp[df_tmp[col_studyID].isin(valid_ids)]
        if verbose:
            print(f"\t\t{len(df_tmp_drop)} unique study-IDs removed given data for 0 or 1 sessions. [{(len(df_tmp) + len(df_tmp_drop))} rows before, {len(df_tmp)} rows remain]")

        # relabel session numbers to be sequential
        df_tmp = df_tmp.sort_values(by=['ID_study', 'Date'])
        df_tmp[col_sesNum] = df_tmp.groupby(['ID_study']).cumcount() + 1
        
        # sort df_tmp by number of unique sessions
        df_tmp['unique_sessions'] = df_tmp.groupby(col_studyID)[col_sesNum].transform('nunique')
        df_tmp = df_tmp.sort_values(by=['unique_sessions', col_studyID, col_sesNum], ascending=[False, True, True])
        #print(df_tmp.iloc[:25, [df_tmp.columns.get_loc(col) for col in ['UID', col_studyID, 'unique_sessions', 'SES', col_sesNum, *coi]]])
        
        n_before = df_mapPaths[col_studyID].nunique()
        n_after = df_tmp[col_studyID].nunique()
        n_removed = n_before - n_after

        if n_removed > 0:
            print(f"\t\t{n_after} unique study-ID pairs remain after removing {n_removed} IDs due to having one or zero sessions with data for this column.")
            if verbose:
                print(f"\t\tsutdy-IDs removed: {sorted(df_tmp_drop[col_studyID].unique())}")
        
        if n_after == 0:
            print(f"\t\t[extractMap] WARNING. No participants remain after filtering for complete study data. Skipping this map.")
            continue
        
        if coi is not None: # retain only the columns of interest in df_demo 
            df_tmp = df_tmp[['UID', 'study', col_studyID, col_sesNum, 'SES'] + coi + [col_L, col_R]].copy()

        for sesNum in df_tmp[col_sesNum].unique():  # TODO. Keep first two sessions with data, not necessaily group by session number
            df_tmp_ses = df_tmp[df_tmp[col_sesNum] == sesNum] # filter for rows from this study
            
            if verbose:
                print(f"\t\t\t[Session: {sesNum}] {len(df_tmp_ses)} rows")

            maps = get_maps(df_tmp_ses, mapCols=[col_L, col_R], col_ID = col_studyID, verbose=verbose)
            
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
                'sesNum': sesNum,
                'region': region,
                'surf': surf,
                'label': lbl,
                'feature': ft,
                'smth': smth,
                'df_demo': df_tmp_ses,
                'df_maps': maps,
            })
    
    if save:
        if save_pth is None:
            save_pth = os.getcwd()  # Default to current working directory
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        
        date = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
        if test:
            save_name = f"TEST_{save_name}"
        out_pth = f"{save_pth}/{save_name}_{date}.pkl"
        with open(out_pth, "wb") as f:
            pickle.dump(out_dl, f)
        print(f"Saved map_dictlist with z-scores to {out_pth}")
    
        if dlPrint: # print summary of output dict list
                if test:
                    print_dict(out_dl, df_print=False)
                else:
                    print_dict(out_dl)

    if verbose:
        if region:
            print(f"\n[extractMap] Returning list with {len(out_dl)} dictionary items (region: {region}).")
        else:
            print(f"\n[extractMap] Returning list with {len(out_dl)} dictionary items.")
    
    return out_dl

def parcellate_items(dl, df_keys, parcellationSpecs, df_save_pth, 
                     stats=None, save_pth=None, save_name=None,
                     verbose=False, test=False):
    """
    Parcellate vertex-wise dataframes.

    Input:
        dl [lst]: list of dicts
            Each dict should contain:
                'study'
                'grp'
                'grp_labels'
                'label'
                'feature'
                'region'
                'map_pths': pd.DataFrame with columns for subject ID, session, date and map_paths
                    Assumes map path is missing if either : map_pth
        df_keys [lst]: 
            list of strings of keys holding df to parcellate (should be in form of IDs (rows) by vertex (cols))
        parcellationSpecs [lst]: list of dicts
            Each dict should have keys:
                'region':       region that following settings apply to (should correspond to region value in dl items)
                'parcellate':   name of parcellation, Options: 'glasser', 'DK25' <eventually: 'schaefer100'>
                                False if not to parcellate
                'parc_lbl':     how to return parcellated index naming. Default = None (allowing default of parcellation function)        
        df_save_pth: str
            path to save parcellated dataframes to.
        stats: lst
            whether and how to summarise parcellated maps. 
            Options: 
                None - returns relabeled vertex names without summarising
                'mean', 'median', 'max', 'min', 'std','iqr'

        save_pth: str
            path to save the .pkl file to. If None, do not save.
        save_name: str
            name of the .pkl file to save
                
        verbose: bool
            whether to print progress and warnings
        test: bool 
            if True, only apply parcellation to first item in dl and first df_key in df_keys

    Output:
        dl_out [lst]: list of dicts
            Each dict is same as input dl, but with additional key:
                '{df_keys}_parc-{parc_name_shrt}': pd.DataFrame of identical size to df in df_keys
                    vertex names renamed according to parcellation
        parcellationSpecs [lst]: same as input parcellationSpecs, but with additional keys
                    
    """
    import datetime
    import numpy as np
    import os
    
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")
    # Prepare log file path
    logger = _get_file_logger(__name__, log_file_path)
    logger.info("Log started for winComp function.")
    print(f"[winComp] Saving log to: {log_file_path}")

    logger.info("Parameters:")
    logger.info(f"  df_keys: {df_keys}")
    logger.info(f"  parcellationSpecs: {parcellationSpecs}")
    logger.info(f"  stats: {stats}")
    logger.info(f"  df_save_pth: {df_save_pth}")
    logger.info(f"  save_pth: {save_pth}")
    logger.info(f"  save_name: {save_name}")
    logger.info(f"  verbose: {verbose}")
    logger.info(f"  test : {test}")
    
    dl_out = [] # initiate output list

    try:
        assert type(dl) == list, f"\t[parcellate_items] dl should be a list of dicts. Found {type(dl)}."
        assert type(parcellationSpecs) == list, f"\t[parcellate_items] parc_region should be a list of dicts. Found {type(parcellationSpecs)}."
        if type(stats) == str:
            stats = [stats]
        for s in stats:
            assert s.lower() in ['mean', 'mdn', 'median', 'max', 'min', 'std','iqr', None], f"\t[parcellate_items] Unknown stat: {s}. Supported: 'mean', 'mdn', 'median', 'max', 'min', 'std','iqr', `None`."
        
        if type(df_keys) == str:
            df_keys = [df_keys]
        
        parc_regions = [pr['region'] for pr in parcellationSpecs if pr.get('parcellate', False) != False]

        for index, region in enumerate(parcellationSpecs):
            parc = region.get('parcellate', None)

            if parc is None:
                continue
            elif parc == 'glasser':
                parc_name_shrt = "glsr"
            elif parc.lower() == 'dk25':
                parc_name_shrt = "dk25"
            else:
                raise ValueError(f"\t[parcellate_items] Unknown parcellation: {parc}. Supported: 'glasser', 'dk25'.")
        
            parcellationSpecs[index]['parc_name_shrt'] = parc_name_shrt
                
        if test:
            idx_len = 2
            idx_rdm = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
            logger.info(f"{start}: TEST MODE. Applying parcellations to {idx_len} random items in dl: indices {idx_rdm}.")
            print(f"{start}: TEST MODE. Applying parcellations to {idx_len} random items in dl: indices {idx_rdm}.")
            dl = [dl[i] for i in idx_rdm]
        else:
            logger.info(f"{start}: Applying parcellations for regions {parc_regions} in dictionary list of length {len(dl)}.\n\tSummarising with stat: {stats}.")
        
        dl_iterate = dl.copy()

        counter = 0 
        for idx, item in enumerate(dl_iterate):
            counter += 1
            if counter % 10 == 0:
                print(f"{datetime.datetime.now()}: Processing item {counter} of {len(dl_iterate)}...")
            
            key_outs = []
            if test:
                index = idx_rdm[idx]
            else:
                index = idx

            logger.info(f"\t{printItemMetadata(item, idx = index, return_txt=True)}")

            itm_region = item.get('region', None)
            itm_surf = item['surf'] # should raise error if this key doesnt exist
            
            if itm_region in parc_regions:
                pr = parcellationSpecs[parc_regions.index(itm_region)]
                parc = pr['parcellate']
                parc_name_shrt = pr['parc_name_shrt']

                parc_lbl = pr.get('parc_lbl', None)
                
            else:
                if verbose:
                    logger.info(f"\t[parcellate_items] Skipping item {idx} with region {itm_region} not in regions to parcellate.")
                dl_out.append(item)
                continue
            
            for df_key in df_keys:
                for s in stats:
                    
                    if s is None:
                        key_out = f'{df_key}_parc-{parc_name_shrt}'
                    else:
                        s = s.lower()
                        key_out = f'{df_key}_parc_{parc_name_shrt}_{s}'
                    key_outs.append(key_out)

                    df = item.get(df_key, None)
                    if type(df) == str:
                        df = loadPickle(df, verbose=False)

                    if df is None:
                        logger.info(f"\t[parcellate_items] WARNING. Key {df_key} not found in item {idx}. Skipping.")
                        continue
                    if df.shape[0] == 0:
                        logger.info(f"\t[parcellate_items] WARNING. No data found in item {idx} for key {df_key}. Skipping.")
                        continue
                    
                    if verbose:
                        logger.info(f"\t\tApplying parcellation `{parc}` with statistics `{s}`")

                    if parc == 'glasser':
                        df_parc = apply_glasser(df=df, surf=itm_surf, labelType=parc_lbl, addHemiLbl = False, ipsiTo = None, verbose = verbose)
                        item['parcellation'] = 'glasser'
                    elif parc == 'DK25': # for implementation of other parcellations
                        df_parc = apply_DK25(df=df, surf=itm_surf, labelType=parc_lbl, addHemiLbl = True, ipsiTo = None, verbose = verbose)
                        item['parcellation'] = 'DK25'
                    else:
                        pass

                    if s is None:
                        pass
                    elif s == 'mean':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).mean()
                    elif s == 'median' or s == 'mdn':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).median()
                    elif s == 'max':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).max()
                    elif s == 'min':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).min()
                    elif s == 'std':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).std()
                    elif s == 'iqr':
                        df_parc = df_parc.groupby(df_parc.columns, axis=1).quantile(0.75) - df_parc.groupby(df_parc.columns, axis=1).quantile(0.25)
                    else: # this check should not be necessary considering above check
                        raise ValueError(f"\t[parcellate_items] Unknown stat: {s}. Supported: None, 'mean', 'median', 'max', 'min', 'std','iqr'.")

                    df_save_name = f"{index}_{key_out}"
                    pth, sv_txt = savePickle(obj = df_parc, root = df_save_pth, name = df_save_name, 
                                    timeStamp = False, append = start, 
                                    test = test, rtn_txt = True, verbose = False)
                    logger.info(f"\t\t{sv_txt}")
                    item[key_out] = pth
            
            parcellationSpecs[parc_regions.index(itm_region)]['key_outs'] = key_outs # TODO. Properly add these new keys such that the dict related to region has all unique keys created 

            dl_out.append(item)
        
        if save_pth is not None and save_name is not None:
            if len(stats) == 1 and stats[0] is not None:
                save_name = save_name + "-" + stats  [0]  
            out_pth = savePickle(obj = dl_out, root = save_pth, name = save_name, test = test)
    
    except Exception as e:
        logger.error(f"An error occurred in parcellate_items: {e}", exc_info=True)
        print(f"[parcellate_items] ERROR: An error occurred. Check log file for details.")
    
    logger.info(f"\nCompleted. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[parcellate_items] Completed. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return dl_out, parcellationSpecs

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

def winComp(dl, demographics, keys_maps, col_grp, ctrl_grp, covars, out_df_save_pth,
            stat=['z'], key_demo = "df_demo", 
            save=True, save_pth=None, save_name="05a_stats_winStudy",  
            test=False, verbose=False, dlPrint=False):
    """
    Compute within study comparisons between control distribution and all participants
    Save path to statistics pickle files in the item.

    Input:
        dl: (list)
            list of dictionaries with map and demographic data for each comparison
        demographics: (dict)
            demographic column names
        keys_maps: (lst)
            keys in the dict items of dl that contains the maps to compute statistics on (eg. df_maps, df_maps_parc-glsr)
        ctrl_grp: (dict) 
            all control group patterns in the grouping column
        covars: (list) 
            covariates to ensure complete data for and to include in w-scoring.
        out_df_save_pth: (str)
            path to save vertex/parcel-wise z/w scores to avoid excessively large dictionary lists. 

        stat: (lst) of str
            statistics to compute. Options:
                'z' - z-score relative to control group <default>
                'w' - w-score relative to control group, adjusting for covariates
        key_demo: (str) <default: "df_demo">
            key in the dict items of dl that contains the demographics dataframe
        col_grp: (str) 
            name of the grouping column in demographics dataframe
        save: (bool) <default: True>
            whether to save the output dict list as a pickle file
        save_name: (str) <default: "05a_stats_winStudy">
            where to save output dictionary list
        test: (bool) <default: False>
            whether to run in test mode (randomly select 2 dict items to run)
        verbose: (bool) <default: False>
            whether to print shapes and other info
        dlPrint: (bool) <default: False>
            whether to print summary of output dict list
    
    Output:
        <saves computed statistics dfs>
        dl_winStats: (list) 
            list of dict items with path to dataframes in appropriately named keys
    """

    import numpy as np
    import pandas as pd
    import os
    import time
    import datetime

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    if test:
        save_name = f"TEST_{save_name}"
    
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")
    
    # Configure module logger (handlers added once)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info("Log started for winComp function.")
    print(f"[winComp] Saving log to: {log_file_path}")

    dl_out = []

    try:
        logger.info(f"[winComp] Saving log to: {log_file_path}")
        logger.info(f"\nComputing within study comparisons. Start time: {start}")
        logger.info(f"\tParameters: stats={stat}, covars={covars}, col_grp={col_grp}, ctrl_grp={ctrl_grp}")
        logger.info(f"\tDemographics columns: {demographics}")

        ctrl_values = [val for sublist in ctrl_grp.values() for val in sublist]

        if test:
            idx_len = 2 # number of indices
            if len(dl) < idx_len:
                idx_len = len(dl)
            idx = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
            dl_iterate = [dl[i] for i in idx]
            logger.info(f"\t[TEST MODE] Running z-scoring on {idx_len} randomly selected dict items: {idx}")
        else:
            dl_iterate = dl.copy()  # Create a copy of the original list to iterate over
            logger.info(f"\tNumber of dictionary items to process: {len(dl)}")
        
        stat = [s.lower() for s in stat] # make all strings in stat lower case

        counter = 0

        for i, item in enumerate(dl_iterate): # can be parallelized
            # print progress statements
            counter += 1
            if counter % 10 == 0:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{now} : Processing {counter} of {len(dl_iterate)}...")

            study = item['study']
            col_ID = get_IDCol(study, demographics) # determine ID col name based on study name

            if test:
                txt = printItemMetadata(item, idx = idx[i], return_txt=True)
            else:
                txt = printItemMetadata(item, idx = i, return_txt=True)

            logger.info(f"\n{txt}")

            df_demo = item.get(key_demo, None) # contains all participants
            
            item_orig = item.copy() # keep original item to compare changes

            for key in keys_maps:
                
                df_maps = item.get(key, None) # contains all participants, indexed by <IUD_>ID_SES
                if type(df_maps) == str:
                    df_maps = loadPickle(df_maps, verbose=False)
                
                logger.info(f"\n")

                if df_demo is None:
                    logger.info(f"\t\tKey '{key_demo}' is either not in dictionary or is a `None` object. Skipping this dict item.")
                    continue
                elif df_demo.shape[0] == 0:
                    logger.info(f"\t\tKey '{key_demo}' has no rows. Skipping this dict item.")
                    continue

                if df_maps is None or df_maps.shape[0] == 0:
                    logger.info(f"\t\tKey '{key}' is either not in dictionary or is a `None` object. Skipping this dict item.")
                    continue
                elif df_maps.shape[0] == 0:
                    logger.info(f"\t\tKey '{key}' has no rows. Skipping this dict item.")
                    continue
                else:
                    logger.info(f"\t\tProcessing maps in key '{key}' with shape {df_maps.shape}...")
                
                z_key_out = key + "_z"
                w_key_out = key + "_w"
                wMdl_key_out = key + "_wModels"

                if verbose: 
                    logger.info(f"\t\t\tInput shapes:\t\t[df_demo] {df_demo.shape} | [{key}] {df_maps.shape}")

                # 0.i Index df_demo appropriately
                col_ID = get_IDCol(study, demographics)
                col_SES = demographics['SES']
                
                if 'UID' in df_demo.columns:
                    df_demo['UID_ID_SES'] = df_demo['UID'].astype(str) + '_' + df_demo[col_ID].astype(str) + '_' + df_demo[col_SES].astype(str) # concat UID, ID and SES into single col 
                    df_demo.set_index('UID_ID_SES', inplace=True)
                else:
                    df_demo['ID_SES'] = df_demo[col_ID].astype(str) + '_' + df_demo[col_SES].astype(str) # concat ID and SES into single col
                    df_demo.set_index('ID_SES', inplace=True)

                # 0.ii Prepare covars
                if covars is None: # set defaults
                    covars = [demographics['age'], demographics['sex']]
                
                covars_copy = covars.copy()
                for c in list(covars_copy): 
                    if c not in demographics.keys(): # ensure covar is a key in demographics dict
                        logger.warning(f"Covariate '{c}' not a key in the demographics dictionary. Skipping this covar.")
                        covars_copy.remove(c)
                        continue
                    if c not in df_demo.columns: # ensure covar column exists in demo dataframe
                        logger.warning(f"Covariate '{c}' not found in demographics dataframe. Skipping this covar.")
                        covars_copy.remove(c)
                        continue

                # 0.iia Format covar to numeric, dummy code if categorical
                exclude_cols = [col for col in df_demo.columns if col not in covars_copy]  # exclude all columns but covars
                demo_numeric, catTodummy_log = catToDummy(df_demo, exclude_cols = exclude_cols)
                logger.info(f"\t\t\tConverted categorical covariates to dummy variables: \t{catTodummy_log}")

                # 0.iib Handle covariates
                missing_idx = []

                if 'w' in stat and (covars_copy == [] or covars_copy is None):
                    logger.warning("No valid covariates specified. Skipping w-scoring.")
                    w_internal = False
                    demo_num = demo_numeric.copy() # keep all rows in demo_numeric
                elif 'w' in stat: # Remove rows with missing covariate data
                    w_internal = True
                    covar_cols = [demographics[c] for c in covars_copy]
                    demo_num = demo_numeric.loc[:, covar_cols].copy() # keep only covariate columns in demo dataframe
                    missing_cols = demo_num.columns[demo_num.isnull().any()]

                    if len(missing_cols) > 0:
                        # count total number of cases with missing data
                        missing_idx = demo_num.index[demo_num.isnull().any(axis=1)].tolist()
                        logger.warning(f"{demo_num.isnull().any(axis=1).sum()} indices with missing covariate values: {missing_idx}")
                        demo_num_clean = demo_num.dropna().copy()
                        maps_clean = df_maps.loc[demo_num_clean.index, :].copy()
                    else:
                        missing_idx = []
                        demo_num_clean = demo_num.copy()
                        maps_clean = df_maps.copy()
                    
                    if demo_num_clean.shape[0] < 5: # Skip w-scoring if insufficient cases
                        logger.warning("Skipping w-scoring: ≤5 controls.")
                        w_internal = False

                else: # Do not drop cases for missing covariate data
                    demo_num_clean = demo_numeric.copy()
                    maps_clean = df_maps.copy()
                    w_internal = False
                
                # A. Create control and comparison subsets
                ids_ctrl = [j for j in df_demo[df_demo[col_grp].isin(ctrl_values)].index if j not in missing_idx]
                df_demo_ctrl = demo_num_clean.loc[ids_ctrl].copy() # extract indices from demo_num_clean
                # print(f"Control IDs: {ids_ctrl}")
                df_maps_ctrl = maps_clean.loc[ids_ctrl].copy() # extract indices from maps_clean

                if verbose: 
                    logger.info(f"\t\t\tControl group shapes:\t[demo] {df_demo_ctrl.shape} | [{key}] {df_maps_ctrl.shape}")
                
                demo_test = demo_num_clean.copy()
                maps_test = maps_clean.copy()

                if col_grp in demo_num_clean.columns:
                        demo_num_clean.drop(columns=[col_grp], inplace=True)

                if verbose: 
                    logger.info(f"\t\t\tTest group shapes:\t[demo] {demo_test.shape} | [{key}] {maps_test.shape}")
                
                df_demo_ctrl = demo_numeric.loc[df_demo_ctrl.index, :].copy() # keep only rows in demo_ctrl
                demo_test = demo_numeric.loc[demo_test.index, :].copy() # keep only rows in demo_test

                item[f'ctrl_IDs'] = list(df_maps_ctrl.index) # add ctrl_IDS to output dictionary item
                
                # B. Calculate statistics    
                # B.i. Prepare output dataframes
                df_out = pd.DataFrame(index=maps_test.index, columns=df_maps.columns)
                
                if verbose:
                    logger.info(f"\t\t\tOutput shape:\t\t[map stats] {df_out.shape}")
                
                if 'z' in stat and df_demo_ctrl.shape[0] > 3:
                    logger.info(f"\n\t\tComputing z scores [{df_demo_ctrl.shape[0]} controls]...")
                    start_time = time.time()
                    
                    z_scores = get_z(x = maps_test, ctrl = df_maps_ctrl)
                    
                    # Save outputs as pickle objects and keep path to these files in the dictionary
                    if test:
                        z_name = f"TEST_{idx[i]}_{z_key_out}"
                    else:
                        z_name = f"{i}_{z_key_out}"
                    
                    z_scores_pth, log = savePickle(obj = z_scores, root = out_df_save_pth, name = z_name, 
                                                   timeStamp = False, append = start,
                                                    rtn_txt=True, verbose = False)
                    logger.info(f"\t\t\t{log}")
                    item[z_key_out] = z_scores_pth

                    duration = time.time() - start_time
                    if duration > 60:
                        logger.info(f"\t\t\tComputed in {int(duration // 60):02d}:{int(duration % 60):02d} (mm:ss).")

                elif 'z' in stat:
                    logger.warning("\tWARNING. Skipping z-score: ≤2 controls.")

                if w_internal and df_demo_ctrl.shape[0] > 5 * len(covars_copy): #  SKIP if fewer than 5 controls per covariate
                    logger.info(f"\n\t\tComputing w scores [{df_demo_ctrl.shape[0]} controls, {len(covars_copy)} covars]...")
                    start_time = time.time()
                    if df_demo_ctrl.shape[0] < 10 * len(covars_copy):
                        logger.warning(f"INTERPRET W-SCORE WITH CAUTION: Few participants for number of covariates. Linear regression likely to be biased.")

                    df_w_out = df_out.copy() # n row by p map cols
                    
                    df_w_out, w_models = get_w(map_ctrl = df_maps_ctrl, demo_ctrl=df_demo_ctrl, map_test = maps_test, demo_test = demo_test, covars=covars_copy)

                    # Save outputs as pickle objects and keep path to these files in the dictionary
                    if test:
                        w_name = f"TEST_{idx[i]}_{w_key_out}"
                        wMdl_name = f"TEST_{idx[i]}_{wMdl_key_out}"
                    else:
                        w_name = f"{i}_{w_key_out}"
                        wMdl_name = f"{i}_{wMdl_key_out}"

                    df_w_pth, log_w = savePickle(obj = df_w_out, root = out_df_save_pth, name = w_name, 
                                                 timeStamp = False, append = start,
                                                 rtn_txt=True, verbose = False)
                    df_w_mdls_pth, log_wMdl = savePickle(obj = w_models, root = out_df_save_pth, name = wMdl_name, 
                                                         timeStamp = True, append = start, 
                                                         rtn_txt=True, verbose = False)
                    logger.info(f"\t\t\t{log_w}")
                    logger.info(f"\t\t\t{log_wMdl}")
                    item[w_key_out] = df_w_pth
                    item[wMdl_key_out] = df_w_mdls_pth
                
                    duration = time.time() - start_time
                    if duration > 60:
                        logger.info(f"\t\t\tW-scores computed in {int(duration // 60):02d}:{int(duration % 60):02d} (mm:ss).")

                elif w_internal:               
                    logger.warning(f"Skipping w-scoring: {df_demo_ctrl.shape[0]} controls ≤{5 * len(covars_copy)} controls (5 * number of covars [{len(covars_copy)}]).")
                else:
                    pass
            
            dl_out.append(item) # add item to output dl

        # Save dictlist to pickle file
        if save and len(dl_out) > 0:
            out_pth, txt = savePickle(obj = dl_out, root = save_pth, name = save_name,
                                      timeStamp = False, append = start,
                                    rtn_txt = True, verbose = True)            
            logger.info(f"{txt}")
        
        if dlPrint: # print summary of output dict list
            try:
                print_dict(dl_out)
            except Exception as e:
                logger.error(f"Error printing dict: {e}")
                logger.error(dl_out)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

    logger.info(f"Completed. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed. End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return dl_out

def search_df(df, ptrn, search_col, searchType='end'): 
    """
    Search column for matching pattern and return values from other columns joined with '_'.

    Input:
        df: pd.DataFrame
            df with `search_col`.
        ptrn: str
            Value to search for in the `search_col`.
        search_col: str
            Column to search for pattern.

        searchType: str
            Type of search. Options:
                'end' - search for values that end with the specified value,
                'begin' - search for values that begin with the specified value.
                'contains' - search for values that contain the specified value.
        
    Output:
        out: lst
            Indices with matching values matching pattern
    """
    import pandas as pd
    
    # ensure search_col exists
    if search_col not in df.columns:
        KeyError(f"[search_df] Error: search_col `{search_col}` not found in dataframe columns.")

    # return index of values in col_search matching ptrn
    if searchType.lower() not in ['end', 'begin', 'contains']:
        print(f"[search_df] Warning: searchType `{searchType}` not recognized. Use 'end', 'begin', or 'contains'.")
        return []
    elif searchType.lower() == 'begin':
        index_true = df[search_col].astype(str).str.startswith(ptrn, na=False) == True
    elif searchType.lower() == 'end':
        index_true = df[search_col].astype(str).str.endswith(ptrn, na=False)
    elif searchType.lower() == 'contains':
        index_true = df[search_col].astype(str).str.contains(ptrn, na=False)
    
    out = index_true[index_true].index.tolist()
    
    return out

def toIC(df_r, df_l):
    """
    Take in two dataframes, one for patients with L sided pathology, one for R sided pathology.
    Dataframes have column names ending with _L or _R to indicate hemisphere.

    Return dataframe with ipsi and contra columns.
    out shape: (n_r + n_l) x n_vertices [assuming both inputs have identical column names]
    """
    import pandas as pd
    
    # if pathology on R then col names ending with _R --> ipsi, _L --> contra
    df_r_ic = df_r.rename(columns=lambda x: x.replace('_R', '_ipsi').replace('_L', '_contra') if x.endswith('_R') or x.endswith('_L') else x)
    
    # if pathology on L then col names ending with _L --> ipsi, _R --> contra
    df_l_ic = df_l.rename(columns=lambda x: x.replace('_L', '_ipsi').replace('_R', '_contra') if x.endswith('_L') or x.endswith('_R') else x)
    
    df_ic = pd.concat([df_r_ic, df_l_ic], axis=0) # concatenate both dataframes

    return df_ic

def grp_flip(dl, demographics, goi, df_keys, col_grp, save_pth_df,
             save=True, save_pth=None, save_name="05b_stats_winStudy_grp", test=False, verbose=False):
    """
    Group participants and ipsi/contra flip maps according to side of lesion.

    Inputs:
        dl: (list)              List of dictionary items, each with keys:
        demographics: (dict)    Demographics file path and column names.
        goi: (list)             Groups of interest. List of group names to extract from demographics file.
        df_keys: (list)         Keys in the dict items to apply grouping and flipping to (eg. df_z, df_w).
            NOTE. Indices of these dfs should be UID_ID_SES
        col_grp: (str)          Column name in demographics file with group labels.
        save_pth_df: (str)
            path to save vertex/parcel-wise z/w scores to avoid excessively large dictionary lists. 

        save: (bool)            Whether to save the output dictionary list.
        save_pth: (str)         Directory path to save the output dictionary list.
        save_name: (str)        Base name for the saved output file.
        test: (bool)            Whether to run in test mode (process a small subset of data.
        verbose: (bool)         Whether to print detailed processing information.
        dlPrint: (bool)         Whether to print the output dictionary list.
    """
    import os
    import pandas as pd
    import numpy as np
    import copy
    import datetime

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    if test:
        save_name = f"TEST_{save_name}"
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")
    print(f"[grp_flip] Saving log to: {log_file_path}")

    # Configure module logger (handlers added per-file by _get_file_logger)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info("Log started for winComp function.")

    try:
        logger.info(f"[grp_flip] Saving log to: {log_file_path}")
        logger.info(f"Performing two steps:\n\ta. Selecting patients belonging to {goi}.\n\tb. Ipsi/contra flip.")
        logger.info(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\tParameters:\n\tinput dl: {len(dl)}\n\tgoi: {goi}\n\tdf_keys: {df_keys}\n\tsave_pth_df: {save_pth_df}\n\tcol_grp: {col_grp}\n\ttest: {test}\n\tverbose: {verbose}\n")

        if test:
            idx_len = 1 # number of indices
            if idx_len > len(dl):
                idx_len = len(dl)
            idx = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
            dl_iterate = [dl[i] for i in idx]
            dl_grp_ic = copy.deepcopy([dl[i] for i in idx])
            logger.info(f"[TEST MODE] Running ipsi/contra flipping on {idx_len} randomly selected dict items: {idx}")
        else:
            dl_iterate = dl.copy()  # Create a copy of the original list to iterate over
            dl_grp_ic = copy.deepcopy(dl)  # Create a copy of the original list for output and to iterate over

        counter = 0 
        for i, item in enumerate(dl_iterate):
            counter += 1
            if counter % 10 == 0:
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : Processing {counter} of {len(dl_iterate)}...")
            if test: 
                index = idx[i]
            else: 
                index = i
            logger.info(f"\n{printItemMetadata(item, idx=index, return_txt=True)}")

            df_demo = item[f'df_demo'].copy() # contains all participants
            IDs_ctrl = item.get('ctrl_IDs', None)
            col_ID = get_IDCol(item['study'], demographics)
            col_SES = demographics['SES']

            if df_demo is None or df_demo.empty or df_demo.shape[0] == 0:
                logger.warning(f"[grp_flip] WARNING. df_demo has no rows, is None or is empty [item index: {index}]. Skipping this item.")
                continue
            if col_grp not in df_demo.columns:
                logger.warning(f"[grp_flip] WARNING. col_grp `{col_grp}` not found in df_demo columns {df_demo.columns} [item index: {i}]. Skipping this item.")
                continue
            if IDs_ctrl is None or len(IDs_ctrl) == 0:
                logger.warning(f"[grp_flip] WARNING. No control IDs found for item index {index}. Skipping this item.")
                continue
            
            logger.info(f"\t\t{len(IDs_ctrl)} IDs CTRL: {IDs_ctrl}")
            for grp_val in goi: # for each group of interest
                
                if verbose:
                    logger.info(f"\tGrouping {grp_val}")
                
                # Get participants in this group
                demo_grp = df_demo[df_demo[col_grp].str.contains(grp_val)].copy()
                
                # extract IDs for grp_L and grp_R
                IDs_R = search_df(df=demo_grp, ptrn='R', search_col=col_grp,  searchType='end')
                IDs_L = search_df(df=demo_grp, ptrn='L', search_col=col_grp, searchType='end')
                
                # add group IDs to output dictionary item
                dl_grp_ic[i][f'{grp_val}_R_IDs'] = IDs_R
                dl_grp_ic[i][f'{grp_val}_L_IDs'] = IDs_L
                
                logger.info(f"\t\t{len(IDs_L)} IDs {grp_val}_L: {IDs_L}")
                logger.info(f"\t\t{len(IDs_R)} IDs {grp_val}_R: {IDs_R}")
                
                if type(df_keys) is str:
                    df_keys = [df_keys]

                for key in df_keys:
                    if verbose:
                        logger.info(f"\n\tProcesing key: {key}")

                    # Create df_{stat} for each side
                    df_stat = item.get(key, None)

                    if type(df_stat) is pd.DataFrame:
                        pass
                    elif type(df_stat) is str:
                        df_stat = loadPickle(df_stat, verbose = False)
                    
                        if df_stat is None: 
                            logger.info(f"\t\tKey '{key}' is either not in dictionary or is a `None` object. Skipping.")
                            continue

                    else:
                        logger.info(f"\t\tKey '{key}' is of type {type(df_stat)}, not str nor pd.DataFrame. Skipping.")
                        #print(f"\t\t[idx: {index}] Key '{key}' is of type {type(df_stat)}, not str nor pd.DataFrame. Skipping.")
                        continue

                    if df_stat.shape[0] == 0:
                        if verbose:
                            logger.info(f"\t\t\tKey '{key}' has no rows. Skipping.")
                        continue
                        
                    # split according to grps                
                    df_stat_grp = df_stat.copy() # search indexes of df_z for values in IDs_right. No need to split by SES, as df_z index is UID_ID_SES
                    df_stat_r = df_stat_grp[df_stat_grp.index.isin(IDs_R)]
                    df_stat_l = df_stat_grp[df_stat_grp.index.isin(IDs_L)]
                    
                    df_stat_ic = toIC(df_r = df_stat_r, df_l = df_stat_l)
                    
                    if verbose:
                        logger.info(f"\t\tShapes of {key}: L {df_stat_l.shape} | R {df_stat_r.shape} | IC {df_stat_ic.shape}")
                    
                    # save these dfs, add path to output dictionary item
                    for df, suffix in zip([df_stat_l, df_stat_r, df_stat_ic], ['L', 'R', 'ic']):
                        name = f"{key}_{grp_val}_{suffix}"
                        pkl_name = f"{index}_{name}"
                        if test:
                            pkl_name = f"TEST_{index}_{name}"

                        pth, svPkl_txt = savePickle(obj = df, root = save_pth_df, name = pkl_name, 
                                        timeStamp = False, append = start,
                                        rtn_txt=True, verbose = False)
                        
                        logger.info(f"\t\t\tPath {grp_val}_{suffix}: {pth}")

                        dl_grp_ic[i][name] = pth

                    dl_grp_ic[i][f'{key}_{grp_val}_R'] = df_stat_r
                    dl_grp_ic[i][f'{key}_{grp_val}_L'] = df_stat_l
                    dl_grp_ic[i][f'{key}_{grp_val}_ic'] = df_stat_ic

                    if dl_grp_ic[i].get(f'{key}_ctrl', None) is None: # if ctrl df not yet created, create it
                        
                        df_stat_ctrl = df_stat_grp[df_stat_grp.index.isin(IDs_ctrl)]
                        name = f"{key}_ctrl"
                        pkl_name = f"{index}_{name}"
                        if test:
                            pkl_name = f"TEST_{index}_{name}"
                        
                        pth, svPkl_txt = savePickle(obj = df_stat_ctrl, root = save_pth_df, name = pkl_name, 
                                         timeStamp = False, append = start,
                                          rtn_txt=True, verbose = False)
                        
                        logger.info(f"\t\t\tCTRL: {key} <{df_stat_ctrl.shape}> : {pth}")
                        
                        dl_grp_ic[i][name] = pth
                        
                        

        # Save dl to a pickle file
        if save:
            out_pth, log = savePickle(obj = dl_grp_ic, root = save_pth, name = save_name, 
                                      timeStamp = False, append = start, 
                                      rtn_txt=True, verbose = True)
            
            logger.info(f"\n{log}")
            print(f"\n{log}")

        logger.info(f"\n[grp_flip] Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n[grp_flip] Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EXITED WITH ERROR, see log.")
    

    return dl_grp_ic

def winD(dl, df_keys, save_pth_df,
         ipsiTo = 'L', 
         save=True, save_pth=None, save_name="05c_stats_winD", verbose=False, 
         test=False, test_len=1):
    """
    [winD: within study D-scoring]
    Calculate D-scores between groups and controls for all keys provided.
    Identifies all groups present in each dictionary item.
    
    Inputs:
        dl: (list)              Dictionary items keys holding vertex-wise within study statistics for groups of interest
                                    Assumes key for dfs holding statistics to have structure: df_{stat} 
        df_keys: (list)         Keys in dl items to compute d-scores from.
                                There should be one for the grp of interest and another for controls.
        ipsiTo: (str)           Hemisphere to which ipsi/contra vertices should be mapped to for controls. 
                NOTE. Only used if vertex columns have ipsi/contra data present. Default = 'L'
        save_pth_df: (str)      path to save dfs created in this function; keep path to these dfs in output dictionaries
        
        save: (bool)            Whether to save the output dictionary list.
        save_pth: (str)         Directory path to save the output dictionary list.
        save_name: (str)        Base name for the saved output file.
        test: (bool)            Whether to run in test mode (randomly select a subset of dict items to run d-scoring on)
        test_len: (int)         Number of random dict items to run d-scoring on if test=True
        verbose: (bool)         Whether to print detailed processing information.
    
    """
    
    import os
    import pandas as pd
    import numpy as np
    import datetime

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    
    if test:
        save_name = f"TEST_{save_name}"
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')

    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")
    print(f"[winD] Saving log to: {log_file_path}")
    
    # Configure module logger (handlers added per-file by _get_file_logger)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info(f"[winD] Saving log to: {log_file_path}")
    logger.info("Log started for winComp function.")
    
    logger.info(f"\nParameters:")
    logger.info(f"\tinput dl: {len(dl)}")
    logger.info(f"\tdf_keys: {df_keys}")
    logger.info(f"\tipsiTo: {ipsiTo}")
    logger.info(f"\tsave_pth_df: {save_pth_df}")
    logger.info(f"\tsave: {save}")
    logger.info(f"\tsave_pth: {save_pth}")
    logger.info(f"\tsave_name: {save_name}")
    logger.info(f"\ttest: {test}")
    logger.info(f"\ttest_len: {test_len}")
    logger.info(f"\tverbose: {verbose}")
                
    logger.info(f"\n")

    try:
        
        if test:
            idx = np.random.choice(len(dl), size=test_len, replace=False).tolist()  # randomly choose index
            dl_iterate = [dl[i] for i in idx]
            logger.info(f"[TEST MODE] Running d-scoring on {test_len} randomly selected dict items: {idx}")
            print(f"[winD] TEST MODE. Running d-scoring on {test_len} randomly selected dict items: {idx}")
        else:
            dl_iterate = dl.copy()  # Create a copy of the original list to iterate over

        logger.info(f"Calculating D-scores for {len(dl_iterate)} dictionary items for the following df_keys: {df_keys}\n")

        counter = 0
        
        dl_out = []
        for i, item in enumerate(dl_iterate):
            dl_out.append(item) # add item to output dl
            
            counter += 1
            if counter % 10 == 0:
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : Processing {counter} of {len(dl_iterate)}...")

            if test: 
                index = idx[i]
            else: 
                index = i
            itm_txt = printItemMetadata(item, idx=index, return_txt=True)
            
            logger.info(itm_txt)
            # list to save before concatenating all dfs into single df
            d_dfs = [] 
            d_dfs_ic = []

            # compare every other df to the control df, put results into a single df_{stat}_d
            for key in df_keys: # iterate over all statistics of interest [df_z, df_w]

                # identify all other dfs for this stat
                key_grps = [k for k in item.keys() if k.startswith(f'{key}_') and k != f'{key}_ctrl' and k != f'{key}_models']
                if len(key_grps) == 0:
                    logger.info(f"\tWARNING: `{key}`: No group dfs. Skipping.")
                    continue
                
                df_stat_ctrl = item.get(f'{key}_ctrl', None)
                if type(df_stat_ctrl) is str:
                    df_stat_ctrl = loadPickle(df_stat_ctrl, verbose=False)

                if df_stat_ctrl is None:
                    logger.info(f"\tWARNING: `{key}`: No control group. Skipping.")
                    continue
                elif df_stat_ctrl.shape[0] == 0:
                    logger.info(f"\tWARNING: `{key}`: Control gorup has no rows. Skipping.")
                    continue
                else:
                    logger.info(f"\t`{key}: CTRL grp <{df_stat_ctrl.shape}>")
                    if verbose:
                        logger.info(f"\t\tGroups: {key_grps}")

                # create df_stat_ctrl_ic
                if any('_ic' in k for k in key_grps): # if any of the keys are ipsi/contra flipped, create a df_stat_ctrl_ic
                    df_stat_ctrl_ic = df_stat_ctrl.copy()
                    if ipsiTo == 'L':
                        df_stat_ctrl_ic.columns = [col.replace('_R', '_contra') if '_R' in col else col.replace('_L', '_ipsi') for col in df_stat_ctrl.columns]
                    else:
                        df_stat_ctrl_ic.columns = [col.replace('_L', '_contra') if '_L' in col else col.replace('_R', '_ipsi') for col in df_stat_ctrl.columns]

                    item['ipsiTo'] = ipsiTo
                    df_ctrl_name = f"{index}_{key}_ctrl_ic"
                    if test:
                        df_ctrl_name = f"TEST_{df_ctrl_name}"

                    out_pth, log_df_ctrl_ic = savePickle(obj = df_stat_ctrl_ic, root = save_pth_df, name = df_ctrl_name,
                                                   timeStamp = False, append = start, rtn_txt = True, verbose = False)
                    item[f'{key}_ctrl_ic'] = out_pth # add to dictionary item
                    logger.info(f"\t\t{log_df_ctrl_ic} [ipsiTo: {ipsiTo}]")
                    
                # initialize output dfs
                d_df = pd.DataFrame(columns=df_stat_ctrl.columns)
                d_df_ic = pd.DataFrame(columns=df_stat_ctrl_ic.columns)

                # compute d scores
                for kg in key_grps:
                    logger.info(f"\t`{kg}`")
                    df_stat = item.get(kg, None)
                    
                    if df_stat is None:
                        logger.info(f"\tWARNING: Key '{kg}' is either not in dictionary or is a `None` object. Skipping.")
                        continue
                    elif type(df_stat) is str:
                        df_stat = loadPickle(df_stat, verbose=False)
                        if df_stat is None: 
                            logger.info(f"\tWARNING: Key '{kg}' could not be loaded from path {df_stat}. Skipping.")
                            continue
                    elif type(df_stat) is not pd.DataFrame:
                        logger.info(f"\tWARNING: Key '{kg}' is of type {type(df_stat)}, not str nor pd.DataFrame. Skipping.")
                        continue

                    if verbose:
                        logger.info(f"\t\t{kg} (shape {df_stat.shape})")
                    else:
                        logger.info(f"\t\t{kg}")
                    
                    if df_stat is None or df_stat.shape[0] == 0:
                        logger.info(f"\tWARNING: No data in {kg}. Skipping.")
                        continue
                    
                    grp_name = kg.replace(f"{key}_", '')

                    if 'ic' in kg.lower(): # use appropriate control df with ipsi/contra labelled cols
                        grp_name = f"{grp_name}_ipsiTo-{ipsiTo}"
                        out_ic = get_d(ctrl = df_stat_ctrl_ic, test = df_stat, varName = key, test_name = grp_name) # use control df with ic flipped vertices
                        d_df_ic = pd.concat([d_df_ic, out_ic], axis=0) # append out_ic to d_df_ic
                        d_df_ic.drop_duplicates(inplace=True)
                    else:
                        out = get_d(ctrl = df_stat_ctrl, test = df_stat, varName = key, test_name = grp_name)
                        d_df = pd.concat([d_df, out], axis=0) # append out to d_df
                        d_df.drop_duplicates(inplace=True)

                d_dfs.append(d_df) # add to list of dfs
                d_dfs_ic.append(d_df_ic)
            
            # concatenate all d_dfs into single df
            if len(d_dfs) > 1:
                d_df = pd.concat(d_dfs, axis=0)
            elif len(d_dfs) == 1:
                d_df = d_dfs[0]
            else:
                d_df = None

            # concatenate all d_dfs_ic into single df
            if len(d_dfs_ic) > 1:
                d_df_ic = pd.concat(d_dfs_ic, axis=0)
            elif len(d_dfs_ic) == 1:
                d_df_ic = d_dfs_ic[0]
            else:
                d_df_ic = None

            # save D-score dfs to items
            df_name = f"{index}_d"
            df_ic_name = f"{index}_d_ic"
            if test:
                df_name = f"TEST_{df_name}"
                df_ic_name = f"TEST_{df_ic_name}"

            pth_df_d, log_df_d = savePickle(obj = d_df, root = save_pth_df, name = df_name,
                                timeStamp = False, append = start, rtn_txt = True, verbose = False)
            dl_out[i]['df_d'] = pth_df_d
            logger.info(f"\n\t{log_df_d}")
            
            pth_df_d_ic, log_df_d_ic = savePickle(obj = d_df_ic, root = save_pth_df, name = df_ic_name,
                                timeStamp = False, append = start, rtn_txt = True, verbose = False)
            dl_out[i]['df_d_ic'] = pth_df_d_ic          
            logger.info(f"\t{log_df_d_ic}")

        if save: # save dl
            out_pth, log_dl = savePickle(obj = dl_out, root = save_pth, name = save_name, 
                                 timeStamp = False, append = start, rtn_txt = True, verbose = False)
            
            logger.info(f"\n{log_dl}")
            print(f"[winD] Dictlist saved to: {out_pth}")

        end = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"Completed: {end}")
        print(f"[windD] Completed: {end}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EXITED WITH ERROR, see log.")

    return dl

def btwD(dl, save_pth_df,
         save=True, save_pth=None, save_name="05d_stats_btwD", 
         verbose=False, test=False, test_len=1):
    """
    [btwD: within study D-scoring]
    Calculate D-statistic differences between analogous dictionary items for each study.
    Difference scores calculated (managed by function [get_d]):
        d_7T - d_3T
        d_7T - d_3T / d_7T
        d_7T - d_3T / d_3T
    
    Inputs:
        dl: (list)              Dictionary items keys holding vertex-wise within study statistics for groups of interest
                                    Assumes key for dfs holding statistics to have structure: df_{stat} 
        save_pth_df: (str)      path to save dfs created in this function; keep path to these dfs in output dictionaries

        save: (bool)            Whether to save the output dictionary list.
        save_pth: (str)         Directory path to save the output dictionary list.
        save_name: (str)        Base name for the saved output file.
        test: (bool)            Whether to run in test mode (randomly select a subset of dict items to run d-scoring on)
        test_len: (int)         Number of random dict items to run d-scoring on if test=True
        verbose: (bool)         Whether to print detailed processing information.
    
    """
    
    import os
    import pandas as pd
    import numpy as np
    import datetime
    import logging

    # Prepare log file path
    if save_pth is None:
        print("WARNING. Save path not specified. Defaulting to current working directory.")
        save_pth = os.getcwd()  # Default to current working directory
    
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    
    if test:
        save_name = f"TEST_{save_name}"
    
    start = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')

    log_file_path = os.path.join(save_pth, f"{save_name}_log_{start}.txt")
    print(f"[btwD] Saving log to: {log_file_path}")
    
    # Configure module logger (handlers added per-file by _get_file_logger)
    logger = _get_file_logger(__name__, log_file_path)
    logger.info(f"[btwD] Start time: {start}")
    logger.info(f"Saving log to: {log_file_path}")
    
    logger.info(f"\nParameters:")
    logger.info(f"\tinput dl: {len(dl)}")
    logger.info(f"\tsave_pth_df: {save_pth_df}")
    logger.info(f"\tsave: {save}")
    logger.info(f"\tsave_pth: {save_pth}")
    logger.info(f"\tsave_name: {save_name}")
    logger.info(f"\ttest: {test}")
    logger.info(f"\ttest_len: {test_len}")
    logger.info(f"\tverbose: {verbose}")
                
    logger.info(f"\n")

    comps = [] # initialize output

    try:
        
        if test:
            idx = np.random.choice(len(dl), size=test_len, replace=False).tolist()  # randomly choose index
            if isinstance(idx, int):
                idx = [idx]
            
            test_indices = list(idx)
            for i in idx: # for each randomly selected index, find its matching pai
                idx_other = get_pair(dl, idx = i, mtch=['region', 'surf', 'label', 'feature', 'smth'], skip_idx=idx)
                if idx_other is None:
                    test_indices.remove(i)
                    continue
                if isinstance(idx_other, int):
                    idx_other = [idx_other]
               
                for j in idx_other:
                    if j not in test_indices:
                        test_indices.append(j)
                
            dl_iterate = [dl[i] for i in test_indices]
            logger.info(f"[TEST MODE] Running d-scoring on {test_len} pair(s) of randomly selected dict items: {test_indices}")
        else:
            dl_iterate = dl.copy() # Create a copy of the original list to iterate over

        logger.info(f"Comparing D-scores between studies for {len(dl_iterate)} dictionary items.\n")

        comps = []
        skip_idx = []
        counter = 0

        for i, item in enumerate(dl_iterate):
            counter += 1
            if counter % 10 == 0:
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : Processing {counter} of {len(dl_iterate)}...")
            
            if i in skip_idx:
                continue
            else:
                skip_idx.append(i)
                       
            idx_other = get_pair(dl_iterate, idx = i, mtch=['region', 'surf', 'label', 'feature', 'smth'], skip_idx=skip_idx)
            if idx_other is None:
                logger.info(f"\tNo matching index found. Skipping.")
                continue
            skip_idx.append(idx_other)
            
            if test:
                index = test_indices[i]
                index_other = test_indices[idx_other]
            else:
                index = i
                index_other = idx_other
            
            txt_tT = printItemMetadata(item, idx=index, return_txt=True)
            txt_sT = printItemMetadata(dl_iterate[idx_other], idx=index_other, return_txt=True)
            logger.info(f"{txt_tT}\t{txt_sT}")

            # identify which study is which (to know how to subtract)
            tT_idx, sT_idx = determineStudy(dl_iterate, i, idx_other, study_key = 'study')

            item_tT = dl_iterate[tT_idx]
            item_sT = dl_iterate[sT_idx]
            """
            study = item['study']
            if study == 'MICs':
                item_tT = item
                item_sT = item_other
            else:
                item_tT = item_other
                item_sT = item
            """

            # if df_z and df_w are None, skip
            if item.get('df_d', None) is None and item.get('df_d_ic', None) is None:
                logger.warning(f"\tNo D-score dataframes in this dictionary item. Skipping.")
                continue

            # Initialize output item
            out_item = {
                    'studies': (item_tT['study'], item_sT['study']),
                    'region': item_tT['region'],
                    'feature': item_tT['feature'],
                    'surf': item_tT['surf'],
                    'label': item_tT['label'],
                    'smth': item_tT['smth'],
                    'parcellation': item_tT.get('parcellation', None)
                }
            
            ID_keys = [key for key in item_tT.keys() if 'IDs' in key]
            keys_to_copy = ID_keys + ['df_d', 'df_d_ic']
            if verbose:
                logger.info(f"\tCopying keys: {keys_to_copy}")
            for key in keys_to_copy: # add all ID_keys to corresponding dataframes to out_item
                out_item[key] = [item_tT[key], item_sT[key]] # stores as list of items. In case of dfs, list of dataframes            

            for df in ['df_d', 'df_d_ic']:
                metrics_df = None
                df_tT = item_tT.get(df, None)
                df_sT = item_sT.get(df, None)
                
                if df_tT is None:
                    logger.warning(f"\t Object stored in key `{df} of 3T item is None object. Skipping comparisons.")
                    continue
                elif type(df_tT) is str:
                    df_tT = loadPickle(df_tT, verbose=False)
                elif type(df_tT) is not pd.DataFrame:
                    logger.warning(f"\tObject stored in key '{df}' of 3T item is of type {type(df_tT)}, not str nor pd.DataFrame. Skipping comparison.")
                    continue
                elif df_tT.shape[0] == 0:
                    logger.warning(f"\t3T {df} has no rows. Skipping {df} comparison.")
                    continue
                
                if df_sT is None:
                    logger.warning(f"\t Object stored in key `{df} of 7T item is None object. Skipping comparisons.")
                    continue
                elif type(df_sT) is str:
                    df_sT = loadPickle(df_sT, verbose=False)
                elif type(df_sT) is not pd.DataFrame:
                    logger.warning(f"\tObject stored in key '{df}' of 7T item is of type {type(df_sT)}, not str nor pd.DataFrame. Skipping comparison.")
                    continue
                elif df_sT.shape[0] == 0:
                    logger.warning(f"\t7T {df} has no rows. Skipping {df} comparison.")
                    continue

                logger.info(f"\tComputing difference metrics for {df}...")
                
                # TODO. Implement better method for this
                try: # catch persisting NoneTypes 
                    test_shape = df_tT.shape
                    test_shape_2 = df_sT.shape
                except:
                    printItemMetadata(item_tT, idx=index, return_txt=False)
                    printItemMetadata(item_sT, idx=index_other, return_txt=False)
                    logger.info(f"\tItem index {index} study {item_tT['study']} | Item index {index_other} study {item_sT['study']}")
                    logger.warning(f'\tCould not get shapes of dfs. Skipping')
                    print("ERROR: Could not get shapes of dfs.")
                    print(f"\tType df_tT: {type(df_tT)}\n\tType df_sT: {type(df_sT)}.")
                    print(f"\tSkipping")
                    continue

                if verbose:
                    logger.info(f"\t\t3 T shape: {df_tT.shape}\n\t\t7 T shape: {df_sT.shape}.")
                
                if df == 'df_d_ic': # add ipsiTo key to output
                    assert item_tT.get('ipsiTo', "Not found") == item_sT.get('ipsiTo', "Found not"), f"[btwD] ipsiTo values do not match between dictionary items. {item_tT.get('ipsiTo', 'Not found')} != {item_sT.get('ipsiTo', 'Found not')}"
                    out_item['ipsiTo'] = item_tT['ipsiTo']

                assert df_tT.columns.equals(df_sT.columns) == True, f"[comps] Columns of d-scores DataFrames do not match. Check input data. {df_tT.columns} != {df_sT.columns}"

                # Identfiy the rows that refer to d statistics and ensure that both dataframes have these rows
                stats_tT = df_tT.index.tolist()
                stats_sT = df_sT.index.tolist()
                d_stats = [s for s in stats_tT if s in stats_sT and s.startswith('d_')]
                if verbose:
                    logger.info(f"\t\tCohen's d statistic indices: {d_stats}")
                
                # want to only apply operations to rows in d_stats
                df_d_stats_tT = df_tT.loc[d_stats, :].copy()
                df_d_stats_sT = df_sT.loc[d_stats, :].copy()

                # compute difference metrics
                d_dif = df_d_stats_sT - df_d_stats_tT # 7 T - 3 T
                d_dif_by3T = d_dif / df_d_stats_tT
                d_dif_by7T = d_dif / df_d_stats_sT
                
                # Stack the rows from each matrix into the metrics_df
                d_dif_renamed = d_dif.copy()
                d_dif_renamed.index = [idx + '_Δd' for idx in d_dif_renamed.index]
                
                d_dif_by3T_renamed = d_dif_by3T.copy()
                d_dif_by3T_renamed.index = [idx + '_Δd_by3T' for idx in d_dif_by3T_renamed.index]
                
                d_dif_by7T_renamed = d_dif_by7T.copy()
                d_dif_by7T_renamed.index = [idx + '_Δd_by7T' for idx in d_dif_by7T_renamed.index]

                metrics_df = pd.concat([d_dif_renamed, d_dif_by3T_renamed, d_dif_by7T_renamed])

                # Add original stats from both datasets to metrics_df
                df_tT_renamed = df_tT.copy()
                df_tT_renamed.index = [idx + '_3T' for idx in df_tT_renamed.index]
                df_sT_renamed = df_sT.copy()
                df_sT_renamed.index = [idx + '_7T' for idx in df_sT_renamed.index]
                metrics_df = pd.concat([metrics_df, df_tT_renamed, df_sT_renamed])
                logger.info(f"\t\tShape of metrics_df: {metrics_df.shape}")
                logger.info(f"\t\tIndices: {metrics_df.index}")        

                # save dfs to csv, add path to out_item
                out_pth, rtn_txt = savePickle(obj = metrics_df, root = save_pth_df, name = f"{index}_{df}_btwD",
                                 timeStamp = False, append = start, rtn_txt = True, verbose = False)
                logger.info(f"\t{rtn_txt}\n")
                out_item[f'comps_{df}'] = out_pth # add to output item

            comps.append(out_item) # append to dict list
            out_item = None

        # Save the updated map_dictlist to a pickle file
        if save:
            out_pth, rtn_txt = savePickle(obj = comps, root = save_pth, name = save_name, 
                                 timeStamp = False, append = start, rtn_txt = True, verbose = False)
            logger.info(f"\n{rtn_txt}")
            print(f"[btwD] Dictlist saved to: {out_pth}")

        
        end_time = datetime.datetime.now()
        logger.info(f"{end_time}: Completed btw.D")
        print(f"[btwD] Completed: {end_time}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EXITED WITH ERROR, see log.")

    return comps


def determineStudy(dl, idx, idx_other, study_key='study'):
    """
    Determine study assignment of each provided dictionary item.
    Assumes that 3T study code is 'MICS', does not assume name for 7T study.
    
    input:
        dl: (list)            List of dictionary items
        idx: (int)            One index of item in dict list
        idx_other: (int)      Another index of item in dict list
        study_key: (str)      Key in both dictionary items that holds the study name. Default is 'study'.
    """

    study = dl[idx].get(study_key, None)
    study_other = dl[idx_other].get(study_key, None)
    assert study is not None and study_other is not None, f"One of the dictionary items does not have the key '{study_key}'."
    assert study != study_other, f"Both dictionary items have the same study '{study}'. They should be different."

    if study == 'MICs':
        idx_tT, idx_sT = idx, idx_other
    else:
        idx_sT, idx_tT = idx, idx_other

    return idx_tT, idx_sT

def df_head(obj, n=5):
    """
    Return a representation of the first n rows for a DataFrame/Series or for a pickled DataFrame path.
    Falls back to str(obj) for other types.
    """
    import pandas as pd
    # If it's a path to a pickle, try to load
    if isinstance(obj, str):
        try:
            maybe_df = loadPickle(obj, verbose=False)
            if isinstance(maybe_df, (pd.DataFrame, pd.Series)):
                return maybe_df.head(n)
            else:
                return str(maybe_df)[:1000]
        except Exception:
            return f"<unreadable path: {obj}>"
    if isinstance(obj, pd.Series):
        return obj.head(n)
    if isinstance(obj, pd.DataFrame):
        return obj.head(n)
    if isinstance(obj, list) and all(isinstance(x, pd.DataFrame) for x in obj):
        # return list of heads for each dataframe
        return [x.head(n) for x in obj]
    # fallback
    try:
        return str(obj)
    except Exception:
        return "<unrepresentable object>"

def print_dict(dict, df_print=False, idx=None, verbose=False, return_txt=False):
    """
    Print the contents of a dictionary with DataFrames in a readable format.
    Input:
        dict: list of dict items.
        df_print: bool
            if True, prints DataFrame contents; if False, only print the shape of the DF keys
        idx: list of ints
            if provided, only print the items at these indices in the dict list.
        
        return_txt: bool
            if True, returns the printed output as a string instead of printing to console.
        verbose:
            if True, prints additional information.
    Output:
        Prints the keys and values of each dictionary item.
    """
    import pandas as pd
    if return_txt:
        import io
        import sys
        # Capture the output of the print statements
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

    def mainPrint(k, v, df_print):
        import pandas as pd
        
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series): # value is dataframe
            if df_print:
                head = df_head(v, n=5)
                print(f"\t{k}:\n\t\t<DataFrame shape={v.shape}>\n{head}")
            else:
                print(f"\t{k}:\n\t\t<DataFrame shape={v.shape}>")
        
        elif 'df_' in k and isinstance(v, str): # value is path to dataframe
            v_df = loadPickle(v, verbose = False)
            if df_print:
                head = df_head(v_df, n=5)
                print(f"\t{k}: {v}\n\t\t<DataFrame shape={v_df.shape}>\n{head}")
            else:
                print(f"\t{k}: {v}\n\t\t<DataFrame shape={v_df.shape}>")

        elif isinstance(v, list) and all(isinstance(x, pd.DataFrame) for x in v): # value is list of dataframes
            if df_print:
                for idx_df, df_v in enumerate(v):
                    print(f"\t{k}[{idx_df}]: {df_v}")
            else:
                shapes = [df_v.shape for df_v in v]
                print(f"\t{k}: <list (len {len(v)}) of DataFrames. Shapes : {shapes}>")
            #print(f"\t{k}: <DataFrame shape={v.shape}>")
            if df_print: print(f"\t{k}: {v}")
        else: # other values
            print(f"\t{k}: {v}")

    if idx is not None:
        if type(idx) is int:
            idx = [idx]
        if verbose:
            print(f"\n Printing the following {len(idx)} indices: {idx}")
            print(f"\tKeys: {list(d.keys())}")
        for i in idx:
            d = dict[i]
            print(f"\n[{i}]")
            for k, v in d.items():
                mainPrint(k, v, df_print)
    else:
        print(f"\n Dict list length ({len(dict)} items)")
        if verbose:
            print(f"\tKeys: {list(d.keys())}")
        for i, item in enumerate(dict):
            d = item
            print(f"\n[{i}]")
            for k, v in d.items():
                mainPrint(k,v, df_print)
    
    if return_txt:
        # Restore original stdout
        sys.stdout = old_stdout
        return mystdout.getvalue()

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
    
    matches = []
    for j, other in enumerate(dl):
        if j not in skip_idx:
            match = True
            for key in mtch:
                if other.get(key, None) != item.get(key, False):
                    match = False
                    break
            if match:
                matches.append(j)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"[get_pair] WARNING: Multiple matches found for index {idx} with keys {mtch}. Returning all matches: {matches}")
        return matches
    else:
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

def printItemMetadata(item, return_txt = False, idx=None, clean = False, printStudy = True):
    """
    Print metadata of a dictionary item in a readable format.
    
    Parameters:
    item: Dictionary containing metadata
    return_txt: If True, returns the formatted string instead of printing it
    idx: Optional index to include in the output
    clean: If True, removes tabs and problematic characters from the output
    printStudy: If True, includes the study name in the output
    
    Returns:
        txt or print statement
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
        if printStudy == True:
            txt = f"[{study}] - {region}: {feature}, {surf}, {label}, {smth}mm (idx {idx})"
        else:
            txt = f"{region}: {feature}, {surf}, {label}, {smth}mm (idx {idx})"
    else:
        if printStudy == True:
            txt = f"[{study}] - {region}: {feature}, {surf}, {label}, {smth}mm"
        else:
            txt = f"{region}: {feature}, {surf}, {label}, {smth}mm"
    
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
        
def relabel_vertex_cols(df, ipsiTo=None, split = False, verbose = False):
    """
    Take df with columns '{idx}_{hemi}' and return two dfs, split according to hemisphere suffix.
    Supports ipsi/contra labelled columns if ipsiTo is provided.

    Input:
        df: vertex-wise dataframe with vertex in columns, pts in rows. All vertices from both hemispheres should be present.
            Number of columns per hemisphere should be 32492 for fsLR-32k
        
        ipsiTo: if provided, searches for columns ending with '_ipsi' and '_contra' and maps '_ipsi' indices to  
        split: 
            If true, returns two dataframes, one for each hemisphere. 
            If false, returns a single dataframe with combined hemispheres (adding length of df_l to indices on the right).
        verbose:
            If print additional information
    """
    import numpy as np

    if verbose:
        cols_list = list(dict.fromkeys(df.columns))
        print(f"[rlbl_vrtx] Input cols (first 5 and last 5 cols): ({list(cols_list[:5] + cols_list[-5:])})")
    # split df into cols for each hemi
    if ipsiTo is not None:
        # seperate by ipsi and contra
        df_hemiIpsi = df[[col for col in df.columns if col.endswith('_ipsi')]]
        df_hemiContra = df[[col for col in df.columns if col.endswith('_contra')]]

        # replace suffixes
        df_hemiIpsi.columns = [col.replace('_ipsi', '') for col in df_hemiIpsi.columns]
        df_hemiContra.columns = [col.replace('_contra', '') for col in df_hemiContra.columns]

        # assign L and R
        if ipsiTo == 'L':
            df_hemiL = df_hemiIpsi
            df_hemiR = df_hemiContra
        elif ipsiTo == 'R':
            df_hemiR = df_hemiIpsi
            df_hemiL = df_hemiContra

    else:
        df_hemiL = df[[col for col in df.columns if col.endswith('_L')]]
        df_hemiR = df[[col for col in df.columns if col.endswith('_R')]]

        # replace suffixes
        df_hemiL.columns = [col.replace('_L', '') for col in df_hemiL.columns]
        df_hemiR.columns = [col.replace('_R', '') for col in df_hemiR.columns]  
    
        if verbose:
            print(f"\tTotal vertices per hemi: {len(df_hemiL.columns)}")
            print(f"\tL: n unique, min, max = {len(np.unique(df_hemiL.columns.astype(int)))}, {min(df_hemiL.columns.astype(int))}, {max(df_hemiL.columns.astype(int))}")
            print(f"\tR: n unique, min, max = {len(np.unique(df_hemiR.columns.astype(int)))}, {min(df_hemiR.columns.astype(int))}, {max(df_hemiR.columns.astype(int))}")

    if split:
        if verbose:
            print("\t[relabel_vertex_cols] Returning two dataframes, one for each hemisphere.")
        return df_hemiL, df_hemiR
    else:
        assert len(df_hemiL.columns) == len(df_hemiR.columns), f"[relabel_vertex_cols] Number of columns in L and R hemispheres do not match. {len(df_hemiL.columns)} != {len(df_hemiR.columns)}"
        df_hemiR.columns = df_hemiR.columns.astype(int) + len(df_hemiL.columns) # shift R indices to be continuous with L
        df_out = df_hemiL.join(df_hemiR, how='outer') # join L and R dfs
        
        if verbose:
            print(f"\tCombined: n unique, min, max = {len(np.unique(df_out.columns.astype(int)))}, {min(df_out.columns.astype(int))}, {max(df_out.columns.astype(int))}")
            print(f"\tTotal vertices combined: {len(df_out.columns)}")
            #print(f"\tObs: {list(dict.fromkeys(df_out.columns.astype(int)))}")
            #print(f"\tCnt: {list(range(len(df_out.columns)))}")
            print("\t[relabel_vertex_cols] Returning single dataframe with combined hemispheres.")
        
        return df_out

def apply_glasser(df, surf, labelType=None, addHemiLbl = False, ipsiTo=None, verbose=False):
    """
    Input:
        df: vertex-wise dataframe with vertex in columns, pts in rows. All vertices from both hemispheres should be present.
            should be fsLR-32k (32492 vertices per hemisphere) 
        surf: 'fsLR-32k' or 'fsLR-5k'
        labelType (case insensitive): final label to return. 
            options:
                - 'glasser_int': integer [0:360] indicating glasser region
                - 'glasser_name_short': string with short glasser region name (e.g. 'V1', 'V2', etc)
                - 'glasser_name_long': string with long glasser region name (e.g. 'Primary_Visual_Cortex', 'Second_Visual_Area', etc)
                - 'lobe': string with lobe name ('Occ', 'Fr', 'Par', 'Temp')
                - 'lobelong': string with long lobe name ('occipital', 'frontal', 'parietal', 'temporal')

        addHemiLbl: if True, adds hemisphere label to the output label.
        ipsiTo: how ipsi/contra is mapped to L/R.
            if provided, searches for columns ending with '_ipsi' and '_contra' and maps '_ipsi' indices to  
            the hemisphere specified by ipsiTo ('L' or 'R'). If not provided, assumes columns end with '_L' and '_R'.
        
    Returns:
        df_glasser: mean values per region for the glasser atlas.
       
    """
    import pandas as pd
    import numpy as np
    
    hemi_col = 'SHemi' # could also be SHemi (for short name)
    
    if surf == 'fsLR-32k':
        glasser_df = pd.read_csv("/host/verges/tank/data/daniel/parcellations/glasser/glasser-360_conte69.csv", header=None, names=["glasser"]) # index is vertex num, value is region number
        nvtx = 32492
    elif surf == 'fsLR-5k':
        glasser_df = pd.read_csv("/host/verges/tank/data/daniel/parcellations/glasser/glasser-360_fsLR-5k.csv", header=None, names=["glasser"]) # index is vertex num, value is region number
        nvtx = 4842
    else:
        raise ValueError("[apply_glasser] Invalid surf value. Choose 'fsLR-32k' or 'fsLR-5k'.")
    
    assert df.shape[1] == nvtx*2, f"[apply_glasser] Input dataframe has {df.shape[1]} columns. Expected {nvtx*2} columns for surf {surf}."
    df_relbl = relabel_vertex_cols(df, ipsiTo, split = False, verbose = verbose) # remove '_L' or '_R'/'_ipsi' or '_contra' suffixes from column names, and convert to integer indices

    # map vertex index -> region index (glasser region per vertex)
    vtx_idx = df_relbl.columns.astype(int)
    region_idxs = glasser_df['glasser'].values[vtx_idx]  # length == n_columns

    # load region details and build base name lookup
    glasser_details = pd.read_csv("/host/verges/tank/data/daniel/parcellations/glasser/glasser_details.csv")
    glasser_details.columns = [str(c).strip().replace('\ufeff', '') for c in glasser_details.columns]
    if 'idx_h' not in glasser_details.columns:
        raise KeyError(f"'unique_idx' not found in glasser_details columns. Available columns: {glasser_details.columns.tolist()}")

    if labelType is None or labelType == 'glasser_int':
        det_col = 'idx_h'
    else:
        det_col = 'idx_h'
        if labelType.lower() == 'glasser_name_short':
            det_col = 'SName'
        elif labelType.lower() == 'glasser_name_long':
            det_col = 'LName'
        elif labelType.lower() == 'lobe':
            det_col = 'SLobe'
        elif labelType.lower() == 'longlobe':
            det_col = 'LLobe'
        else:
            print("[apply_glasser] Warning. Invalid labelType value, using default. Choose from 'glasser_int', 'glasser_name_short', 'glasser_name_long', 'lobe', or 'lobelong'.")

    names_map = glasser_details.set_index('unique_idx')[det_col].to_dict()
        

    # determine hemisphere of each vertex from original vertex index
    # nvtx is number of vertices per hemisphere (set above)
    hemi_raw = np.where(vtx_idx < nvtx, 'L', 'R')

    # choose suffix mapping: plain L/R or ipsi/contra if requested
    if addHemiLbl:
        if ipsiTo is None:
            hemi_suffix_map = {'L': 'L', 'R': 'R'}
        else:
            if ipsiTo.upper() == 'L':
                hemi_suffix_map = {'L': 'ipsi', 'R': 'contra'}
            elif ipsiTo.upper() == 'R':
                hemi_suffix_map = {'L': 'contra', 'R': 'ipsi'}
            else:
                raise ValueError("[apply_glasser] Invalid ipsiTo value. Choose 'L' or 'R'.")
        final_names = [f"{names_map.get(r, r)}_{hemi_suffix_map[h]}" for r, h in zip(region_idxs, hemi_raw)]
    else:
        final_names = [f"{names_map.get(r, r)}" for r in region_idxs]

    df_relbl.columns = final_names

    return df_relbl

def apply_DK25(df, surf, labelType = None, addHemiLbl = True, ipsiTo=None, verbose = False):
    """
    Parcellation of hippocampus into 25 regions. https://github.com/jordandekraker/hippomaps/tree/master/hippomaps/resources/parc-DeKraker25
    DK25: DeKraker25

    Input:
        df
        surf: str
            Options:
            den-0p5mm
            den-1mm
            den-2mm

        
        labelType
        addHemiLbl
        ipsiTo
        verbose: bool
            If True, print progress messages.
    
    Returns:
        df_parc: 
    """
    import pandas as pd
    import nibabel as nb
    import numpy as np
    
    if surf not in ['den-0p5mm', 'den-1mm', 'den-2mm']:
        if surf in ['0p5mm', '1mm', '2mm']:
            surf = f'den-{surf}'
        else:
            raise ValueError("[apply_DK25] Invalid surf value. Choose from 'den-0p5mm', 'den-1mm', 'den-2mm'.")
    
    #sub-bigbrain_hemi-R_label-dentate_den-0p5mm_DeKraker25.label.gii
    dk25_rt = '/host/verges/tank/data/daniel/parcellations/DeKraker25/'
    lbl_file = f"sub-bigbrain_hemi-R_label-hipp_{surf}_DeKraker25.label.gii"

    lbl = nb.load(f"{dk25_rt}/{lbl_file}").darrays[0].data
    
    if verbose:
        print(f"[apply_DK25] Label file has length {len(lbl)} (unique: {len(set(lbl))}). Path: {lbl_file}")

    df_l, df_r = relabel_vertex_cols(df, ipsiTo=None, verbose = verbose, split = True)  # strip suffixes from column names   
    
    # apply labels
    df_l.columns = lbl[df_l.columns.astype(int)]
    df_r.columns = lbl[df_r.columns.astype(int)]
   
    
    # index to appropriate labelType
    if labelType is None or labelType == 'idx': # keep index
        pass
    else:
        dk25_details = pd.read_csv(f"{dk25_rt}/DK25_details.csv")
        label_col = labelType.lower()
        
        if label_col not in ['label']: # add additional columns in DK25_details.csv is needed
            raise ValueError("[apply_DK25] Invalid labelType value. Currently only 'label' is supported.")
        else: 
            df_l.columns = df_l.columns.map(
                lambda x: dk25_details.loc[dk25_details['idx'] == x, label_col].values[0]
                if x in dk25_details['idx'].values else x
            )
            df_r.columns = df_r.columns.map(
                lambda x: dk25_details.loc[dk25_details['idx'] == x, label_col].values[0]
                if x in dk25_details['idx'].values else x
            )

    if addHemiLbl:
        if ipsiTo:
            if ipsiTo.upper() == 'L':
                hemi_l = 'ipsi'
                hemi_r = 'contra'
            elif ipsiTo.upper() == 'R':
                hemi_l = 'contra'
                hemi_r = 'ipsi'
        else:
            hemi_l = 'L'
            hemi_r = 'R'
        
    df_l.columns = [f"{col:g}_{hemi_l}" if isinstance(col, (int, float, np.integer, np.floating)) else f"{col}_{hemi_l}" for col in df_l.columns]
    df_r.columns = [f"{col:g}_{hemi_r}" if isinstance(col, (int, float, np.integer, np.floating)) else f"{col}_{hemi_r}" for col in df_r.columns]
        
    #print(f"Num unique (L, R): ({len(set(df_l))}, {len(set(df_r))})")
    
    df_parc = pd.concat([df_l, df_r], axis=1)
    return df_parc

def addHemifromParc(df, parc):
    """
    given a parcellation index, add hemisphere label to index.

    input:
        df:
            dataframe to relabel
        parc:
            name of parcellation method for data in df

    output;
        df_out: appended hemisphere to column name
    """
    import pandas as pd

    return df_out

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

def plotMatrices(dl, df_keys, 
                 name_append=False, sessions = None, save_pth=None, min_stat = -4, max_stat = 4, test=False):
    """
    Plot matrix visualizations for map values from corresponding study

    dl: 
        dictionary list with paired items from different studies
    df_keys: lst
        keys in the dictionary items to plot (e.g., 'map_smth')
    
    name_append: bool
        if true, adds key name to save file
    sessions: (list of ints)
        if provided, will plot different sessions next to eachother rather than different studies
    save_pth:
        if provided, save the plots to this path instead of showing them interactively
    min_stat, max_stat: int, int
        applied only if the key contains either '_z_' or '_w_': min and max values for color scale
    test: bool
        if true, runs in test mode, applying parcellations to 3 random items and appends 'TEST' to outputs
    """
    import matplotlib.pyplot as plot
    from matplotlib import gridspec
    import numpy as np
    import datetime
    import os

    skip_idx = []
    counter = 0
    
    if type(df_keys) == str:
        df_keys = [df_keys] 

    print(f"Plotting matrices for {list(df_keys)}...")
    
    if test:
        idx_len = 3
        idx_rdm = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
        idx_other_rdm = [get_pair(dl, idx = i, mtch=['region', 'surf', 'label', 'feature', 'smth']) for i in idx_rdm]
        print(f"TEST MODE. Applying parcellations to {idx_len} random items in dl: indices {idx_rdm}.")
        rdm_indices = idx_rdm + idx_other_rdm
        dl = [dl[i] for i in rdm_indices]

    # If dl is a single dictionary (not a list), wrap it in a list
    if isinstance(dl, dict):
        print("[plotMatrices] WARNING: dl is a single dictionary, wrapping in a list.")
        dl = [dl]

    counter = 0
    for idx, item in enumerate(dl):
        counter += 1
        if counter % 10 == 0:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : Processed {counter} of {len(dl)}... ")
        

        if idx in skip_idx:
            continue
        skip_idx.append(idx)
        if sessions:
            if item.get('sesNum', None) not in sessions or item.get('sesNum', None) not in sessions:
                continue
            
        counter = counter+1
        
        idx_other = get_pair(dl, idx = idx, mtch=['region', 'surf', 'label', 'feature', 'smth'], skip_idx=skip_idx)
        
        if test:
            index = rdm_indices[idx]
            index_other = rdm_indices[idx_other]
        else:
            index = idx
            index_other = idx_other

        skip_idx.append(idx_other)
        
        if type(idx_other) == list:
            # get the indices whose session number is in 'sessions'. Add also the index 'idx'. 
            # If not in 'sessions' then, add to skip idx
            matches = []
            
            for indices in [idx, *idx_other]:
                itm = dl[indices]
                sesNum = itm.get('sesNum', None)
                if sesNum in sessions:
                    matches.append(indices)
            if len(matches) == 2:
                # sort matches by the ses num of that index
                matches = sorted(matches, key=lambda x: dl[x].get('sesNum', float('inf')))
                idx = matches[0]
                idx_other = matches[1]
                
                item = dl[idx]
                item_other = dl[idx_other]
            else:
                print(f"[plotMatrices] WARNING: More than 2 matches found for index {index} with keys ['region', 'surf', 'label', 'feature', 'smth'] and sessions {sessions}. Skipping.")
                continue

        if idx_other is None:
            item_txt = printItemMetadata(item, return_txt=True)
            print(f"\tWARNING. No matching index found for: {item_txt} (idx: {index}).\nSkipping.")
            continue
        skip_idx.append(idx_other)
        
        item_other = dl[idx_other]
        if item_other is None:
            item_txt = printItemMetadata(item, return_txt=True)
            print(f"\tWARNING. Item other is None: {item_txt} (idx: {index}).\nSkipping.")
            continue
        
        if sessions is None:
            if item['study'] == 'MICs':
                idx_one = idx
                idx_two = idx_other

                item_one = item
                item_two = item_other
            else:
                idx_one = idx_other
                idx_two = idx

                item_one = item_other
                item_two = item
        else:
            
            idx_one = idx
            idx_two = idx_other

            item_one = item
            item_two = item_other

            ses_one = item_one.get('sesNum', None)
            ses_two = item_two.get('sesNum', None)

        if item_one is None and item_two is None:
            item_one_txt = printItemMetadata(item_one, idx=idx_one, return_txt=True)
            item_two_txt = printItemMetadata(item_two, idx=idx_two, return_txt=True)
            print(f"\tWARNING. Both items are None (item one: {item_one_txt}, item two: {item_two_txt}).\nSkipping.")
            continue
        elif item_one is None:
            item_one_txt = printItemMetadata(item_one, idx=idx_one, return_txt=True)
            print(f"\tWARNING. Item_one is None: {item_one_txt}.\nSkipping.")
            continue
        elif item_two is None:
            item_two_txt = printItemMetadata(item_one, idx=idx_one, return_txt=True)
            print(f"\tWARNING. Item_two is None: {item_two_txt}.\nSkipping.")
            continue
        
        if sessions is None:
            print(f"\t[idx 3T, 7T: {idx_one}, {idx_two}] {printItemMetadata(item_one, return_txt = True, clean = True, printStudy = False)}")
        else:
            print(f"\t[ses 1, 2: {idx_one} ({ses_one}), {idx_two} ({ses_two})] {printItemMetadata(item_one, return_txt = True, clean = True, printStudy = False)}")
        
        for key in df_keys:

            if item_one.get('study', None) and item_two.get('study', None):
                if test:
                    title_one = f"{key} {item_one['study']} [idx: {rdm_indices[idx_one]}]"
                    title_two = f"{key} {item_two['study']} [idx: {rdm_indices[idx_two]}]"
                else:
                    title_one = f"{key} {item_one['study']} [idx: {idx_one}]"
                    title_two = f"{key} {item_two['study']} [idx: {idx_two}]"
            else:
                if test:
                    title_one = f"{key} SES num: {ses_one} [idx: {rdm_indices[idx_one]}]"
                    title_two = f"{key} SES num: {ses_two} [idx: {rdm_indices[idx_two]}]"
                else:
                    title_one = f"{key} SES num: {ses_one} [idx: {idx_one}]"
                    title_two = f"{key} SES num: {ses_two} [idx: {idx_two}]"
            
            feature_one = item_one['feature']
            feature_two = item_two['feature']
            
            try:
                df_one = item_one[key]
                if type(df_one) is str:
                    pth = df_one 
                    df_one = loadPickle(pth, verbose = False) 
            except KeyError:
                print(f"\t\tWARNING: Could not access key '{key}' for item at index {idx_one}. Skipping.")
                #print_dict(dl, idx = [idx_one])
                print(f"\t\t{'-'*50}")
                continue
            except Exception as e:
                print(f"\t\tERROR: Unexpected error while accessing key '{key}' for item at index {idx_one}: {e}")
                #print_dict(dl, idx=[idx_one])
                print(f"\t\t{'-'*50}")
                continue
            
            try:
                df_two = item_two[key]
                if type(df_two) is str:
                    pth = df_two
                    df_two = loadPickle(pth, verbose = False) 
            except KeyError:
                print(f"\t\tWARNING: Could not access key '{key}' for item at index {idx_two}. Skipping.")
                #print_dict(dl, idx = [idx_two])
                print(f"\t\t{'-'*50}")
                continue
            except Exception as e:
                print(f"\t\tERROR: Unexpected error while accessing key '{key}' for item at index {idx_one}: {e}")
                #print_dict(dl, idx=[idx_one])
                print(f"\t\t{'-'*50}")
                continue
            
            if df_one is None and df_two is None:
                item_one_txt = printItemMetadata(item_one, idx=idx_one, return_txt=False)
                item_two_txt = printItemMetadata(item_two, idx=idx_two, return_txt=False)
                print(f"\t\tWARNING. Missing key '{key}'. Skipping {item_one_txt} and {item_two_txt}\n")
                print('-'*50)
                continue
            elif df_one is None:
                item_one_txt = printItemMetadata(item_one, idx=idx_one, return_txt=True)
                print(f"\t\tWARNING. Missing key '{key}' for {item_one_txt}. Skipping.\n")
                print('-'*50)
                continue
            elif df_two is None:
                item_two_txt = printItemMetadata(item_two, idx=idx_two, return_txt=True)
                print(f"\t\tWARNING. Missing key '{key}' for {item_two_txt}. Skipping.\n")
                print('-'*50)
                continue
            else:
                print(f"\t\tPlotting key: {key}...")
            # determine min and max values across both matrices for consistent color scaling
            assert feature_one == feature_two, f"Features do not match: {feature_one}, {feature_two}"
            assert item_one['region'] == item_two['region'], f"Regions do not match: {item_one['region']}, {item_two['region']}"
            assert item_one['surf'] == item_two['surf'], f"Surfaces do not match: {item_one['surf']}, {item_two['surf']}"
            assert item_one['label'] == item_two['label'], f"Labels do not match: {item_one['label']}, {item_two['label']}"
            assert item_one['smth'] == item_two['smth'], f"Smoothing kernels do not match: {item_one['smth']}, {item_two['smth']}"
        
            if "_z" in key or "_w" in key:
                cmap = "seismic"
                min_val = min_stat
                max_val = max_stat
            else:
                cmap = 'inferno'
                if feature_one.lower() == "thickness":
                    min_val = 0
                    max_val = 4
                    cmap = 'Blues'
                elif feature_one.lower() == "flair":
                    min_val = -500
                    max_val = 500
                    cmap = "seismic"
                elif feature_one.lower() == "t1map":
                    min_val = 1000
                    max_val = 2800
                    cmap = "inferno"
                elif feature_one.lower() == "fa":
                    min_val = 0
                    max_val = 1
                    cmap="Blues"
                elif feature_one.lower() == "adc": # units: mm2/s
                    min_val = 0
                    max_val = 0.0025
                    cmap = "Blues"
                else:
                    min_val = min(np.percentile(df_two.values, 95), np.percentile(df_one.values, 95))
                    max_val = max(np.percentile(df_two.values, 5), np.percentile(df_one.values, 5))

            # Create a grid layout with space for the colorbar
            fig = plot.figure(figsize=(30, 25))
            spec = gridspec.GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.43)

            # Create subplots
            ax1 = fig.add_subplot(spec[0])
            ax2 = fig.add_subplot(spec[2])

            # Define x-axis label
            if item_one.get('parcellation', None) is not None and 'parc' in key:
                x_axisLbl = item_one.get('parcellation', None).upper()
            else:
                x_axisLbl = 'Vertex'

            # Plot the matrices
            visMatrix(df_one, feature=feature_one, title=title_one, 
                    show_index=True, ax=ax1, x_axisLbl = x_axisLbl,
                    min_val=min_val, max_val=max_val, cmap=cmap, nan_side="left")
            visMatrix(df_two, feature=feature_two, title=title_two, 
                    show_index=True, ax=ax2, x_axisLbl = x_axisLbl,
                    min_val=min_val, max_val=max_val, cmap=cmap, nan_side="right")

            # Add a colorbar between the plots
            cmap_title = feature_one

            if "_z_" in df_keys:
                cmap_title = f"Z-score [{cmap_title}]"
            elif "_w_" in df_keys:
                cmap_title = f"W-score [{cmap_title}]"
            else:
                if feature_one.upper() == "ADC":
                    cmap_title = "ADC (mm²/s)"
                elif feature_one.upper() == "T1MAP":
                    cmap_title = "T1 (ms)"
            
            cbar_ax = fig.add_subplot(spec[1])
            norm = plot.Normalize(vmin=min_val, vmax=max_val)
            cbar = plot.colorbar(plot.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label(cmap_title, fontsize=20, labelpad=0)
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.tick_params(axis='x', direction='in', labelsize=20)

            # Add a common title
            region = item_one['region']
            surface = item_one['surf']
            label = item_one['label']
            smth = item_one['smth']
            if sessions:
                fig.suptitle(f"{region}: {feature_one}, {surface}, {label}, {smth}mm (SES: {ses_one}, {ses_two})", fontsize=25, y=0.9)
            else:
                fig.suptitle(f"{region}: {feature_one}, {surface}, {label}, {smth}mm", fontsize=30, y=0.9)

            if save_pth is not None:
                if name_append:
                    if sessions is not None:
                        save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_key-{key}_ses-{ses_one}{ses_two}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                    else:
                        save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_key-{key}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                elif sessions is not None:
                    save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                else:
                    save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                
                if test:
                    save_name = f"TEST_{save_name}"

                fig_pth = f"{save_pth}/{save_name}.png"
                fig.savefig(fig_pth, dpi=300, bbox_inches='tight')
                file_size = os.path.getsize(fig_pth) / (1024 * 1024)  # size in MB
                print(f"\tSaved ({file_size:0.1f} MB): {fig_pth}")
                plot.close(fig)


def visMatrix(df, feature="Map Value", title=None, min_val=None, max_val=None, x_axisLbl = None,
              cmap='seismic', show_index=False, ax=None, nan_color='green', nan_side="right"):
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
    
    x_axisLbl: str, optional
        Label for the x-axis. If None, use default: 'Vertex'
    min_val : float, optional
        Minimum value for colormap scaling. If None, uses the minimum of the data.
    max_val : float, optional
        Maximum value for colormap scaling. If None, uses the maximum of the data.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use for visualization (default is 'seismic').
    show_index : bool, optional
        If True, displays DataFrame index labels on the y-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure and axes.
    
    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes containing the visualization, depending on `return_fig`.
    """
    
    import numpy as np
    from matplotlib import use
    import matplotlib.pyplot as plt
    use('Agg')

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
        fig_length = max(6, min(0.1, 0.1 * data.shape[0]))
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

    if x_axisLbl is None:
        ax.set_xlabel("Vertex", fontsize=15)
    else:
        ax.set_xlabel(x_axisLbl, fontsize=15)
    
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


    plt.close(fig)
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
    import datetime
    
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
        time_stmp = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
        output_pdf = os.path.join(output, f"{base_name}_{time_stmp}.pdf")

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
            print(f"\tPDF created: {output_pdf}")

def sortCols(df):
    """
    Sort DataFrame columns. Rules: 
        All _L columns first (sorted by number), then all _R columns (sorted by number).
        If ends with _contra, put before _ipsi (same logic).
        If neither, put at the end.

    Input:
        df: DataFrame with columns to sort

    Output:
        df_sorted: DataFrame with sorted columns
    """
    import re
    import pandas as pd

    def col_type(col):
        col = str(col)
        if col.endswith('_L'):
            return 'L'
        elif col.endswith('_R'):
            return 'R'
        elif col.endswith('_contra'):
            return 'contra'
        elif col.endswith('_ipsi'):
            return 'ipsi'
        else:
            return 'other'

    def col_num(col):
        col = str(col)
        match = re.match(r"(\d+)", col)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    # Group columns
    L_cols = [col for col in df.columns if col_type(col) == 'L']
    R_cols = [col for col in df.columns if col_type(col) == 'R']
    contra_cols = [col for col in df.columns if col_type(col) == 'contra']
    ipsi_cols = [col for col in df.columns if col_type(col) == 'ipsi']
    other_cols = [col for col in df.columns if col_type(col) == 'other']

    # Sort each group by number
    L_cols = sorted(L_cols, key=col_num)
    R_cols = sorted(R_cols, key=col_num)
    contra_cols = sorted(contra_cols, key=col_num)
    ipsi_cols = sorted(ipsi_cols, key=col_num)
    other_cols = sorted(other_cols)

    # Order: L, contra, R, ipsi, other
    sorted_cols = L_cols + ipsi_cols + R_cols + contra_cols + other_cols
    df_sorted = df[sorted_cols]
    return df_sorted

def plot_ridgeLine(df_a, df_b, lbl_a, lbl_b, title, 
                   parc = None, stat=None, hline = None, marks = False, 
                   hline_idx = None, pad_top_rows=2,
                   offset = 3, spacing=None, alpha = 0.3):
    """
    Create ridgeplot-like graph plotting two distributions over common vertices for each participant.

    Input:
        df_a, df_b:         DataFrames with identical columns and identical row indices.
                                NOTE. Numerical indeces should be sorted 
        lbl_a, lbl_b:       Labels for the two groups
        title:              Title for the plot
        
        <optionals>
        parc:               Indicate how the surface has been parcellated. If None, assumes vertex names have suffix '_L', '_R' 
            Options: 'glasser', None
        stat:  str             If parcellation is provided, this  provided, this variable is added to the x-axis label 
            (made for when multiple vertices are summarised with a statistic for a single value per parcel)
        hline: <int or None> if provided, draw a horizontal line at this y-value. 
        marks:              Whether to use marks instead of lines
        pad_top_rows: int   Adds empty top rows to increase spacing between title and plot    
        offset:             Vertical distance between plots
        spacing: <int or None>  If provided, use this list to set the y-tick positions.
    
    Output:
        axis object
    """

    import numpy as np
    import pandas as pd
    from matplotlib import use
    import matplotlib.pyplot as plt
    use('Agg')  # Use a non-interactive backend, prevents memory build up.
    # see if ipsiContra. If not, assume L/R
    
    n = df_a.shape[0]
    
    # sort columns
    df_a_sort = sortCols(df_a)
    df_b_sort = sortCols(df_b)

    # choose a random column, check if ends with '_ipsi' or '_contra'
    rdm_col = df_a_sort.columns[np.random.randint(0, df_a_sort.shape[1])].lower()
    if rdm_col.endswith('_ipsi') or rdm_col.endswith('_contra'):
        ipsi_contra = True
    else:
        ipsi_contra = True

    # Reverse the order of rows so the top row of the plot corresponds to the top row of the DataFrame
    df_a_sort = df_a_sort.iloc[::-1]
    df_b_sort = df_b_sort.iloc[::-1]
    
    # Optionally prepend a few blank rows at the top to increase space between title and first real row.
    # These rows will not be plotted; they simply create extra vertical gap.
    original_n = df_a_sort.shape[0]
    if pad_top_rows and pad_top_rows > 0:
        pad_idx = [f"_pad_{i}" for i in range(pad_top_rows)]
        pad_df = pd.DataFrame(np.nan, index=pad_idx, columns=df_a_sort.columns, dtype="float64")
        df_a_sort = pd.concat([pad_df, df_a_sort], axis=0)
        df_b_sort = pd.concat([pad_df, df_b_sort], axis=0)

    vertices = df_a_sort.columns
    n_rows = df_a_sort.shape[0]

    fig_length = min(50, 0.75 * n_rows) # max height of 50
    fig_width = 55
    fig, ax = plt.subplots(figsize=(fig_width, fig_length))
    
    # relative font sizes
    scale = fig_width / 100
    sizes = {
        'linewdth': 2,
        'mrkrsize': 4,
        'title': 100,
        'legend': 70,
        'y_tick': 50,
        'x_tick': 65,
        'x_lbl': 90,
        'annot': 80,
    }

    if spacing is None:
        # compute typical amplitude per participant and choose spacing > max amplitude
        per_row_amp = (df_a_sort.max(axis=1) - df_a_sort.min(axis=1)).abs().tolist() + \
                      (df_b_sort.max(axis=1) - df_b_sort.min(axis=1)).abs().tolist()
        max_amp = max(per_row_amp) if len(per_row_amp) > 0 else float(offset)
        spacing_val = max(float(offset), float(max_amp) * 1.2 + 1.0)  # margin
    else:
        spacing_val = float(spacing)

    x = np.arange(len(vertices))
    # iterate through rows but skip pad rows (they are just for spacing)
    for i in range(n_rows):
        idx_label = df_a_sort.index[i]
        if str(idx_label).startswith("_pad_"):  # don't plot pad rows
            continue
        baseline = i * spacing_val

        y_a = df_a_sort.iloc[i].values + baseline
        y_b = df_b_sort.iloc[i].values + baseline
        
        if marks:
            ax.scatter(x, y_a, color='red', alpha=alpha, s=sizes['mrkrsize'], label=lbl_a if i == 0 else "")
            ax.scatter(x, y_b, color='blue', alpha=alpha, s=sizes['mrkrsize'], label=lbl_b if i == 0 else "")
        else:
            ax.plot(x, y_a, color='red', alpha=alpha, linewidth = sizes['linewdth'], label=lbl_a if i == 0 else "")
            ax.plot(x, y_b, color='blue', alpha=alpha, linewidth = sizes['linewdth'], label=lbl_b if i == 0 else "")
        
        if hline is not None:
            ax.axhline(y=baseline + hline, color='black', linestyle='--', linewidth=1, alpha=1)

    if parc is None:
        split_idx = next((k for k, col in enumerate(vertices) if '_R' in col), None)
        if split_idx is None: # assume ipsi contra labels instead
            split_idx = next((k for k, col in enumerate(vertices) if '_ipsi' in col), None)
            if split_idx is not None:
                ipsi_contra = True
            else:
                # assume split idx is half way through
                split_idx = len(vertices) // 2
                print("[plot_ridgeLine] WARNING: Could not determine hemisphere split from column names. Assuming halfway split.")
                
    elif parc.lower() == "glasser" or parc.lower() == "glsr":
        split_idx = 181
    elif parc.lower() == "dk25" or parc.lower() == "dk":
        split_idx = 25
    else:
        ValueError("[plot_ridgeLine] Invalid parc value. Choose 'glasser', 'DK25' or None.")
    
    # Set title
    fig.suptitle(title, fontsize=int(sizes['title'] * scale), y=0.995)

    # Legend (lay entries out horizontally next to title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = max(1, len(labels))  # typically 2 (lbl_a, lbl_b); adjust if more labels appear
        legend_x = 0.85  # tune to move legend horizontally; closer to 1.0 moves it to the right edge
        legend_y = 0.995  # just under the top of the figure (near title)
        fig.legend(handles, labels,
                   loc='upper right',
                   bbox_to_anchor=(legend_x, legend_y),
                   ncol=ncol,
                   frameon=False,
                   fontsize=int(sizes['legend'] * scale),
                   markerscale=max(1, sizes['mrkrsize'] * scale))
        
    """ place legend just outside the right side with the same visual spacing
    ax.legend(fontsize=sizes['legend']*scale,
              markerscale=max(1, sizes['mrkrsize']*2),
              loc='upper left',
              bbox_to_anchor=(1.0 + pad_frac, 1.0),
              borderaxespad=0)
    """

    # y ticks
    # y ticks: place at participant baselines only (exclude pad rows)
    participant_idx = [i for i, lbl in enumerate(df_a_sort.index) if not str(lbl).startswith("_pad_")]
    ytick_positions = [i * spacing_val for i in participant_idx]
    ax.set_yticks(ytick_positions)
    labels = ax.set_yticklabels(df_a_sort.index[participant_idx].astype(str), fontsize=int(sizes['y_tick'] * scale))
    
    for lbl in labels: # ensure vertical alignment is centered for all tick label Text objects
        lbl.set_va('center')
    
    ax.tick_params(axis='y', which='major', pad=max(2, int(4 * scale))) # horizontal padding so labels don't touch axis

    approx_char_width_px = (sizes['x_tick'] * scale) * 0.6
    pad_px = approx_char_width_px * 5  # 5 characters worth of space

    # figure physical width in pixels
    fig_w_in, _ = fig.get_size_inches()
    dpi = fig.dpi if hasattr(fig, "dpi") else plt.rcParams.get("figure.dpi", 100)
    fig_w_px = fig_w_in * dpi

    # fraction of figure width to reserve on each side
    pad_frac = pad_px / fig_w_px

    # convert fraction -> data units (x-axis runs from 0 .. len(vertices)-1)
    x_min = 0
    x_max = max(0, len(vertices) - 1)
    data_span = x_max - x_min if x_max > x_min else 1.0
    pad_data = pad_frac * data_span
    
    ax.set_xlim(x_min - pad_data, x_max + pad_data) # set x-limits so data starts/ends with the requested padding
    ax.margins(x=0)

    # xtick placement/labels (keep using vertex labels at three positions)
    xticks = np.linspace(0, len(vertices) - 1, 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels([vertices[int(j)] for j in xticks], fontsize=sizes['x_tick']*scale, rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=sizes['x_tick']*scale)
    
    if parc is not None:
        if stat is not None:
            ax.set_xlabel(f"{parc.upper()} ({stat})", fontsize=sizes['x_lbl']*scale)
        else:
            ax.set_xlabel(parc, fontsize=sizes['x_lbl']*scale)
    else:
        ax.set_xlabel("Vertex", fontsize=sizes['x_lbl']*scale)

    # Hemisphere labels
    x0, x1 = ax.get_xlim()
    data_span_eff = x1 - x0 if (x1 - x0) != 0 else 1.0
    left_data = (split_idx / 2.0)
    right_data = ((split_idx + len(vertices)) / 2.0)
    left_frac = (left_data - x0) / data_span_eff
    right_frac = (right_data - x0) / data_span_eff
    
    # clamp to avoid text going off-figure
    left_frac = min(1.0, max(0.0, left_frac))
    right_frac = min(1.0, max(0.0, right_frac))
    
    # axes fraction for vertical placement (negative = below axis)
    y_ax_frac = -0.03  # move labels away from y-axis
    label_fs = int(sizes['annot'] * scale)
    va_setting = 'top'  # anchor the top of the text at the y coordinate so it sits below the axis
    if not ipsi_contra:
        ax.text(left_frac, y_ax_frac, "Left",
                fontsize=label_fs, ha='center', va=va_setting, transform=ax.transAxes)
        ax.text(right_frac, y_ax_frac, "Right",
                fontsize=label_fs, ha='center', va=va_setting, transform=ax.transAxes)
    else:
        ax.text(left_frac, y_ax_frac, "Contralateral",
                fontsize=label_fs, ha='center', va=va_setting, transform=ax.transAxes)
        ax.text(right_frac, y_ax_frac, "Ipsilateral",
                fontsize=label_fs, ha='center', va=va_setting, transform=ax.transAxes)

    # add vertical lines
    row_offsets = (np.arange(n_rows) * spacing_val)[:, None]  # shape (n,1)
    
    y_a_all = (np.nan_to_num(df_a_sort.values.astype(float), nan=np.nan) + row_offsets).reshape(-1)
    y_b_all = (np.nan_to_num(df_b_sort.values.astype(float), nan=np.nan) + row_offsets).reshape(-1)
    y_all = np.hstack([y_a_all, y_b_all])
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = max(0.5, 0.1 * spacing_val)  # small padding so lines don't touch markers
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.axvline(x=split_idx, color='black', linestyle='--', linewidth=max(1, sizes['mrkrsize'] * 0.75), alpha=0.4)
    
    if hline_idx is not None: # plot hlines at each provided index
         for hx in hline_idx:
            ax.axvline(x=hx, color='gray', linestyle='--', linewidth=max(0.5, sizes['mrkrsize'] * 0.75), alpha=0.3)
    
    # remove y-axis line
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('none')

    ax.margins(x=fig_width*0.01, y=fig_width*0.01)
    fig.tight_layout(rect=[0, 0, 1, 1]) # add padding
    fig.subplots_adjust(top=0.97) # lower values move title up more

    return fig

def plotLine(dl, df_keys = ['df_maps'], marks=True, 
            parc=[None], stat =[None],
            alpha = 0.02, spacing = 1,
            hlines = None, name_append=None, save_pth=None, 
            verbose = False, test = False):
    
    """
    Plot ridgeline graphs to compare maps between corrsponding dfs (eg., 3T vs 7T).
    Note. All values are rescaled participant-wise: (x - mdn(x)) / IQR(x) 


    Input:
        dl:           list of dict items with dataframes to plot
        df_key: lst        keys in dict items for dataframe to plot
        marks:          whether to use marks instead of lines
        parc: lst          indicate if and how the surface has been parcellated. If None, assumes vertex with suffixes '_L', '_R' or '_ipsi', '_contra'.
        stat: lst       if parcellation is provided, this variable is added to the x-axis label (made for when vertices are summarised with a statistic for a single value per parcel) 
        alpha: int         transparency of lines/marks
        offset: int        vertical distance between each individual's plot
        
        hline_idx: lst  list of list of indices to draw horitzontal lines at
        name_append:    string to append to saved file names
        save_pth:       path to save figures. If None, will not save.
        verbose:        whether to print item metadata  
        test:           whether to run in test mode (only first 2 items in dl)
    
    Output:
        saves plot to path    
    """
    import matplotlib.pyplot as plt
    import datetime
    import numpy as np
    import os

    skip_idx = []
    counter = 0
    start_time = datetime.datetime.now()
    print(f"[{start_time}] Plotting ridgeline plots for {list(df_keys)}...")
    
    if type(df_keys) is str:
        df_keys = [df_keys]
    if type(parc) is str:
        parc = [parc]
    if type(stat) is str:
        stat = [stat]

    if test:
        idx_len = 3
        idx_rdm = np.random.choice(len(dl), size=idx_len, replace=False).tolist()  # randomly choose index
        idx_other_rdm = [get_pair(dl, idx=i, mtch=['region', 'surf', 'label', 'feature', 'smth']) for i in idx_rdm]
        # flatten idx_other_rdm if any are lists (get_pair can return a list)
        idx_other_flat = []
        for idx in idx_other_rdm:
            if isinstance(idx, list):
                idx_other_flat.extend(idx)
            elif idx is not None:
                idx_other_flat.append(idx)
        rdm_indices = idx_rdm + idx_other_flat
        dl = [dl[i] for i in rdm_indices]
        print(f"\tTEST MODE. Plotting {idx_len} random items and their pairs: {rdm_indices}")
    
    for idx in range(len(dl)):

        if idx in skip_idx:
            continue
        counter += 1

        skip_idx.append(idx)
        idx_other = get_pair(dl, idx, mtch = ['region', 'feature', 'label', 'surf', 'smth'], skip_idx=skip_idx)
        skip_idx.append(idx_other)
        if idx_other is None:
            print(f"\tNo matching index found. Skipping.")
            continue
        
        # determine which study is tT and which is sT
        idx_tT, idx_sT = determineStudy(dl, idx = idx, idx_other = idx_other, study_key = 'study')
        item_tT = dl[idx_tT]
        item_sT = dl[idx_sT]
        
        print(f"\t[idx 3T, 7T: {idx_tT}, {idx_sT}] {printItemMetadata(item_tT, return_txt = True, clean = True, printStudy = False)}")
       
        # extract df
        for df_key, p, s, hlines_idx in zip(df_keys, parc, stat, hlines if hlines is not None else [None]*len(df_keys)):
            df_tT = item_tT.get(df_key, None)
            df_sT = item_sT.get(df_key, None)

            if type(df_tT) is str:
                pth = df_tT 
                df_tT = loadPickle(pth, verbose = False)
            if type(df_sT) is str:
                pth = df_sT 
                df_sT = loadPickle(pth, verbose = False)

            if df_tT is None or df_sT is None:
                if verbose:
                    print(f"\t{df_key} is None. Skipping.")
                continue
            elif df_tT.shape[0] == 0 or df_sT.shape[0] == 0:
                if verbose:
                    print(f"\t{df_key} is empty. Skipping.")
                continue
            else:
                print(f"\tPlotting {df_key}")
            
            # ensure that all columns overlap
            cols_tT = set(df_tT.columns) # use as x-axis
            cols_sT = set(df_sT.columns)
            cols_common = list(cols_tT.intersection(cols_sT))
            assert len(cols_common) == len(cols_tT) and len(cols_common) == len(cols_sT), f"Columns do not match between studies: {cols_tT} vs {cols_sT}"
            
            # rename indices to match. Use UID only
            uid_tT = df_tT.index.str.split('_').str[0]
            uid_sT = df_sT.index.str.split('_').str[0]
            
            # ensure that all indices overlap
            idxs_common = list(set(uid_tT).intersection(set(uid_sT)))
            idxs_common = sorted(idxs_common) # sort by UID

            assert len(idxs_common) == len(uid_tT) and len(idxs_common) == len(uid_sT), f"Indices do not match between studies: {set(uid_tT)} vs {set(uid_sT)}"
            
            # set index to UID
            df_tT.index = uid_tT
            df_sT.index = uid_sT
            
            # take only overlapping indices
            df_tT = df_tT.loc[idxs_common, cols_common]
            df_sT = df_sT.loc[idxs_common, cols_common]

            # create title
            region = item_tT.get('region', None)
            feature = item_tT.get('feature', None)
            surface = item_tT.get('surf', None)
            label = item_tT.get('label', None)
            smth = item_tT.get('smth', None)
            title = f"{region}: {feature}, {surface}, {label}, smth-{smth}mm ({df_key})"
            if 'TLE' in df_key:
                title = title + f" [TLE only]"  # TODO CHANGE SO THAT NOT HARD CODED 
            
            hline = None
            if '_z' in df_key.lower():
                title = title + " [Z-score]"
                hline = 0 # value at which to plot a horizontal line for each subject
            elif '_w' in df_key.lower():
                title = title + " [W-score]"
                hline = 0 # value at which to plot a horizontal line for each subject
            elif feature.lower() in ['t1map', 'flair']: # rescale values (participant wise)
                # Robust rescaling: subtract median and divide by IQR (interquartile range)
                df_tT = (df_tT - df_tT.median(axis=1).values[:, None]) / (df_tT.quantile(0.75, axis=1).values - df_tT.quantile(0.25, axis=1).values)[:, None]
                df_sT = (df_sT - df_sT.median(axis=1).values[:, None]) / (df_sT.quantile(0.75, axis=1).values - df_sT.quantile(0.25, axis=1).values)[:, None]
                title = title + " [standardized]"

            # ridge line plot
            fig = plot_ridgeLine(df_tT, df_sT, lbl_a="3T", lbl_b="7T", 
                                hline=hline, marks=marks, title=title, 
                                parc=p, stat=s, alpha=alpha, 
                                hline_idx = hlines_idx, spacing = spacing)

            if save_pth is not None:
                if name_append is not None:
                    save_name = f"{region}_{feature}_{surface}_{label}_smth-{smth}mm_{name_append}_{df_key}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                else:
                    save_name = f"{region}_{feature}_{surface}_{label}_smth-{smth}mm_{df_key}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                if test:
                    save_name = f"TEST_{save_name}"

                fig.savefig(f"{save_pth}/{save_name}.png", dpi=300, bbox_inches='tight', pad_inches = 0.08)
                file_size = os.path.getsize(f"{save_pth}/{save_name}.png") / (1024 * 1024)  # size in MB
                print(f"\t\tSaved ({file_size:0.1f} MB): {save_pth}/{save_name}.png")
                plt.close(fig)

    end_time = datetime.datetime.now()
    print(f"[{end_time}] Complete. Duration: {end_time - start_time}")
    
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

    #lbl_text = {'top': [title if title else '', '','',''], 'bottom': [feature_lbl if feature_lbl else '', '','','']}
    lbl_text = {}
    if ipsiTo is not None and ipsiTo == "L":
        lbl_text.update({
            'left': 'ipsi',
            'right': 'contra'
        })    
    elif ipsiTo is not None and ipsiTo == "R":
        lbl_text.update({
            'left': 'contra',
            'right': 'ipsi'
        })
    else:
        lbl_text.update({
            'left': 'L',
            'right': 'R'
        })

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
            label_text = lbl_text
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
    from matplotlib import use
    use('Agg')  # Use non-interactive backend for compatibility

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
        plt.close()
    else:
        plt.show()
        plt.close()

