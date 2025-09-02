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

def get_vol_pth(root, sub, ses, feature, space="nativepro"):
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

def idToMap(df_demo, studies, dict_demo, specs, verbose=False):
    """
    From demographic info, identify path to--and if needed compute--smoothed map for parameters of interest (surface, label, feature, smoothing kernel) provided in a dictionary item.

    Input:
        df_demo: DataFrame 
            containing demographic information. Each row is one participant at a unique session. 
                Necessary columns: ID, SES
                Optional columns: study (if multiple studies with different root directories). If this is present, then 'studies' must be provided.
        studies: list of dictionary items
            Each dictionary should contain information about the studies to include. Each dict should have the following keys:
                'study': study code (e.g., "3T", "7T")
                'dir_root': root directory of the study
                'dir_deriv': name of derivative folder containing the surface data (e.g., "micapipe", "hippunfold")
                'dir_mp': name of micapipe derivative folder (if applicable)
                'dir_hu': name of hippunfold derivative folder (if applicable)
            If multiple studies are provided, then 'study' column must be present in df_demo to indicate which study each row belongs to.
        dict_demo: dict
            containing column names for demographic information.
                Required keys:
                    'nStudies': whether multiple studies are included (True/False). If this is the case a column 'study' must be present in df_demo.
                    <'ID'> or <'ID[_*study identifier*]'>: column name for subject ID in this study. If multiple studies: multiple ID keys must be present, each with a suffix identifying the study.
                    'SES': column name for session ID
        specs: dict
            containing specifications for the maps to extract and compute. 
                Required keys:
                    'prjDir_root': root directory for saving smoothed maps
                    'prjDir_maps': directory for saving smoothed maps (relative to prjDir_root)
                    
                    'ctx': whether to include cortical analyses (True/False)
                    'surf_ctx': list of surface types and resolutions to include (e.g., ["fsLR-32k", "fsLR-5k"])
                    'lbl_ctx': list of surface labels to include (e.g., ["pial", "white", "midthickness"])
                    'ft_ctx': list of features to include (e.g., ["thickness", "T1map", "FLAIR", "ADC", "FA"])
                    'smth_ctx': list of smoothing kernel sizes in mm to include (e.g., [5, 10, 15])

                    'hipp': whether to include hippocampal analyses (True/False)
                    'surf_ctx': list of surface types and resolutions to include (e.g., ['0p5mm','2mm'])
                    'lbl_hipp': list of hippocampal labels to include (e.g., ["hipp_inner", "hipp_outer", "hipp_midthickness", "dentate_])
                    'ft_hipp': list of hippocampal features to include (e.g., ["thickness", "T1map", "FLAIR", "ADC", "FA"])
                    'smth_hipp': list of hippocampal smoothing kernel sizes in mm to include (e.g., [1, 2, 5])
        

    """
    import os
    import re

    def ctx_maps(out_dir, study, df, idx, sub, ses, surf, ft, smth, lbl, verbose=False):
        """
        Get or compute cortical smoothed maps for a given subject and session, surface, label, feature, and smoothing kernel size.
        """

        print(f"\tsurf: {surf}, label: {lbl}, feature: {ft}, kernel: {smth}mm")
        root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
        study_code = study['study']

        # A. Find path to unsmoothed map
        if ft == "thickness":
            out_pth_L_filename = f"hemi-L_surf-{surf}_label-{ft}"
            out_pth_R_filename = f"hemi-R_surf-{surf}_label-{ft}"
        else:
            out_pth_L_filename = f"hemi-L_surf-{surf}_label-{lbl}_{ft}"
            out_pth_R_filename = f"hemi-R_surf-{surf}_label-{lbl}_{ft}"
                                        
        pth_map_unsmth_L = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_L_filename}.func.gii"
        pth_map_unsmth_R = f"{root_mp}/sub-{sub}/ses-{ses}/maps/sub-{sub}_ses-{ses}_{out_pth_R_filename}.func.gii"
        
        pth_map_unsmth_L = pth_map_unsmth_L if chk_pth(pth_map_unsmth_L) else None
        pth_map_unsmth_R = pth_map_unsmth_R if chk_pth(pth_map_unsmth_R) else None
        
        if pth_map_unsmth_L is None and pth_map_unsmth_R is None:
            print(f"\t\t[WARNING] Unsmoothed map not found for [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check rawdata and/or micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ).\n")
            return df
        elif pth_map_unsmth_L is None:
            print(f"\t\t[WARNING] Unsmoothed map missing for L hemi ONLY [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ).")
        elif pth_map_unsmth_R is None:
            print(f"\t\t[WARNING] Unsmoothed map missing for R hemi ONLY [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ).")    
        else:
            if verbose:
                print(f"\t\tUnsmoothed maps:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
        
        # B. Smooth map and save to project directory
        
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
        
        # Check if smoothed map already exists. If so, do not recompute. Assums would be saved under same naming convention.
        if chk_pth(out_pth_L) and chk_pth(out_pth_R):
            if verbose:
                print(f"\t\tSmoothed maps exists, adding to df: {out_pth_L}\t{out_pth_R}\n")
            df.loc[idx, col_base_L] = out_pth_L
            df.loc[idx, col_base_R] = out_pth_R
            return df
        
        # i. Get path to surface .func files
        surf_L, surf_R = get_surf_pth(
            root=root_mp,
            sub=sub,
            ses=ses,
            surf=surf,
            lbl=lbl
        )
        
        if not chk_pth(surf_L) and not chk_pth(surf_R):
            print(f"\t\t[WARNING] Nativepro surface not found for [{study_code}] {sub}-{ses}. Cannot continue to map smoothing. Check micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_L}\t{surf_R}\n")
            return df
        elif not chk_pth(surf_L):
            print(f"\t\t[WARNING] Nativepro surface missing for L hemi for [{study_code}] {sub}-{ses}. Skipping map smoothing for this hemi. Check micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_L}")
            surf_L = None
        elif not chk_pth(surf_R):
            print(f"\t\t[WARNING] Nativepro surface missing for R hemi for [{study_code}] {sub}-{ses}. Skipping map smoothing for this hemi. Check micapipe processing ( {root_mp}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_R}")
            surf_R = None
        else:
            if verbose:
                print(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")

        # ii. Smooth
        pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, out_pth_L, kernel=smth, verbose=False)
        pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_R, out_pth_R, kernel=smth, verbose=False)

        if pth_map_smth_L is None or pth_map_smth_R is None:
            print(f"\t\t[WARNING] Smoothing failed for [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check smoothing function output.\n")
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
        
        print(f"\tsurf: {surf}, label: {lbl}, feature: {ft}, kernel: {smth}mm")
        root_mp = f"{study['dir_root']}{study['dir_deriv']}{study['dir_mp']}"
        root_hu = f"{study['dir_root']}{study['dir_deriv']}{study['dir_hu']}"
        study_code = study['study']

        skip_L, skip_R = False, False

        # Define output paths                 
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
        
        # do not recompute if smoothed path already exists
        if chk_pth(out_pth_L) and chk_pth(out_pth_R):
            if verbose:
                print(f"\t\tSmoothed maps exist, adding paths to df:\t{out_pth_L}\t{out_pth_R}\n")
            df.loc[idx, col_base_L] = out_pth_L
            df.loc[idx, col_base_R] = out_pth_R
            return df
            
        if chk_pth(out_pth_L):
            if verbose:
                print(f"\t\tSmoothed L map exists, adding path to df:\t\t{out_pth_L}")
            df.loc[idx, col_base_L] = out_pth_L
            skip_L = True
        
        if chk_pth(out_pth_R):
            if verbose:
                print(f"\t\tSmoothed R map exists, adding path to df:\t\t{out_pth_R}")
            df.loc[idx, col_base_R] = out_pth_R
            skip_R = True
        
        surf_L, surf_R = get_surf_pth(
                root=root_hu,
                sub=sub,
                ses=ses,
                surf=surf,
                lbl=lbl,
                space="T1w"
            )
        
        if not chk_pth(surf_L) and not chk_pth(surf_R):
            print(f"\t\t[WARNING] Hippocampal surface not found for [{study_code}] {sub}-{ses}. Cannot continue to map smoothing. Check hippunfold processing ( {root_hu}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_L}\t{surf_R}\n")
            return df
        elif not chk_pth(surf_L):
            print(f"\t\t[WARNING] Hippocampal surface missing for L hemi for [{study_code}] {sub}-{ses}. Skipping map smoothing for this hemi. Check hippunfold processing ( {root_hu}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_L}")
            surf_L = None
        elif not chk_pth(surf_R):
            print(f"\t\t[WARNING] Hippocampal surface missing for R hemi for [{study_code}] {sub}-{ses}. Skipping map smoothing for this hemi. Check hippunfold processing ( {root_hu}/sub-{sub}/ses-{ses}/maps/ ). Missing: {surf_R}")
            surf_R = None
        else:
            if verbose:
                print(f"\t\tSurfaces:\t{surf_L}\t{surf_R}")
        
        # A. Get unsmoothed maps
        if ft == "thickness": # get path to unsmoothed map from hippunfold outputs
            pth_map_unsmth_L, pth_map_unsmth_R = get_surf_pth(root=root_hu, sub=sub, ses=ses, lbl="thickness", surf=surf, space="T1w")
        
        else: # generate feature map from volume and hippunfold surface
            vol_pth = get_vol_pth(root=root_mp, sub=sub, ses=ses, feature=ft) # get the volume path from micapipe outputs
            vol_pth = vol_pth if chk_pth(vol_pth) else None

            if vol_pth is None:
                print(f"\t\t[WARNING] Volume not found for [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check rawdata and/or micapipe processing. Skipping.\n")
                return df
            
            pth_save_map_unsmth_L = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_L_filename}_smth-NA.func.gii"
            pth_save_map_unsmth_R = f"{out_dir}/sub-{sub}_ses-{ses}_hipp_{out_pth_R_filename}_smth-NA.func.gii"
            
            if chk_pth(pth_save_map_unsmth_L): 
                pth_map_unsmth_L = pth_save_map_unsmth_L
            else:
                if not skip_L:
                    pth_map_unsmth_L = map(vol=vol_pth, surf=surf_L, out=pth_save_map_unsmth_L, verbose=verbose)
            
            if chk_pth(pth_save_map_unsmth_R): 
                pth_map_unsmth_R = pth_save_map_unsmth_R
            else:
                if not skip_R:
                    pth_map_unsmth_R = map(vol=vol_pth, surf=surf_R, out=pth_save_map_unsmth_R, verbose=verbose)
        
        # B. Smooth
        if verbose:
            print(f"\t\tUnsmoothed map paths:\t{pth_map_unsmth_L}\t{pth_map_unsmth_R}")
        
        pth_map_unsmth_L = pth_map_unsmth_L if chk_pth(pth_map_unsmth_L) else None
        pth_map_unsmth_R = pth_map_unsmth_R if chk_pth(pth_map_unsmth_R) else None
        
        if pth_map_unsmth_L is None and pth_map_unsmth_R is None:
            print(f"\t\t[WARNING] Unsmoothed map missing for [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check rawdata and/or micapipe processing.\n")
            return df
        elif pth_map_unsmth_L is None and not skip_L:
            print(f"\t\t[WARNING] Unsmoothed map missing for L hemi ONLY [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check micapipe processing.")
        elif pth_map_unsmth_R is None and not skip_R:
            print(f"\t\t[WARNING] Unsmoothed map missing for R hemi ONLY [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check micapipe processing.")
        
        if not skip_L:
            pth_map_smth_L = smooth_map(surf_L, pth_map_unsmth_L, out_pth_L, kernel=smth, verbose=False)
        if not skip_R:
            pth_map_smth_R = smooth_map(surf_R, pth_map_unsmth_L, out_pth_R, kernel=smth, verbose=False)
        
        if pth_map_smth_L is None or pth_map_smth_R is None:
            print(f"\t\t[WARNING] Smoothing failed [{study_code}] {sub}-{ses} (surf: {surf}, label: {lbl}, feature: {ft}). Check smoothing function output.")
        
        # Add paths to DataFrame
        if verbose:
            print(f"\t\tAdding to df: {pth_map_smth_L}\t{pth_map_smth_R}\n")
        
        if not skip_L:
            df.loc[idx, col_base_L] = pth_map_smth_L
        if not skip_R:
            df.loc[idx, col_base_R] = pth_map_smth_R
        
        return df


    print("\n[idToMap] Finding/computing smoothed maps for provided surface, label, feature and smoothing combinations. Adding paths to dataframe...")

    if dict_demo['nStudies']: # Check if 'study' column exists, else skip or set default
        assert 'study' in df_demo.columns, "[idToMap] 'study' column not found in df_demo, but 'nStudies' is True in dict_demo."
    else: 
        if len(studies) > 1:
            raise ValueError("[idToMap] 'study' column not found in df_demo, but multiple studies provided.\n\tEither a) provide a 'study' column to df_demo to indicate which study each row belongs to OR b) keep a single dictionary item with study directory information.")
    
    for idx, row in df_demo.iterrows(): # Iterate through rows and add map paths
        study_code = row['study']
        
        if dict_demo['nStudies']: # determine study dictionary item
            study_item = next((s for s in studies if s['study'] == study_code), None)
            if study_item is None:
                print(f"[idToMap] WARNING. Unknown study code `{study_code}`. Skipping row.")
                continue
            else:
                if verbose:
                    print(f"[idToMap] Processing index {idx} of {df_demo.shape[0]-1}.")
                ID_col = dict_demo['ID_' + study_item['study']]
        else: # if no matches, then take first study
            print(f"[idToMap] No 'study' column provided. Defaulting to first study in studies list: {studies[0]['study']}.")
            study_item = studies[0]
            ID_col = dict_demo['ID']
        
        sub = row[ID_col]
        ses = row['SES']

        if idx % 10 == 0 and idx > 0:
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
                                
    return df_demo


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
        print(f"\t[smooth_map] WARNING: Smoothed map not properly saved. Expected file: {out}")
        return None
    else:
        if verbose:
            print(f"\t[smooth_map] Smoothed map saved to: {out}")
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


def ses_clean(df_in, ID_col, method="newest", silent=True):
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
    if df_in.empty:
        print(f"[ses_clean] WARNING: Empty dataframe. Skipping.")
        return

    if not silent: print(f"[ses_clean] Choosing session according to method: {method}")
    
    # Find repeated IDs (i.e., subjects with multiple sessions)
    # sort df by ID_col
    df = df_in.sort_values(by=[ID_col]).copy()
    repeated_ids = df[df.duplicated(subset=ID_col, keep=False)][ID_col].unique()
    
    if not silent:
        if len(repeated_ids) == 0:
            print(f"\tNo repeated IDs found")

    rows_to_remove = []
    
    # Convert 'Date' column to datetime for comparison
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    today = pd.to_datetime('today').normalize()

    if len(repeated_ids) > 0:
        if not silent: print(f"\t{len(repeated_ids)} IDs with multiple sessions found. Processing...")
        if method == "newest":
            for id in repeated_ids:
                sub_df = df[df[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_dt'].idxmax()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        elif method == "oldest":
            for id in repeated_ids:
                sub_df = df[df[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_keep = sub_df['Date_dt'].idxmin()
                    idx_to_remove = sub_df.index.difference([idx_to_keep])
                    rows_to_remove.extend(idx_to_remove)
        else:
            # Assume method is a session code (e.g., '01', 'a1', etc)
            for id in repeated_ids:
                sub_df = df[df[ID_col] == id]
                if sub_df.shape[0] > 1:
                    idx_to_remove = sub_df[sub_df['SES'] != method].index
                    rows_to_remove.extend(idx_to_remove)

    # Remove the rows marked for removal
    df = df.drop(rows_to_remove)
    #if not silent: print(df_clean[[ID_col, 'SES']].sort_values(by=ID_col))

    # if num rows =/= to num unique IDs then write warning
    if df.shape[0] != df[ID_col].nunique():
        print(f"[ses_clean] WARNING: Number of rows ({df.shape[0]}) not equal to num unique IDs ({df[ID_col].nunique()})")
        print(f"\tMultiple sessions for IDs: {df[df.duplicated(subset=ID_col, keep=False)][ID_col].unique()}")

    if not silent: 
        print(f"\t{df_in.shape[0] - df.shape[0]} rows removed, Change in unique IDs: {df_in[ID_col].nunique() - df[ID_col].nunique()}")
        print(f"\t{df.shape[0]} rows remaining")

    return df

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

def get_d(col1, col2):
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

def visMean(dl, df_name='df_z_mean', df_metric=None, dl_indices=None, ipsiTo="L", title=None, save_name=None, save_path=None):
    """
    Create brain figures from a list of dictionary items with vertex-wise dataframes.
    Input:
        dl: list of dictionary items with keys 'study', 'grp', 'label', 'feature', {df_name}
        df_name: name of the dataframe key to use for visualization (default is 'df_z_mean')
        indices: list of indices to visualize. If None, visualize all items in the list.
        ipsiTo: hemisphere to use for ipsilateral visualization ('L' or 'R').
    """
    import pandas as pd
    from IPython.display import display

    for i, item in enumerate(dl):
        print(f"[visMean] [{i}] ({item.get('study','')} {item.get('grp','')} {item.get('label','')})")
        
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
        if len(lh_cols) == 32492:
            surface = 'fsLR-32k'
        else: 
            surface = 'fsLR-5k'

        lh = df[lh_cols]
        rh = df[rh_cols]
        #print(f"\tL: {lh.shape}, R: {rh.shape}")
        fig = showBrains(lh, rh, surface, ipsiTo=ipsiTo, save_name=save_name, save_pth=save_path, title=title, min=-2, max=2, inflated=True)

        return fig

def itmToVisual(item, df_name='df_z_mean', metric = 'dD_by3T', metric_lbl = None, ipsiTo=None, save_name=None, save_pth=None, title=None, max_val=2):
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

    df = item[df_name].loc[[metric]]
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
    
    if len(lh_cols) == 32492:
        surface = 'fsLR-32k'
    else: 
        surface = 'fsLR-5k'

    lh = df[lh_cols]
    rh = df[rh_cols]
    #print(f"\tL: {lh.shape}, R: {rh.shape}")

    title = title or f"{item.get('study', '3T-7T comp')} {item['label']}"

    fig = showBrains(lh, rh, surface, metric_lbl = metric_lbl, ipsiTo=ipsiTo, save_name=save_name, save_pth=save_pth, title=title, min=-max_val, max=max_val, inflated=True)

    return fig

def showBrains(lh, rh, surface='fsLR-5k', metric_lbl=None, ipsiTo=None, title=None, min=-2.5, max=2.5, inflated=True, save_name=None, save_pth=None, cmap="seismic"):
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
    crtx_img = itmToVisual(item, df_name=df_crtx_plt, metric=metric, metric_lbl = metric_lbl, ipsiTo=ipsiTo)
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