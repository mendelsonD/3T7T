"""
Goal: Compare zBrains outputs from 3T and 7T images

Intended outputs:
- Dataframe with quantitative metrics of zBrains run

Nb. Surface files are made differently for 3T vs 7T. This must be considered in interpretations. Current to 21 Jan 2025.
-> 3T: freeSurfer https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllOutputFiles
-> 7T: fastSurfer

Daniel Mendelson, 21 Jan 2025
Working under supervision of Boris Bernhardt
"""

import os
import numpy as np
import datetime as dt

import nibabel as nib
import plotly as pl
import pandas as pd


def load_gifti(path_gii, extract = "vertices"):
    """
    Load data from GIFTI (.gii) file.
    
    Parameters:
        path_gii : str
            Path to the `.gii` file. Intended for `surf.gii` and `func.gii` files.
                
                .surf.gii : Array with two elements
                    Vertex indexed array. Retrieved with `NIFTI_INTENT_POINTSET` option. Structure:
                        nVertices x 3 [x, y, z] 
                        coordinates are in unit `milimeter` (mm)

                    Face indexed array. Retrieved with `NIFTI_INTENT_TRIANGLE` option. Structure:
                        nFaces x 3 [vertexIndex1, vertexIndex2, vertexIndex3]
                            Notes: 
                                `vertexIndex` corresponds to the index in the above `vertex indexed array`.
                                vertices that make each face are adjacent.
                
                .func.gii : Array with a single element. Structure:
                    nVertices x [value]
    
    Returns:
        gii : np.array
            gii from the file.

    Requires: 
        nibabel as nib    
        
    """
    
    gii = nib.load(path_gii)

    if path_gii.endswith(".func.gii"):
           return gii.darrays[0].data

    elif path_gii.endswith(".surf.gii"):
        
        if extract == "vertices":
            print ("[load_gifti] Extracting vertex coordinates from %s" % path_gii)
            return gii.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data # vertex format
            
        elif extract == "faces":
            print ("[load_gifti] Extracting faces from %s" % path_gii)
            return gii.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data # face format
        
        else:
            raise ValueError("[load_gifti] `Extract` must be either 'vertices' or 'faces'")
    
    else:
        raise ValueError("[load_gifti] File type not supported. Supported types are `.surf.gii` and `.func.gii`")

    
def v_extremeValue(path_gii, threshold, output = "indices"):
    """
    Returns the number of vertices with values more extreme than a threshold.

    Parameters:
        path_gii : str
            Path to the `.func.gii` file.
        threshold : float
            Threshold for extreme values.

    Returns:
        int : Number of vertices with values more extreme than the threshold.
    
    Future:
        Can specify if want one sided or two sided threshold. Default, assumes two sided.

    Requires:
        numpy as np
    """

    try:
        threshold = float(threshold)
    except ValueError:
        raise ValueError("Threshold must be a positive float or integer.")

    values = load_gifti(path_gii)

    if output == "values": # for .func.gii
        out = values[np.abs(values) > np.abs(threshold)]
        print("[v_extremeValue] Returning values only, no indices.")

    elif output == "indices":
        out = [index for index in range(len(values)) if np.abs(values[index]) > np.abs(threshold)]
        print("[v_extremeValue] Returning indices only, no values.")

    elif output == "both":
        out = (values[np.abs(values) > np.abs(threshold)], [index for index in range(len(values)) if np.abs(values[index]) > np.abs(threshold)])
        print("[v_extremeValue] Returning both values and indices.")
    else:
        raise ValueError("[v_extremeValue] Output parameter illdefined. It is currently %s Must be either `values`, `indices` or `both`." % output)

    return out

def adjVertices(path_surf_gii, listVertices, test = False):
    """
    Identifies sets of adjacent vertices in a list of vertices of interest.

    Parameters:
        path_surf_gii : str
            Path to `.surf.gii` file with surface face information
        vertOfInterest : list
            List of vertices of interest.


    Returns:
        listAdjSets : list
            List of list of adjacent vertices. Each nested 
    """

    def run(faces, v, listVertices, set = [], listAdjSets = []):
        
        # print("VERTEX: %s" %(v))
        overlap = [s for s in listAdjSets if v in s] # if `v` is in any set of listAdjSets, saves overlapping set(s)
            
        if len(overlap) > 1:
            raise ValueError("[adjVertices] Vertex %s is in more than one set. This should not happen. Examine code." %v)
        
        if overlap: # if `v` is in a set, remove that set from listAdjSets 
            # print(f"\tVertex %s is in set %s" %(v, overlap))
            set = overlap[0] # define set as the set that `v` is in
            listAdjSets.remove(overlap[0]) # remove that set from listAdjSets

        else: # add `v` to a new set and continue
            set = [v] # define new set with v

        #print("Set: %s" %set)
        
        # find adjacent vertices to `v`
        adjV = np.unique(faces[np.any(faces == v, axis=1)]) # 1D array

        for i in adjV: # Identify adjacent vertices that are also of interest 
            if i in listVertices:
                #print(f"\t %s" %i)
                if i in set: # if in the current set, skip
                    #print(f"\t%s\tIn current set %s. Skipping." %(i, set))
                    continue

                elif any(i in s for s in listAdjSets): # if in another set, combine current set with that set
                    overlap = [s for s in listAdjSets if i in s]
                    
                    if len(overlap) > 1:
                        raise ValueError("[adjVertices] Vertex %s is in more than one set. This should not happen. Examine code." %i)
                    
                    #print(f"\t%s\tIn another set. Combining current set %s with %s." %(i, set, set + i))

                    old = overlap[0]
                    set = set + old
                    #print(type(set))
                    listAdjSets.remove(old)

                    continue
            
                else: # not in any set, add to the current set
                    #print(f"\t%s\tNew. Appending to current set %s." %(i, set))
                    set.append(i)
                    continue
            
            else:
                continue

        listAdjSets.append(set)
        #print("List of sets: %s" %listAdjSets)

        return listAdjSets

    faces = load_gifti(path_surf_gii, extract = "faces")
    
    set = [] # list of a single set of adjacent vertices
    listAdjSets = [] # list of sets of adjacent vertices

    if test == True:
            print("[adjVertices] Testing for first 10 vertices of interest only.")
            
            for v in listVertices[:40]:
                listAdjSets = run(faces, v, listVertices, set, listAdjSets)
                
    else:
        for v in listVertices:
            listAdjSets = run(faces, v, listVertices, set, listAdjSets)

    print("[adjVertices] COMPLETE. Num sets: %s, Max length: %s, Longest set: %s, List of sets: %s" %(len(listAdjSets), max(map(len, listAdjSets)), max(listAdjSets, key=len), listAdjSets))
    # order listAdjSets by length of sets
    listAdjSets.sort(key = len, reverse = True)
    return listAdjSets

def main(name, path_func_gii, path_surf_gii, threshold = 1.96, save_sets = True, save_path = None, test = False):
    """
    Steps: 
        1. Get vertices where z above threshold (get_extremeZ)
        2. Get list of adjacent vertices to those vertices (count_adjVertices, requires 'vertices of interest')
        3. Return data of interest from these results (e.g., number of vertices, number of sets, maximum length of set, etc.)

    Parameters:


    Requires:
        datetime as dt
    """

    print(f"[main] Extracting zBrains analysis data.\n\tThreshold: {threshold}\n\tfunc.gii: {path_func_gii}\n\tsurf.gii: {path_surf_gii}")
    # 1. Get vertices with value > threshold
    v_extreme = v_extremeValue(path_gii = path_func_gii, threshold = threshold, output = "indices") # Returns list of vertices

    # 2. Get list of adjacent vertices to those vertices
    adjSets = adjVertices(path_surf_gii, v_extreme, test = test) # Returns list of lists of adjacent vertices

    # 3. Return data of interest
    ## Extract data of interest
    num_vertices = len(v_extreme)
    num_sets = len(adjSets)
    max_set_length = max(map(len, adjSets))

    ## Save data
    
    if save_sets == True:
        date = dt.datetime.now().strftime("%d%b%Y")
        out_name = save_path + "/" + name + "_SetsOfAdjVofInterest_" + date + ".txt"
        print("[main] Saving sets for %s to %s" %(name, out_name))
        with open(out_name, "w") as f:
            for s in adjSets:
                f.write("%s\n" %s)

    ## Return data
    out = [num_vertices, num_sets, max_set_length]
    return out


# Participant information ------
sheet_path = "/Users/danielmendelson/Library/CloudStorage/OneDrive-McGillUniversity/Documents/PhD/Boris/Epilepsy_7T/zBrainsAnalyses/data/pt_13Jan2025_ages.xlsx"
pt_sheet = pd.read_excel(sheet_path, sheet_name = "Sheet1")

ID_colName = '7T_ID'
session_colName = 'sT_ses_num'

# Parameters for analyses ------
threshold = 1.96

wd = "/Users/danielmendelson/Documents/Boris_projects/data/PNE003/zBrains/ses-a1"
set_save_path = "/Users/danielmendelson/Documents/Boris_projects/code/output"

# Declare output dataframe
out_df = pd.DataFrame(columns = ["scanner", "ID", "session", "hemisphere", "num_vertices", "num_sets", "max_set_length"])
out_df_save_path = "/host/verges/data/tank/daniel/zBrainsAnalyses/output"

# Loop through participants and RUN
for subject in pt_sheet[ID_colName]:
    # NOTE: SESSION OUGHT TO HAVE prefix '0' or 'a' in data sheet
    session = pt_sheet.loc[pt_sheet[ID_colName] == subject, session_colName].values[0].item()
    print("Subject %s, session: %s" %(subject, session))

    scanner = "7T" if "PNE" in subject or "PNC" in subject or "PNA" in subject else "3T" # identifier scanner from ID label
    #print("scanner: %s" %scanner) 

    hemi = "L" # can eventually iterate through hemispheres

    wd_subject = wd + "/sub-" + subject + "/ses-" + str(session)

    func_path = wd_subject + "/norm-z/cortex/sub-%s_ses-%s_hemi-%s_surf-fsLR-32k_label-midthickness_feature-T1map_smooth-10mm_analysis-regional.func.gii" % (subject, str(session), hemi)
    surf_path = wd_subject + "/structural/sub-%s_ses-%s_hemi-%s_space-nativepro_surf-fsnative_label-midthickness.surf.gii" % (subject, str(session), hemi)
    
    #func_path = wd + "/norm-z/cortex/sub-%s_ses-%s_hemi-%s_surf-fsLR-32k_label-midthickness_feature-T1map_smooth-10mm_analysis-regional.func.gii" % (subject, str(session), hemi)
    #surf_path = wd + "/structural/sub-%s_ses-%s_hemi-%s_space-nativepro_surf-fsnative_label-midthickness.surf.gii" % (subject, str(session), hemi)

    analysis_name = subject + "-" + str(session) + "_" + hemi

    metrics = main(
        path_func_gii=func_path, 
        path_surf_gii=surf_path, 
        name=analysis_name, 
        save_sets=True, 
        save_path=set_save_path, 
        threshold=threshold, 
        test=True
        )
    
    out_df = out_df.append({
        "scanner": scanner, 
        "ID": subject, 
        "session": session, 
        "hemisphere": hemi,
        "num_vertices": metrics[0], 
        "num_sets": metrics[1], 
        "max_set_length": metrics[2]
        }, ignore_index=True)

print("ANALYSES COMPLETE.")

# save output
date = dt.datetime.now().strftime("%d%b%Y")
out_df.to_csv(out_df_save_path + "/zBrains_quantAnal_" + date + ".csv", index = False)