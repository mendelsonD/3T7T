# Code to  quantify extreme values on vertices along a surface.
# Made to determine number of adjacent vertices with zBrains scores above a threshold and count number of groups of extreme values.

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

    import nibabel as nib
    
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

    import numpy as np

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

    import nibabel as nib
    import numpy as np

    def run(faces, v, listVertices, set = [], listAdjSets = []):
        
        import numpy as np

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

def get_ID_SES(path):
    """
    From path string, extract ID and SES.
    Assumes a path structure like: .../sub-<ID>/ses-<SES>/...
    """
    import re

    # Regular expressions to extract ID and session
    match = re.search(r"sub-([^/]+)/ses-([^/]+)", eg_dir)

    if match:
        ID = match.group(1)
        SES = match.group(2)
    else:
        print("[extractIDses] Error: ID and SES not found in path: %s" %path)
        print("[extractIDses] Returning NAN for ID and SES.")
        ID = NAN
        SES = NAN
    
    return [ID, SES]

def get_vrtxVals(dir):
    """
    Extract vertex values from a .gii file and build into df

    input:
        dir (str): path to root of BIDS directory with surf.gii files

    return:
        df (pd.dataframe): dataframe with vertex values ID/SES. Colname is ID_SES
    """
    import os
    import Utils.gen
    import pandas as pd

    # check that provided root dir exists
    if not os.path.exists(dir):
        raise ValueError("[get_vrtxVals] Provided directory does not exist: %s" %dir)
    

    values = get_giiVals(dir)
    # search for sub- and ses- in dir, take characters between this pattern and /
    
    
    ID, SES = extractIDses(dir)
    col_name = "_".join([ID, SES])
    df[col_name] = values

def search_files(directory, substrings):
    # Get the list of all files in the directory
    import os

    files = os.listdir(directory)
    
    # Filter files based on substrings
    matching_files = [f for f in files if any(sub in f for sub in substrings)]
    
    return matching_files         
    
def zbFilePtrn(region, hemi=["L", "R"]):
    """
    Return zBrains output file pattern (excluding the ID and session)

    input:
        region (dict) : with attribute 
            'region' : name of region
            'surfaces' : surface names
            'resolution' : resolution
            'features' : MRI features
            'smoothing' : smoothing kernel size
        hemi (list) < optional, default ['L','R'] > : hemisphere(s) of interest

    return:
        ptrn (list): zBrains output file pattern
    """
    ptrn_list = []
    if region["region"] == "subcortex":
        for feat in region["features"]:
            ptrn =  f"feature-{feat}" + ".csv"
            ptrn_list.append(ptrn)
    else:
        res = region["resolution"]

        for h in hemi:
            for surf in region["surfaces"]:
                    for smth in region["smoothing"]:
                        for feat in region["features"]:
                            
                            if region["region"] == "cortex":
                                dir = "/".join([region["region"]])
                                files = search_files()
                                ptrn = "_".join([f"hemi-{h}",f"surf-fsLR-{res}", f"label-{surf}", f"feature-{feat}", f"smooth-{str(smth)}mm"])
                                ptrn = ptrn + ".func.gii"
                                ptrn_list.append(ptrn)
                            
                            elif region["region"] == "hippocampus":
                            
                                ptrn = "_".join([f"hemi-{h}", f"den-{res}", f"label-{surf}", f"feature-{feat}", f"smooth-{str(smth)}mm"])
                                ptrn = ptrn + ".func.gii"
                                ptrn_list.append(ptrn)
    
    return ptrn_list

#def zbFilePatternList(regions, surf)
