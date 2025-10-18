##### Functions common to multiple plotting steps

def pngs2pdf(fig_dir, ptrn, output=None, cleanup = True, verbose=False):
    """
    Combine PNGs in fig_dir whose filename contains `ptrn` into a single PDF.

    Input:
        fig_dir: Directory containing png files.
        ptrn: substring to match in filenames (only files containing this substring are included).
        
        output: Directory to save output pdf file. If None, saves in fig_dir.
        cleanup: bool, if True, delete individual PNGs after creating PDF
        verbose: bool, print progress
    Output:
        Path to created PDF (or None if nothing matched)
    """
    import os
    import datetime
    from PIL import Image

    if output is None:
        output = fig_dir
    else:
        os.makedirs(output, exist_ok=True)
        if verbose:
            print(f"[pngs2pdf] Created/using output directory: {output}")

    if not os.path.isdir(fig_dir):
        if verbose:
            print(f"[pngs2pdf] fig_dir does not exist: {fig_dir}")
        return None

    # Find PNG files containing the pattern
    files = [f for f in os.listdir(fig_dir)
             if os.path.isfile(os.path.join(fig_dir, f)) and f.lower().endswith('.png') and ptrn in f]

    if not files:
        if verbose:
            print(f"[pngs2pdf] No PNG files containing pattern '{ptrn}' found in {fig_dir}")
        return None

    # Sort files alphabetically (simple deterministic order)
    files = sorted(files)

    # Build output filename
    safe_ptrn = ptrn.replace(os.sep, "_")
    time_stmp = datetime.datetime.now().strftime('%d%b%Y-%H%M%S')
    output_pdf = os.path.join(output, f"{safe_ptrn}_{time_stmp}.pdf")

    images = []
    for fname in files:
        p = os.path.join(fig_dir, fname)
        try:
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            if verbose:
                print(f"[pngs2pdf] Skipping {p}: {e}")

    if not images:
        if verbose:
            print(f"[pngs2pdf] No images could be opened for pattern '{ptrn}' in {fig_dir}")
        return None

    # Save combined PDF
    try:
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        if verbose:
            print(f"[pngs2pdf] PDF created: {output_pdf}")
    except Exception as e:
        if verbose:
            print(f"[pngs2pdf] Failed to save PDF {output_pdf}: {e}")
        return None

    if cleanup:
        for fname in files:
            p = os.path.join(fig_dir, fname)
            try:
                os.remove(p)
                if verbose:
                    print(f"[pngs2pdf] Deleted: {p}")
            except Exception as e:
                if verbose:
                    print(f"[pngs2pdf] Failed to delete {p}: {e}")

    return output_pdf


def correl(row_a, row_b, method = "Pearson", verbose = False):
    """
    Compute correlations.

    Parameters:
        row_a, row_b: pd.Series
            Rows of same length to correlate.
        method: str, 
            correlation method. Currently only "Pearson" is implemented.
    """
    
    from scipy.stats import pearsonr
    import pandas as pd

    if not isinstance(row_a, pd.Series) or not isinstance(row_b, pd.Series):
        raise ValueError("Inputs must be pandas Series.")
    
    # Identify NaN indices in both series
    nan_indices = set(row_a[row_a.isna()].index) | set(row_b[row_b.isna()].index)
    n_nan = len(nan_indices)
    
    if len(row_a) != len(row_b):
        raise ValueError("Input Series must be of the same length.")
    
    if n_nan > 0: # drop nan indices from both dfs
        row_a = row_a.drop(index=nan_indices)
        row_b = row_b.drop(index=nan_indices)
        if verbose:
            print(f"[dCor] Removing {n_nan} NaN values from both series.")
            print(f"[dCor] New lengths: {len(row_a)}, {len(row_b)}")
    
    cor, _ = pearsonr(row_a, row_b)
    return cor, n_nan

def correl_df(a,b, method = "Pearson"):
    """
    From two dfs with overlapping indices and same columns, compute correlation between rows with same index.
    NOTE. Assumes that overlapping index label is the substring found before the first '_' char.

    input: 
        a,b: pd.DataFrames or str (paths to dfs)
        method: str
            correlation method. Currently only "Pearson" is implemented.

    return:
        pd.Series with correlation values for each index present in both dataframes
    """
    import pandas as pd
    import tTsTGrpUtils as tsutil

    if isinstance(a, str):
        a = tsutil.loadPickle(a, verbose = False)
        assert isinstance(a, pd.DataFrame), f"Loaded object from path {a} is not a pandas DataFrame."
    if isinstance(b, str):
        b = tsutil.loadPickle(b, verbose = False)
        assert isinstance(b, pd.DataFrame), f"Loaded object from path {b} is not a pandas DataFrame."
    
    a_strip = a.copy()
    b_strip = b.copy()
    a_strip.index = [idx.split('_')[0] for idx in a.index]  
    b_strip.index = [idx.split('_')[0] for idx in b.index]

    cors = pd.Series(dtype = float)
    if not a_strip.index.equals(b_strip.index):
        print("WARNING. Indices do not match perfectly.")
    
    for df_index in a_strip.index.intersection(b_strip.index):
        row_a = a_strip.loc[df_index]
        row_b = b_strip.loc[df_index]
        cor, n_nan = correl(row_a, row_b, method=method)
        cors.loc[df_index] = cor
    
    return cors

def corresp_paths(regions, MICs, PNI, output_dir, values_dir):
    """
    Return array length num studies by num of files.

    Input:
        regions: list of dictionaries, each dictionary contains the region name and the path to the region's data   
        MICs: dictionary with the name of the MICs data and the path to the MICs data
        PNI: dictionary with the name of the PNI data and the path to the PNI data 
    """
    import os
    import numpy as np
    from vrtx import zbFilePtrn

    arr = []

    for region in regions:
    
        region_name = region["region"]
        
        mics_name = MICs["name"]
        pni_name = PNI["name"]

        mics_path = "/".join([output_dir, values_dir, mics_name, region_name])
        pni_path = "/".join([output_dir, values_dir, pni_name, region_name])

        ptrn_lst = zbFilePtrn(region, extension=".csv")
        
        for ptrn in ptrn_lst:
            # define file path
            mics_file = mics_path + "/" + MICs["name"] + "_" + ptrn
            pni_file = pni_path + "/" + PNI["name"] + "_" + ptrn
            #print(mics_file)
            #print(pni_file)

            arr.append([mics_file, pni_file])
    
    return arr


def histStack(df):
    """
    Return a stacked histogram of the data in the input dataframe.

    import:
        df: pandas dataframe, each column a list of values to create a unique histogram from
            e.g.: each column is a single participant; or each column is a different smoothing kernel in same participant; or each column is a different image resoltuion
    
    return:
        fig: plostly figure object, a 3D stacked histogram of the input data

    """
    
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    # plotly setup
    fig=go.Figure()

    # data binning and traces
    for i, col in enumerate(df.columns):
        a0=np.histogram(df[col], bins=10, density=False)[0].tolist()
        a0=np.repeat(a0,2).tolist()
        a0.insert(0,0)
        a0.pop()
        a1=np.histogram(df[col], bins=10, density=False)[1].tolist()
        a1=np.repeat(a1,2)
        fig.add_traces(go.Scatter3d(x=[i]*len(a0), y=a1, z=a0,
                                    mode='lines',
                                    name=col
                                )
                    )
    return fig


def ridge(matrix, matrix_df=None, Cmap='rocket', Range=(0.5, 2.5), Xlab="zScore", save_path=None, title=None):
    '''
    Function that plots a ridgeplot (density plot per row value from a matrix of observations (features x observations))

    Inputs:
    
      matrix: [array] Must contain the observations on the rows (subjects x features) values on the columns
      
      matrix_df: [pandas df] this script will use the key 'id' to add names to each histogram

      Cmap: colormap

    Output:
      ridgeplot
    
    '''
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Calculate the mean row values and sort matrix based on their mean
    mean_row_values = np.mean(matrix, axis=1)
    sorted_indices = np.argsort(mean_row_values)
    sorted_matrix = matrix[sorted_indices]
    sorted_id_x = matrix_df['id'].values[sorted_indices]

    ai = sorted_matrix.flatten()
    subject = np.array([])
    id_x = np.array([])

    for i in range(sorted_matrix.shape[0]):
        label = np.array([str(i+1) for j in range(sorted_matrix.shape[1])])
        subject = np.concatenate((subject, label))
        id_label = np.array([sorted_id_x[i] for j in range(sorted_matrix.shape[1])])
        id_x = np.concatenate((id_x, id_label))

    d = {'feature': ai,
         'subject': subject,
         'id_x': id_x
        }
    df = pd.DataFrame(d)

    f, axs = plt.subplots(nrows=sorted_matrix.shape[0], figsize=(3.468504*2.5, 2.220472*3.5), sharex=True, sharey=True)
    f.set_facecolor('none')

    x = np.linspace(Range[0], Range[1], 100)  # Adjust the range for x values

    for i, ax in enumerate(axs, 1):
        sns.kdeplot(df[df["subject"]==str(i)]['feature'],
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
        
        if i != sorted_matrix.shape[0]:
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel(Xlab)
            
        ax.set_yticks([])
        ax.set_ylabel("")
        
        ax.axhline(0, color="black")

        ax.set_facecolor("none")

    for i, ax in enumerate(axs):
        ax.axvline(x=0, linewidth=2, color='black')
        ax.text(0.05, 0.01, sorted_id_x[i], transform=ax.transAxes, fontsize=10, color='black', ha='left', va='bottom')

    # Calculate and add a single mean line for all subplots
    mean_asym_all = np.mean(sorted_matrix)
    for ax in axs:
        ax.axvline(x=mean_asym_all, linestyle='dashed', color='black', label=f"Mean: {mean_asym_all:.2f}")

    plt.subplots_adjust(hspace=-0.8)
    
    if title:
        plt.suptitle(title, y=0.99, fontsize=16)  # Add a title to the whole plot

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save the plot if save_path is provided
    else:
        plt.show()  # Display the plot if save_path is not provided


def group_hist(df_pths, labels, bounds=[-10,10], save_path=None):
    """
    Plot histogram from paths provided in list of files.

    Input:
        df_pths (list of str) : list of path to csv file (by convention MICs)
        labels (list): list of strings with format:
            [0]: Title of histogram
            [1]: df1 name
            [2]: df2 name
        bounds (list) <optional, default [-3,3]>: list of bounds for x-axis
    Return:
        fig (plotly figure) : plotly figure object of histogram
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    df_0 = pd.read_csv(df_pths[0])
    df_1 = pd.read_csv(df_pths[1])

    df_0 = df_0.values.flatten()
    df_1 = df_1.values.flatten()

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(df_0, bins=500, alpha=0.5, label=labels[1], color='blue', density=True)
    plt.hist(df_1, bins=500, alpha=0.5, label=labels[2], color='red', density=True)
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(0,.4)
    plt.yticks()
    plt.xticks([-10, -5, -2, 0, 2, 5, 10])
    # Labels and legend
    plt.title(labels[0])
    plt.xlabel('z-Score')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[grpHist] Saved: {save_path}")
    else:
        plt.show()


# Side by side plots


def plotMatrices(dl, df_keys, cor = True, 
                 name_append=False, sessions = None, save_pth=None, min_stat = -4, max_stat = 4, test=False):
    """
    Plot matrix visualizations for map values from corresponding study

    dl: 
        dictionary list with paired items from different studies
    df_keys: lst
        keys in the dictionary items to plot (e.g., 'map_smth')
    
    cor: bool
        if true, computes and displays correlation between the same patients at the different studies.
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
    import os
    import datetime
    import numpy as np
    import tTsTGrpUtils as tsutil
    import matplotlib.pyplot as plot
    from matplotlib import gridspec
    
    skip_idx = []
    counter = 0
    
    if type(df_keys) == str:
        df_keys = [df_keys] 

    print(f"Plotting matrices for {list(df_keys)}...")
    
    if test:
        idx_len = 3
        rdm_indices = tsutil.get_rdmPairedIndices(dl, idx_len, mtch = ['region', 'surf', 'label', 'feature', 'smth', 'downsampledRes'])
        
        print(f"TEST MODE. Applying parcellations to {idx_len} random items in dl: indices {rdm_indices}.")
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

        if item.get('downsampledRes', None) != 'native':
            difdsRes = True
            skip_idx.append(idx) # Assume that others should not be matching to this non-native index
        else: # res of pair must match
            difdsRes = False
            pass

        if sessions:
            if item.get('sesNum', None) not in sessions or item.get('sesNum', None) not in sessions:
                continue
    
        idx_other = tsutil.get_pair(dl, idx = idx, mtch=['region', 'surf', 'label', 'feature', 'smth', 'downsampledRes'], difdsRes = difdsRes, skip_idx=skip_idx)
        
        if test:
            index = rdm_indices[idx]
            index_other = rdm_indices[idx_other]
        else:
            index = idx
            index_other = idx_other
        
        if type(index_other) == list:
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
            item_txt = tsutil.printItemMetadata(item, return_txt=True)
            print(f"\tWARNING. No matching index found for: {item_txt} (idx: {index}).\nSkipping.")
            continue

        item_other = dl[idx_other]
        if item_other is None:
            item_txt = tsutil.printItemMetadata(item, return_txt=True)
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
            item_one_txt = tsutil.printItemMetadata(item_one, idx=idx_one, return_txt=True)
            item_two_txt = tsutil.printItemMetadata(item_two, idx=idx_two, return_txt=True)
            print(f"\tWARNING. Both items are None (item one: {item_one_txt}, item two: {item_two_txt}).\nSkipping.")
            continue
        elif item_one is None:
            item_one_txt = tsutil.printItemMetadata(item_one, idx=idx_one, return_txt=True)
            print(f"\tWARNING. Item_one is None: {item_one_txt}.\nSkipping.")
            continue
        elif item_two is None:
            item_two_txt = tsutil.printItemMetadata(item_one, idx=idx_one, return_txt=True)
            print(f"\tWARNING. Item_two is None: {item_two_txt}.\nSkipping.")
            continue
        
        if sessions is None:
            print(f"\t[idx 3T, 7T: {idx_one}, {idx_two}] {tsutil.printItemMetadata(item_one, return_txt = True, clean = True, printStudy = False)}")
        else:
            print(f"\t[ses 1, 2: {idx_one} ({ses_one}), {idx_two} ({ses_two})] {tsutil.printItemMetadata(item_one, return_txt = True, clean = True, printStudy = False)}")
        
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
            
            if item_one.get('downsampledRes', None) != 'native' or item_two.get('downsampledRes', None) != 'native':
                title_one = title_one + f" (res: {item_one.get('downsampledRes', None)})"
                title_two = title_two + f" (res: {item_two.get('downsampledRes', None)})"
                res_svAppend = f"res-{item_one.get('downsampledRes', None)}_{item_two.get('downsampledRes', None)}"
            else:
                res_svAppend = "native"

            feature_one = item_one['feature']
            feature_two = item_two['feature']
            
            try:
                df_one = item_one[key]
                if type(df_one) is str:
                    pth = df_one 
                    df_one = tsutil.loadPickle(pth, verbose = False) 
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
                    df_two = tsutil.loadPickle(pth, verbose = False) 
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
                item_one_txt = tsutil.printItemMetadata(item_one, idx=idx_one, return_txt=False)
                item_two_txt = tsutil.printItemMetadata(item_two, idx=idx_two, return_txt=False)
                print(f"\t\tWARNING. Missing key '{key}'. Skipping {item_one_txt} and {item_two_txt}\n")
                print('-'*50)
                continue
            elif df_one is None:
                item_one_txt = tsutil.printItemMetadata(item_one, idx=idx_one, return_txt=True)
                print(f"\t\tWARNING. Missing key '{key}' for {item_one_txt}. Skipping.\n")
                print('-'*50)
                continue
            elif df_two is None:
                item_two_txt = tsutil.printItemMetadata(item_two, idx=idx_two, return_txt=True)
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

            if cor: # calculate correlation if requested
                cors = correl_df(df_one, df_two, method = "Pearson")
                # add these correlation values to the left of cbar. The index of the series is assumed to be aligned with df_one and df_two
                # Place a separate correlation column left of the colorbar (keep the cbar in the middle)
                # Reuse the existing cbar_ax position to carve out two sub-areas: a narrow left area for correlations
                # and a (smaller) right area for the actual colorbar. The colorbar object `cbar` already exists
                # on `cbar_ax`, moving `cbar_ax` will move the drawn colorbar.
                pos = cbar_ax.get_position()
                cor_frac = 0.65  # fraction of the central slot reserved for correlation texts
                small_pad = pos.width * 0.03

                # new positions in figure coordinates: [x0, y0, width, height]
                new_cor_pos = [pos.x0, pos.y0, pos.width * cor_frac - small_pad, pos.height]
                new_cbar_pos = [pos.x0 + pos.width * cor_frac, pos.y0, pos.width * (1.0 - cor_frac) + small_pad, pos.height]

                # create an (invisible) axis for correlation labels and move the colorbar axis
                cor_ax = fig.add_axes(new_cor_pos)
                cor_ax.axis('off')
                cbar_ax.set_position(new_cbar_pos)

                # Draw correlation values aligned to the image rows (use ax1 data->figure coordinate transform)
                # cors is a pandas.Series where index should match the df_one index (if available)
                for i_row, idx_label in enumerate(df_one.index):
                    cor_val = cors.get(idx_label, np.nan) if hasattr(cors, "get") else (cors.loc[idx_label] if idx_label in cors.index else np.nan)
                    txt = f"{cor_val:0.2f}" if (not np.isnan(cor_val)) else "--"

                    # transform the y position of row i to figure coordinates
                    y_disp = ax1.transData.transform((0.0, float(i_row)))[1]
                    y_fig = fig.transFigure.inverted().transform((0.0, y_disp))[1]

                    # x position just inside the right edge of the correlation axis (figure coords)
                    x_fig = new_cor_pos[0] + new_cor_pos[2] * 0.98

                    # place text in figure coords so it sits between the two main axes
                    fig.text(x_fig, y_fig, txt, ha='right', va='center', fontsize=10, color='black')

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
                        save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{res_svAppend}_key-{key}_ses-{ses_one}{ses_two}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                    else:
                        save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{res_svAppend}_key-{key}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                elif sessions is not None:
                    save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{res_svAppend}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                else:
                    save_name = f"{region}_{feature_one}_{surface}_{label}_smth-{smth}mm_{res_svAppend}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}"
                
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
    import tTsTGrpUtils as tsutil

    use('Agg')  # Use a non-interactive backend, prevents memory build up.
    # see if ipsiContra. If not, assume L/R
    
    n = df_a.shape[0]
    
    # sort columns
    df_a_sort = tsutil.sortCols(df_a)
    df_b_sort = tsutil.sortCols(df_b)

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
            alpha = 0.02, spacing = 1, mtch_ds = True,
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
        
        mtch_ds: bool or any of int, flot, str    
            if True, then corresponding pair is that whose value in the key 'downsampledRes' is identical.
            if not `True` then should be a value of 'downsampledRes' to match to (should be int or float)

        hlines: lst  list of list of indices to draw horitzontal lines at
        name_append:    string to append to saved file names
        save_pth:       path to save figures. If None, will not save.
        verbose:        whether to print item metadata  
        test:           whether to run in test mode (only first 2 items in dl)
    
    Output:
        saves plot to path    
    """

    import os
    import datetime
    import matplotlib.pyplot as plt
    import tTsTGrpUtils as tsutil

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
        rdm_indices = tsutil.get_rdmPairedIndices(dl, idx_len, mtch = ['region', 'surf', 'label', 'feature', 'smth', 'downsampledRes'])
        
        print(f"\tTEST MODE. Plotting {idx_len} random items and their pairs: {rdm_indices}")
        dl = [dl[i] for i in rdm_indices]
    
    for idx in range(len(dl)):

        if idx in skip_idx:
            continue
        counter += 1

        difdsRes = False        
        if dl[idx].get('downsampledRes', None) == str(0.8):
            difdsRes = True
            skip_idx.append(idx) # Assume that others shouldnt be matching to this non-native data

        idx_other = tsutil.get_pair(dl, idx, mtch = ['region', 'feature', 'label', 'surf', 'smth', 'downsampledRes'], difdsRes = difdsRes, skip_idx=skip_idx)
        
        if idx_other is None:
            print(f"\tNo matching index found. Skipping.")
            continue
        
        # determine which study is tT and which is sT
        idx_tT, idx_sT = tsutil.determineStudy(dl, idx = idx, idx_other = idx_other, study_key = 'study')
        item_tT = dl[idx_tT]
        item_sT = dl[idx_sT]
        
        print(f"\t[idx 3T, 7T: {idx_tT}, {idx_sT}] {tsutil.printItemMetadata(item_tT, return_txt = True, clean = True, printStudy = False)}")
       
        # extract df
        for df_key, p, s, hlines_idx in zip(df_keys, parc, stat, hlines if hlines is not None else [None]*len(df_keys)):
            df_tT = item_tT.get(df_key, None)
            df_sT = item_sT.get(df_key, None)

            if type(df_tT) is str:
                pth = df_tT 
                df_tT = tsutil.loadPickle(pth, verbose = False)
            if type(df_sT) is str:
                pth = df_sT 
                df_sT = tsutil.loadPickle(pth, verbose = False)

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
    

# surface plotting
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
    if filename:
        print(f"[showBrain] Plot saved to {filename}")
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

### Plot raw data to z-score distributions
def get_ctrl_ax(df_ctrl, ylbl, xlbl = "Vertex/Parcel", marks = False):
    """
    Create figure object with control mean and std shaded area

    Input:
        df_ctrl: pd.DataFrame
            with indices 'mean' and 'std' and columns = vertices/parcels (order preserved)
        ylbl: str
            label for y-axis (feature)
        marks: bool
            if True, use scatterplot instead of line plot

    Output:
        fig, ax, x, cols, df_ctrl
        - x, cols used by downstream plotting
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cols = list(df_ctrl.columns)
    x = np.arange(len(cols))

    fig, ax = plt.subplots(figsize=(40, 6))
    mean = df_ctrl.loc['mean'].values
    std = df_ctrl.loc['std'].values

    if marks:
        ax.scatter(x, mean, label='Control Mean', color='black', marker='_', s=60, alpha=0.8)
        ax.errorbar(x, mean, yerr=std, fmt='_', color='black', alpha=0.5, label='Control ±1 Std Dev', capsize=0, elinewidth=2, marker=None)
    else:
        ax.plot(x, mean, label='Control Mean', color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(x,
                        mean - std,
                        mean + std,
                        color='black', alpha=0.2, label='Control ±1 Std Dev')

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.legend()

    # set xticks sparsely to avoid overcrowding
    if len(x) > 50:
        step = max(1, len(x) // 120)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([cols[i] for i in x[::step]], rotation=45, ha='right')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    return fig, ax, x, cols, df_ctrl

def plot_rawToZ(
    ctrl_fig_ax, df_grp, df_grp_z, id, save_pth, save_name, min_val, max_value, title=None,
    marks=False, verbose=False, test=True, color_by_z=False
):
    """
    Scatter plot of parcel/vertex-wise patint data overlaid on control mean and std.
    Either colour each point by z-score or plot z-scores in a separate axis below..

    Inputs:
        ctrl_fig_ax: tuple (fig, ax, x, cols, df_ctrl)   <- output from get_ctrl_ax
        df_grp: 1D array-like or pd.Series (participant raw values) or 1-row DataFrame
        df_grp_z: 1D array-like or pd.Series (participant z values) or 1-row DataFrame
        id: label for participant
        save_pth: str
        save_name: str (without file extension)
        title: str (optional) title for figure
            Default: ID-ylbl
        marks: bool
        verbose: bool
        test: bool (if True, still plots single participant; no early break here)
        color_by_z: bool (if True, color participant markers by z-score and add colorbar)
    Output:
        list of tuples: [(fig, ax_top), ...]  (one entry per call)
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import use
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    use('Agg') # Use a non-interactive backend, prevents memory build up.

    # Unpack control axis info
    try:
        fig_ctrl, ax_ctrl, x_ctrl, cols, df_ctrl = ctrl_fig_ax
    except Exception:
        raise ValueError("ctrl_fig_ax must be the output of get_ctrl_ax (fig, ax, x, cols, df_ctrl)")

    # Derive ylabel from control axis if present
    ylbl = ax_ctrl.get_ylabel() if hasattr(ax_ctrl, "get_ylabel") else ""

    # Build a fresh control figure for this participant so figures are independent
    fig, ax_top, x, cols, _ = get_ctrl_ax(df_ctrl, ylbl=ylbl, marks=marks)

    # Ensure df_grp / df_grp_z are 1D arrays aligned to cols
    if isinstance(df_grp, (pd.DataFrame,)) and df_grp.shape[0] == 1:
        row_raw = df_grp.iloc[0].reindex(cols).values
    else:
        row_raw = pd.Series(df_grp, index=cols).reindex(cols).values

    if isinstance(df_grp_z, (pd.DataFrame,)) and df_grp_z.shape[0] == 1:
        row_z = df_grp_z.iloc[0].reindex(cols).values
    else:
        row_z = pd.Series(df_grp_z, index=cols).reindex(cols).values

    # x positions
    x_pos = np.arange(len(cols))

    # Plot participant values
    if color_by_z:
        # Use z-score to color markers, add colorbar
        norm = Normalize(vmin=-4, vmax=4)
        cmap = cm.get_cmap('bwr')
        sc = ax_top.scatter(
            x_pos, row_raw, c=row_z, cmap='bwr', norm=norm,
            label=f'{id} Raw (coloured by Z)', marker='o', alpha=0.8,
            edgecolors='black', linewidths=0.5)
        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax_top, orientation='vertical', pad=0.01)
        cbar.set_label('Z-Score')
        cbar.set_ticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])

        # set xticks
    else:
        # Plot raw values on top axis
        if marks:
            ax_top.scatter(x_pos, row_raw, label=f'{id} Raw', marker='o', alpha=0.8)
        else:
            ax_top.plot(x_pos, row_raw, label=f'{id} Raw', marker='o', alpha=0.8, linewidth=1)

        # Plot z-scores on bottom axis
        divider = make_axes_locatable(ax_top)
        ax_z = divider.append_axes("bottom", size="35%", pad=0.6, sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.tick_params(axis='x', which='both', length=0)

        if marks:
            ax_z.scatter(x_pos, row_z, label=f'{id} Z-Score', color='red', marker='x', alpha=0.7)
        else:
            ax_z.plot(x_pos, row_z, label=f'{id} Z-Score', color='red', marker='x', alpha=0.7, linewidth=1)

        ax_z.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax_z.set_ylabel("Z-Score")
        ax_z.set_xlabel('Vertex/Parcel')
        # set x ticks on bottom axis only
        if len(x_pos) > 50:
            step = max(1, len(x_pos) // 2)
            ax_z.set_xticks(x_pos[::step])
            ax_z.set_xticklabels([cols[i] for i in x_pos[::step]], rotation=90, ha='right')

    ax_top.set_ylim(float(min_val), float(max_value))

    # Titles
    if title is None:
        title = f"{id} - {ylbl}" if ylbl else f"{id}"
    ax_top.set_title(title)
    # Labels and legends
    ax_top.set_ylabel(ylbl if ylbl else "Raw Values")
    ax_top.legend(loc='upper right')

    fig.tight_layout()

    if save_pth:
        os.makedirs(save_pth, exist_ok=True)
        name = os.path.join(save_pth, f"{save_name}.png")
        fig.savefig(name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        if verbose:
            print(f"[plot_rawToZ] Saved: {name}")
        return name


def plot_zToD(df_grp, df_ctrl, df_d, 
              d_mode = 'rect', xlbl="Vertex/Parcel", ylbl="Z-Score", dlab="Cohen's D", title=None, 
              save_path=None, save_name=None, verbose =True):
    """
    Plots Z-score distributions and Cohen's d effect sizes for two studies (e.g., 3T and 7T MRI data).
    This function creates a figure with three rows:
        - Top: Violin-like half-histograms of Z-scores for study 0 (assumed 3T).
        - Middle: Scatter plot of Cohen's d values for both studies, colored by effect size.
        - Bottom: Violin-like half-histograms of Z-scores for study 1 (assumed 7T).
    If only single DataFrames are provided, only the top subplot is populated.
    Parameters
    ----------
    df_grp : pandas.DataFrame or list/tuple of DataFrames
        Patient group Z-score data. If a list/tuple of length 2, interpreted as [3T, 7T].
    df_ctrl : pandas.DataFrame or list/tuple of DataFrames
        Control group Z-score data. If a list/tuple of length 2, interpreted as [3T, 7T].
    df_d : pandas.DataFrame or list/tuple of DataFrames
        Cohen's d values. If a list/tuple of length 2, interpreted as [3T, 7T].
    d_mode : str, optional
        Mode for plotting d values (default: 'rect').
    xlbl : str, optional
        Label for the x-axis (default: "Vertex/Parcel").
    ylbl : str, optional
        Label for the y-axis (default: "Z-Score").
    dlab : str, optional
        Label for the Cohen's d axis (default: "Cohen's D").
    title : str, optional
        Overall figure title.
    save_path : str, optional
        Directory to save the figure.
    save_name : str, optional
        Filename (without extension) for the saved figure.
    verbose : bool, optional
        If True, prints the path and file size of the saved figure.
    Returns
    -------
    pth : str
        Full path to the saved PNG figure.
    Raises
    ------
    ValueError
        If no valid dataframe columns are found in the inputs.
    Notes
    -----
    - Uses matplotlib with the 'Agg' backend for non-interactive plotting.
    - Handles both single-study and dual-study (3T/7T) input formats.
    - Automatically creates output directory if it does not exist.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib import use
    
    use('Agg') # non-interactive backend

    # Allow passing lists for 3T/7T or single objects
    if isinstance(df_grp, (list, tuple)):
        df_grp_list = list(df_grp)
    else:
        df_grp_list = [df_grp]

    if isinstance(df_ctrl, (list, tuple)):
        df_ctrl_list = list(df_ctrl)
    else:
        df_ctrl_list = [df_ctrl]

    if isinstance(df_d, (list, tuple)):
        df_d_list = list(df_d)
    else:
        df_d_list = [df_d]

    # ensure lists length 2 for consistent indexing; pad with None if missing
    while len(df_grp_list) < 2: df_grp_list.append(None)
    while len(df_ctrl_list) < 2: df_ctrl_list.append(None)
    while len(df_d_list) < 2: df_d_list.append(None)

    # get columns from first available dataframe
    cols = None
    for candidate in (df_grp_list + df_ctrl_list + df_d_list):
        if candidate is None:
            continue
        if isinstance(candidate, pd.DataFrame) and candidate.shape[1] > 0:
            cols = candidate.columns
            break
    if cols is None:
        raise ValueError("No valid dataframe columns found in inputs.")

    # layout geometry
    # horizontal spacing between parcels (increase to separate parcels visually)
    pad_half = 0.4
    half_width = 0.8               # half-width for half-violins (tunable)
    pad = half_width * 1.1         # horizontal padding between parcels (slightly > half_width)
    spacing = 2.0 * half_width + pad
    x = np.arange(len(cols)) * spacing

    # Initialize figure and axes
    length = min(60, len(cols)*1.6)
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1,
        figsize=(length, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 3]}
    )

    bins = 30  # density bins per parcel
    cmap = mpl_cm.get_cmap('bwr')  # for d-values

    study_labels = ['3T', '7T']
    study_edge = ['navy', 'darkred']
    # iterate studies for top and bottom violins
    for st_idx, ax in zip([0, 1], [ax_top, ax_bot]):
        dfg = df_grp_list[st_idx]
        dfc = df_ctrl_list[st_idx]
        if dfg is None and dfc is None:
            # nothing to plot for this study => leave axis empty but keep title
            ax.set_title(f"{study_labels[st_idx]}: no data", loc='left')
            continue

        # for each parcel/vertex
        for i, col in enumerate(cols):
            grp_vals = (dfg[col].dropna().values) if (isinstance(dfg, pd.DataFrame)) else np.array([])
            ctrl_vals = (dfc[col].dropna().values) if (isinstance(dfc, pd.DataFrame)) else np.array([])

            if grp_vals.size == 0 and ctrl_vals.size == 0:
                continue

            # fixed bin range for consistency
            edges = np.linspace(-4, 4, bins + 1)
            if grp_vals.size:
                g_counts, _ = np.histogram(grp_vals, bins=edges, density=True)
            else:
                g_counts = np.zeros(bins)
            if ctrl_vals.size:
                c_counts, _ = np.histogram(ctrl_vals, bins=edges, density=True)
            else:
                c_counts = np.zeros(bins)

            centers = (edges[:-1] + edges[1:]) / 2.0

            # density of 1 maps to half_width
            g_norm = g_counts.astype(float) * half_width
            c_norm = c_counts.astype(float) * half_width

            xi = x[i]

            # left: patients, right: controls (colors consistent)
            ax.fill_betweenx(centers, xi - g_norm, xi, facecolor='blue', alpha=0.6, edgecolor='none')
            ax.fill_betweenx(centers, xi, xi + c_norm, facecolor='red', alpha=0.6, edgecolor='none')

            # plot patient mean only (black short horizontal)
            if grp_vals.size:
                gm = np.nanmean(grp_vals)
                ax.plot([xi - half_width, xi], [gm, gm], color='black', linewidth=1)

        # Legend for top axis
        legend_handles_top = [
            Patch(facecolor='blue', edgecolor='blue', alpha=0.6, label='Patients (density)'),
            Patch(facecolor='red', edgecolor='red', alpha=0.6, label='Controls (density)'),
            Line2D([0], [0], color='black', lw=1, label='Patients mean')
        ]

        # Place legend horizontally to the left, next to the title
        ax_top.legend(handles=legend_handles_top,
                  loc='center left',
                  bbox_to_anchor=(0.0, 1.12),
                  ncol=len(legend_handles_top),
                  frameon=False,
                  borderaxespad=0.0)
        
        # dotted zero line across full axis
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{study_labels[st_idx]}", loc='left')
        
    
    # plot D-scores
    markers = ['o', 's']
    labels = [study_labels[0], study_labels[1]]
    dlim = 1.5
    stats = []
    for d, mrkr, colour, lbl in zip(df_d_list, markers, study_edge, labels):
        if d is None:
            continue
        # Align index positions with x (parcel order)
        idx = [cols.get_loc(col) for col in d.index if col in cols]
        xi = x[idx]
        sc = ax_mid.scatter(xi, d.values, 
                marker=mrkr, edgecolor=colour, 
                linewidth=0.8, s=30, label=lbl, facecolor='none')
        stat = f"mean {d.mean():0.2f}, mdn {d.median():0.2f}, range [{d.min():0.2f} : {d.max():0.2f}]"
        stats.append(stat)


    # enforce y-limits for Cohen's D plot so scale is +/-0.5
    ax_mid.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax_mid.set_ylabel(dlab)
    ax_mid.set_ylim(-dlim * 1.05, dlim * 1.05)

    # place Cohen's D legend horizontally to the left above the middle axis (in the gap between top and mid)
    legend_handles_mid = [
        Line2D([0], [0], marker=markers[i], color='w', label=labels[i],
                markeredgecolor=study_edge[i], markersize=8, markeredgewidth=0.8)
         for i in (0, 1)]
    
    # Move legend just above top plot boundary line and add stats annotation
    legend_y = 1.5  # slightly above the top boundary of ax_mid
    ax_mid.legend(handles=legend_handles_mid, frameon=False,
                  loc='upper left', bbox_to_anchor=(0.0, legend_y), ncol=2)

    # Annotate stats
    stats_text = "\n".join([f"{labels[i]}:{stats[i]}" for i in range(len(stats))])
    
    # Compute Pearson correlation
    d_cor, n_nan = correl(df_d[0], df_d[1])
    cor_text = f"r = {d_cor:.2f} (n_nan:{n_nan})" if n_nan > 0 else f"r = {d_cor:.2f}"

    stats_text = f"D-scores statistics ({cor_text}):\n" + stats_text
    ax_mid.annotate(stats_text,
                    xy=(0.04, legend_y-.5), xycoords='axes fraction',
                    fontsize=10, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    # Shared X ticks: sparse ticks to avoid overcrowding
    if len(x) > 0:
        # explicitly set x-limits so all three axes share identical horizontal framing
        x_margin = spacing * 1.0
        x0 = float(x[0]) - x_margin
        x1 = float(x[-1]) + x_margin
        for a in (ax_top, ax_mid, ax_bot):
            a.set_xlim(x0, x1)

        # sparse ticks (use same tick indices/positions on all axes)
        step = max(1, len(cols) // 10)
        tick_idx = np.arange(len(cols))[::step]
        ticks = x[tick_idx]
        tick_labels = [cols[i] for i in tick_idx]
        # show labels only on the bottom axis to avoid repetition/overflow
        ax_mid.set_xticks(ticks)
        ax_mid.set_xticklabels(tick_labels, rotation=45, ha='right')
        # keep tick marks on top/mid for alignment but hide their labels
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax_bot.set_xticks(ticks)
        ax_bot.set_xticklabels(tick_labels, rotation=45, ha='right')

    # overall title (keeps previous title text if provided)
    if title:
            fig.suptitle(title, fontsize=16, y=0.92)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pth = os.path.join(save_path, f"{save_name}.png")
    fig.savefig(pth, dpi=300, bbox_inches='tight')
    
    file_size = os.path.getsize(pth) / 1e6
    print(f"\t[plot_zToD] Saved figure ({file_size:0.1f} MB): {pth}")
    
    plt.close(fig)
    
    return pth

def raw2Z_vis(dl, save_path,
            key_dfs_raw = ['df_maps_parc_glsr_mean', 'df_maps_parc_dk25_mean'],
            marks = True,
            verbose = False, test = False):
    """
    Visualize within study Z-scoring to Cohen's D computations.
    NOTE. Made to only accept z-scores. Would need to change savenaming methods to accept w-scores
    
    
    Parameters
    ----------
    dl: list of dict
        List of dictionarries containing data and metadata for each analysis.
    save_path: str
    
    key_dfs_raw: list of str
        List of keys in each item of dl that contain raw data DataFrames.
    marks: bools
        If True, use scatterplot. If false, use line plots.
    
    verbose: bool
        If True, print progress messages.
    test: bool
        If True, only process the first item in dl for testing purposes.
    """
    import pandas as pd
    import tTsTGrpUtils as tsutil
    import datetime
    import os
    import numpy as np

    print(f"[z2D] Saving figures to {save_path}/raw")

    counter = 0

    for index, item in enumerate(dl): # iterate over all items
        counter += 1  
        item_txt = tsutil.printItemMetadata(item, return_txt = True)
        print(f"\n{item_txt}")
        ft = item.get('feature', None)
        surf_lbl = item.get('label', None)
        commonName = f"{item.get('study',None)}_{item.get('region', None)}_{item.get('feature', None)}_{item.get('label', None)}_{item.get('smth', None)}"
        
        ids_ctrl = item['ctrl_IDs']
        ids_tle_r = item['TLE_R_IDs']
        ids_tle_l = item['TLE_L_IDs']
        ids_tle = ids_tle_r + ids_tle_l
        
        print(f"\t{len(ids_ctrl)} CTRL | {len(ids_tle)} TLE [{len(ids_tle_r)} R, {len(ids_tle_l)} L]")

        for raw_key in key_dfs_raw:
            
            item_sv_name = f"{commonName}_{raw_key}"

            now = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
            
            if verbose:
                print(f"\n\t{raw_key}:")
            df_raw  = item.get(raw_key, None)
            if df_raw is None:
                print(f"\t\tKey not found. Skipping.")
                continue
            df_raw = tsutil.loadPickle(df_raw, verbose = False)

            df_raw_crtl = df_raw[df_raw.index.isin(ids_ctrl)]
            df_raw_ctrl_mn = df_raw_crtl.mean()
            df_raw_ctrl_std = df_raw_crtl.std()
            df_raw_ctrl_stats = pd.concat([df_raw_ctrl_mn, df_raw_ctrl_std], axis=1).T
            df_raw_ctrl_stats.index = ['mean', 'std']

            if verbose:
                print(f"\t\tShape of all raw data: {df_raw.shape}")
                print(f"\t\tShape of raw_ctrls: {df_raw_crtl.shape}")

            # iterate over patients
            df_raw_grp = df_raw_crtl = df_raw[df_raw.index.isin(ids_tle)]
            if verbose:
                print(f"\t\tShape of raw_grp: {df_raw_grp.shape}")
            z_key = f'{raw_key}_z'
            df_z_grp = item.get(z_key, None)
            if df_z_grp is None:
                print(f"\t\tKey not found: {z_key}. Skipping z-score part.")
                continue
            df_z = tsutil.loadPickle(df_z_grp, verbose = False)
            df_z_grp = df_z[df_z.index.isin(ids_tle)]
            if verbose:
                print(f"\t\tShape of z_grp: {df_z_grp.shape}")

            # sort column names by L/R and increasing index number
            df_raw_ctrl_stats_srt = tsutil.sortCols(df_raw_ctrl_stats)
            df_raw_grp_srt = tsutil.sortCols(df_raw_grp)
            df_z_grp_srt = tsutil.sortCols(df_z_grp)

            # create the ctrl axis object since it is the same for all patients that follow. Pass this axis and add to it for each patient
            ctrl_fig_ax = get_ctrl_ax(df_raw_ctrl_stats_srt, ylbl = f"Raw {item.get('feature', None)} values", marks = marks)
            
            # show ctrl_fig
            figs = []

            if ft == "T1map":
                if surf_lbl == "midthickness":
                    min_val = 1100
                    max_val = 2400
                elif surf_lbl == "white":
                    min_val = 1000
                    max_val = 1900
                else:
                    min_val = 1000
                    max_val = 2400
            elif ft == "thickness":
                min_val = 0
                max_val = 4.5
            elif ft == "flair":
                min_val = 50
                max_val = 450
            elif ft == "FA":
                min_val = 0
                max_val = 1
            elif ft == "ADC":
                min_val = 0
                max_val = 0.0015
            else:  # compute actual min and max across both control stats and patient group (handle all-NaN / empty cases)
                ctrl_vals = df_raw_ctrl_stats_srt.values.ravel() if df_raw_ctrl_stats_srt is not None else np.array([])
                grp_vals_all = df_raw_grp_srt.values.ravel() if df_raw_grp_srt is not None else np.array([])
                all_vals = np.concatenate([ctrl_vals, grp_vals_all])
                min_val = float(np.nanmin(all_vals))
                max_value = float(np.nanmax(all_vals))
                if np.isnan(min_val):
                        min_val = 0
                if np.isnan(max_value):
                    max_value = 1.0


            for i, pid in enumerate(df_raw_grp_srt.index):
                save_name = f"{item_sv_name}_{index}_{i}_{now}"
                #print(f"\t\t\t{save_name}")
                
                fig_pth = plot_rawToZ(ctrl_fig_ax = ctrl_fig_ax,
                    df_grp = df_raw_grp_srt.loc[pid].values,
                    df_grp_z = df_z_grp_srt.loc[pid].values,
                    id = pid, min_val = min_val, max_value = max_val,
                    save_pth = f"{save_path}/raw",
                    save_name = save_name,
                    marks = marks,
                    color_by_z = True)
                
                figs.append(fig_pth)

            # merge figures and deleted raw images
            pngs2pdf(fig_dir = f"{save_path}/raw", 
                            ptrn = item_sv_name,
                            output = save_path)
            # delete files in figs
            for f in figs:
                try:
                    os.remove(f)
                except:
                    pass
        if test and counter == 1:
            print("\t[test] Stopping after first item.")
            break

### Plot % voxels above z-threshold: horizontal bar graph, plotted on surface
def h_bar_series(s_tT, s_sT, title, save_pth, save_name, xlbl = None, min_val = 0, max_val = 50):
    """
    Horizontal paired-bar plot for two series (3T and 7T) with paired indices like "<base>_<suffix>".
    Both series are plotted together for the same base names; 7T = blue, 3T = red.

    Inputs
    ------
    s_tT, s_sT : pd.Series
        indices of form "base_suffix" where suffix indicates hemisphere ('L','R','left','right','ipsi','contra', etc.)
    save_pth, save_name : save location pieces (function saves PNG)
    min_val, max_val : numeric bounds (used to help determine symmetric plotting range)

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime
    import os
    from matplotlib.patches import Patch

    # validate inputs
    if not isinstance(s_tT, pd.Series) or not isinstance(s_sT, pd.Series):
        raise ValueError("series3 and series7 must be pandas Series")

    # helper: parse series into mapping base -> {role: value}
    def parse_series(s):
        mapping = {}
        names = list(s.index)
        for n in names:
            if not isinstance(n, str) or len(n.split('_')) != 2:
                continue
            base, suf = n.rsplit('_', 1)
            s0 = str(suf).lower()
            # normalize role
            if s0 in ('l', 'left', 'lh'):
                role = 'L'
            elif s0 in ('r', 'right', 'rh'):
                role = 'R'
            elif s0 in ('ipsi', 'ipsilateral'):
                role = 'ipsi'
            elif s0 in ('contra', 'contralateral'):
                role = 'contra'
            else:
                role = s0
            try:
                val = float(s.loc[n])
            except Exception:
                val = np.nan
            mapping.setdefault(base, {})[role] = val
        return mapping

    map3 = parse_series(s_tT)
    map7 = parse_series(s_sT)

    # union of bases
    bases = sorted(set(list(map3.keys()) + list(map7.keys())))

    rows = []
    for base in bases:
        d3 = map3.get(base, {})
        d7 = map7.get(base, {})
        # resolve left/right for each study, prefer ipsi/contra mapping if present else L/R
        def resolve_side(d):
            if 'ipsi' in d or 'contra' in d:
                left = d.get('ipsi', np.nan)
                right = d.get('contra', np.nan)
                if pd.isna(left):
                    left = d.get('L', d.get('l', np.nan))
                if pd.isna(right):
                    right = d.get('R', d.get('r', np.nan))
            else:
                left = d.get('L', d.get('l', np.nan))
                right = d.get('R', d.get('r', np.nan))
            left_val = float(left) if not pd.isna(left) else 0.0
            right_val = float(right) if not pd.isna(right) else 0.0
            return left_val, right_val

        left3, right3 = resolve_side(d3)
        left7, right7 = resolve_side(d7)
        rows.append({'base': base, 'left3': left3, 'right3': right3, 'left7': left7, 'right7': right7})

    df_pairs = pd.DataFrame(rows)
    if df_pairs.empty:
        raise ValueError("No paired rows found after parsing series")

    # sorting by max absolute across both studies/sides
    df_pairs['absmax'] = df_pairs[['left3','right3','left7','right7']].abs().max(axis=1)
    df_pairs = df_pairs.sort_values('absmax', ascending=True).reset_index(drop=True)

    n = df_pairs.shape[0]
    height = max(4, int(n * 0.45) + 1)
    fig, ax = plt.subplots(figsize=(10, height))

    y = np.arange(n)
    # offsets so 7T and 3T bars don't perfectly overlap vertically
    offset = 0.18
    y7 = y - offset
    y3 = y + offset
    bar_h = 0.36

    # plotting bounds (symmetric around zero)
    all_abs = np.nanmax(np.abs(df_pairs[['left3','right3','left7','right7']].values))
    try:
        usr_min = float(min_val)
    except Exception:
        usr_min = 0.0
    try:
        usr_max = float(max_val)
    except Exception:
        usr_max = 0.0
    max_bound = max(all_abs if not np.isnan(all_abs) else 0.0, abs(usr_min), abs(usr_max), 1e-6)
    xmin = -max_bound
    xmax = max_bound

    # colors
    col7 = '#1f77b4'  # blue
    col3 = '#d62728'  # red
    def col_for(v, base_col):
        # keep base_col color but allow tinting if needed; sign-based tinting not required here
        return base_col

    # magnitudes for plotting (bars extend from 0 towards +ve on each side; left bars drawn as negative widths)
    left7_vals = np.abs(df_pairs['left7'].values.astype(float))
    right7_vals = np.abs(df_pairs['right7'].values.astype(float))
    left3_vals = np.abs(df_pairs['left3'].values.astype(float))
    right3_vals = np.abs(df_pairs['right3'].values.astype(float))

    # draw bars: 7T then 3T (so 3T overlays slightly on top)
    bars7_L = ax.barh(y7, -left7_vals, height=bar_h, color=col7, edgecolor='k', align='center', label='7T (ipsi)')
    bars7_R = ax.barh(y7, right7_vals, height=bar_h, color=col7, edgecolor='k', align='center', label='7T (contra)')

    bars3_L = ax.barh(y3, -left3_vals, height=bar_h, color=col3, edgecolor='k', align='center', label='3T (ipsi)')
    bars3_R = ax.barh(y3, right3_vals, height=bar_h, color=col3, edgecolor='k', align='center', label='3T (contra)')

    # central base labels at midline
    for i, base in enumerate(df_pairs['base']):
        ax.text(0, y[i], str(base).upper(), ha='center', va='center', fontsize=9, fontweight='bold', color='black')

    # numeric annotations near inner edge (towards midline)
    inner_offset = 0.01 * (xmax - xmin)
    # helper to annotate bars
    # Annotate values OUTSIDE the bars (left values to the left of the bar, right values to the right)
    for b, val in zip(bars7_L, df_pairs['left7']):
        y_center = b.get_y() + b.get_height()/2
        outer = b.get_x() + b.get_width() - inner_offset  # left bar: annotate left of bar
        ax.text(outer - 2*inner_offset, y_center, f"{val:.1f}", ha='right', va='center', fontsize=8, color='black', fontweight='bold')
    for b, val in zip(bars3_L, df_pairs['left3']):
        y_center = b.get_y() + b.get_height()/2
        outer = b.get_x() + b.get_width() - inner_offset
        ax.text(outer - 2*inner_offset, y_center, f"{val:.1f}", ha='right', va='center', fontsize=8, color='black', fontweight='bold')

    for b, val in zip(bars7_R, df_pairs['right7']):
        y_center = b.get_y() + b.get_height()/2
        outer = b.get_x() + b.get_width() + inner_offset  # right bar: annotate right of bar
        ax.text(outer + 2*inner_offset, y_center, f"{val:.1f}", ha='left', va='center', fontsize=8, color='black', fontweight='bold')
    for b, val in zip(bars3_R, df_pairs['right3']):
        y_center = b.get_y() + b.get_height()/2
        outer = b.get_x() + b.get_width() + inner_offset
        ax.text(outer + 2*inner_offset, y_center, f"{val:.1f}", ha='left', va='center', fontsize=8, color='black', fontweight='bold')

    # aesthetics
    ax.set_yticks([])
    ax.axvline(0, color='k', linewidth=1)
    ax.set_xlim(xmin * 1.05, xmax * 1.05)
    xticks = np.linspace(xmin, xmax, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):.0f}" if t != 0 else "0" for t in xticks], fontsize=10)

    # xlabel from series name if available
    xlabel = (xlbl if xlbl is not None else "% vertices above threshold")
    ax.set_xlabel(xlabel)

    # IPSI / CONTRA labels below x-axis
    ax.text(0.25, -0.12, "IPSI", transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold')
    ax.text(0.75, -0.12, "CONTRA", transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold')

    ax.set_title(title if title is not None else "Horizontal bar series")
    # legend (single combined legend)
    handles = [Patch(facecolor=col7, edgecolor='k', label='7T'),
               Patch(facecolor=col3, edgecolor='k', label='3T')]
    ax.legend(handles=handles, loc='upper right')

    fig.tight_layout()

    # save
    now = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
    os.makedirs(save_pth, exist_ok=True)
    name = f"{save_pth}/{save_name}_{now}.png"
    fig.savefig(name, dpi=300, bbox_inches='tight')
    file_size = os.path.getsize(name) / 1e6
    
    print(f"\t[h_bar_series] Saved ({file_size:0.2f} MB): {name}")
    
    return 

def nToPer(df, num, denom, decimal = True):
    """
    Convert raw counts to percentages.

    Inputs:
        df: pd.DataFrame
            DataFrame with rows: num, denom
        num: str
            row name for numerator (e.g., count of extreme vertices).
        denom: str
            row name for denominator (e.g., total num vertices in parcel).
        decimal: bool
            If True, return decimal (0-1). If False, return percentage (0-100).
    """

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame. Got {type(df)}")
    if num not in df.index or denom not in df.index:
        raise ValueError(f"num '{num}' or denom '{denom}' not found in df index: {df.index.tolist()}")

    # avoid division by zero
    perc = (df.loc[num] / df.loc[denom])

    if not decimal:
        perc = perc * 100.0

    return perc

def percentXtremeVrtxPerParc_vis(dl,grp, key_df_ctx, key_df_hip, numerator, denominator, save_path, test = False):
    """

    Input:
        dl: list
            list of dictionary items with:
                key_df_ctx or key_df_hip: vertex-wise dataframe
                region: region to determine type of parcellation
                surf: surface type (e.g., 'fsLR-32k', 'den-0p5mm')
                group: group label ('TLE_ic', 'TLE_L', 'TLE_R', 'ctrl')
        grp: str
            Label for group. Options: 'TLE_ic', 'TLE_L', 'TLE_R', 'ctrl'
        key_df_ctx: str
            Key for dataframe with statistics regarding number of atypical vertices per parcel/lobe for cortical data
        key_df_hip: str
            Key for dataframe with statistics regarding number of atypical vertices per parcel for hippocampal data
        numerator: str
            key for dataframe with number of atypical vertices per parcel
        denominator: str
            key for dataframe with total number of vertices per parcel
        save_path: str
        
    """
    import tTsTGrpUtils as tsutil
    import pandas as pd

    skip_idx = []
    counter = 0
    print(f"\n[percentXtremeVrtxPerParc_vis] Group: {grp}, Numerator: {numerator}, Denominator: {denominator}")
    print(f"Saving plots to: {save_path}")
    for idx in range(len(dl)):
        if idx in skip_idx:
            continue
        item = dl[idx]
        # find paired item for other study
        idx_other = tsutil.get_pair(dl, idx = idx, mtch=['region', 'surf', 'label', 'feature', 'smth', 'parcellation'], skip_idx=[idx])
        if idx_other is None:
            continue
        else:
            skip_idx += [idx, idx_other]
            counter += 1

        if counter % 10 == 0:
            print(f"Progress: {counter} pairs of {len(dl)//2}")

        tT_idx, sT_idx = tsutil.determineStudy(dl, idx, idx_other, study_key = 'study')
        item_tT = dl[tT_idx]
        item_sT = dl[sT_idx]
        region = item_tT['region']
        
        title = tsutil.printItemMetadata(item_tT, printStudy=False, return_txt=True) + f" [{grp}]"
        if region == "hippocampus":
            key_df = key_df_hip
        elif region == "cortex":
            key_df = key_df_ctx

        df_tT = tsutil.loadPickle(item_tT[key_df])
        df_sT = tsutil.loadPickle(item_sT[key_df])
        
        thresh_tT = df_tT.loc['z_thresh'].values[0]
        thresh_sT = df_sT.loc['z_thresh'].values[0]
        assert thresh_tT == thresh_sT, "Z-thresholds do not match between studies"
        thresh_lbl = str(df_tT.loc['z_thresh'].values[0]).replace('.', 'p')

        save_name = f"{idx}-{idx_other}_{item_tT['region']}_{item_tT['feature']}_{item_tT['label']}_{item_tT['smth']}mm_stat-{numerator}_lobes_zThr-{thresh_lbl}_{grp}"
        if test:
            save_name = "TEST_" + save_name
        per_tT = nToPer(df_tT, numerator, denominator, decimal = False)
        per_sT = nToPer(df_sT, numerator, denominator, decimal = False)
        h_bar_series(per_tT, per_sT, title= title, xlbl = f"{numerator} % vertices with |z| > {thresh_tT:0.1f}",
                                    save_pth = save_path, 
                                    save_name = save_name,
                                    min_val = 0, max_val = 50)
        
        if test and counter == 1:
            print("Test mode: breaking after 2 pairs")
            break

### Within study Z-score distribution to Cohen's D
def z2D_vis(dl, save_path,
             key_df_d = 'df_d_ic', idx_d = 'd_TLE_ic_ipsiTo-L', 
             test = False, verbose = False):
    """
    Parcel-wise z-score distributions and corresponding D-scores for matched 3T and 7T studies.
    NOTE. Assumes naming structure of df with z-scores.

    Parameters
    ----------
    dl: list of dict
        List of dictionarries containing data and metadata for each analysis.
    save_path: str
        
    key_df_d: str
        Key in each item of dl that contain d-scoring DataFrame.
    idx_d: str
        Index in df_d DataFrame to plot.
    """
    
    import datetime
    import tTsTGrpUtils as tsutil
    import pandas as pd
    
    skip_idx = []
    counter = 0
    now = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
    commonNames = set()

    for i, item in enumerate(dl):
        
        # find pair
        if i in skip_idx:
            continue

        idx_other = tsutil.get_pair(dl, idx = i, mtch=['region', 'surf', 'label', 'feature', 'smth', 'parcellation'], skip_idx=[i])
        if idx_other is None:
            continue
        else:
            skip_idx += [i, idx_other]
            counter += 1
        
        if counter % 10 == 0:
            print(f"Progress: {counter} pairs of {len(dl)//2}")

        tT_idx, sT_idx = tsutil.determineStudy(dl, i, idx_other, study_key = 'study')
        
        tT_item = dl[tT_idx]
        sT_item = dl[sT_idx]

        title = tsutil.printItemMetadata(tT_item, return_txt = True, printStudy=False)
        
        commonName =  f"{tT_item.get('region', None)}_{tT_item.get('parcellation', None)}_{tT_item.get('feature', None)}_{tT_item.get('label', None)}"
        commonNames.add(commonName)
        
        save_name = f"{commonName}_{tT_item.get('smth', None)}_{now}"
        print(f"\n[{i}, {idx_other}]\t{title}")
        region = item.get('region', None)
        if region is not None:
            if region == 'hippocampus':
                key_str = 'dk25'
            elif region == 'cortex':
                key_str = 'glsr'
            else:
                ValueError(f"Region not recognized: {region}")
        else:
            ValueError("Region key not found.")

        # load dataframes
        dfs_d = []
        dfs_grp = []
        dfs_ctrl = []
        
        for itm_study in  [tT_item, sT_item]:

            df_d_ic = itm_study.get(key_df_d, None)
            if df_d_ic is not None and isinstance(df_d_ic, str):
                df_d_ic = tsutil.loadPickle(df_d_ic, verbose = False)
            elif df_d_ic is None:
                print("WARNING. Key not found: df_maps_parc_d_score")
                continue
            else:
                assert isinstance(df_d_ic, pd.DataFrame), f"df_d not a dataframe nor string: {type(df_d_ic)}"
            #print(f"\tRetrieving d values for index {d_key}. [Available indices: {df_d_ic.index.tolist()}]")
            
            try:
                df_d_ic_srt = tsutil.sortCols(df_d_ic)
                d_ic = df_d_ic_srt.loc[idx_d] # extract appropriate row from df_d
            except Exception as e:
                print(f"Type df_d_ic: {type(df_d_ic)}")
                print(f"WARNING. Could not retrieve d values for index {idx_d}. Error: {e}")
                print(f"Indices:\n{df_d_ic.index.tolist()}")
                continue

            df_grp_ic_key = f"df_maps_parc_{key_str}_mean_z_TLE_ic"
            df_grp_ic = itm_study.get(df_grp_ic_key, None)
            if df_grp_ic is not None and isinstance(df_grp_ic, str):
                df_grp_ic = tsutil.loadPickle(df_grp_ic, verbose = False)
            elif df_grp_ic is None:
                print(f"WARNING. Key not found: {df_grp_ic_key}")
                continue
            else:
                assert isinstance(df_grp_ic, pd.DataFrame), f"df_grp not a dataframe nor string: {type(df_grp_ic)}"
            df_grp_ic = tsutil.sortCols(df_grp_ic)

            df_ctrl_key = f"df_maps_parc_{key_str}_mean_z_ctrl"
            df_ctrl = itm_study.get(df_ctrl_key, None)
            if df_ctrl is not None and isinstance(df_ctrl, str):
                df_ctrl = tsutil.loadPickle(df_ctrl, verbose = False)
            elif df_ctrl is None:
                ValueError(f"Key not found: {df_ctrl_key}")
            else:
                assert isinstance(df_ctrl, pd.DataFrame), f"df_ctrl not a dataframe nor string: {type(df_ctrl)}"

            # rename ctrl columns ipsi/contra
            ispiTo = itm_study.get('ipsiTo', None)
            df_ctrl_ic = df_ctrl.copy()
            if ispiTo is not None:
                if ispiTo == 'L':
                    df_ctrl_ic.columns = [col.replace('_L', '_ipsi').replace('_R', '_contra') for col in df_ctrl_ic.columns]
                elif ispiTo == 'R':
                    df_ctrl_ic.columns = [col.replace('_R', '_ipsi').replace('_L', '_contra') for col in df_ctrl_ic.columns]
                else:
                    ValueError(f"ipsiTo not recognized: {ispiTo}")
            else:
                df_ctrl_ic.columns = [col.replace('_L', '_ipsi').replace('_R', '_contra') for col in df_ctrl_ic.columns]
                print("Warning: ipsiTo key not found. Defaulting to `L`.")
            df_ctrl_ic = tsutil.sortCols(df_ctrl_ic)

            
            dfs_d.append(d_ic)
            dfs_grp.append(df_grp_ic)
            dfs_ctrl.append(df_ctrl_ic)
            #print(f"\t[{item['study']}] d: {df_d_ic.shape}\tgrp: {df_grp_ic.shape}\tctrl: {df_ctrl_ic.shape}")

        if len(dfs_d) != 2 or len(dfs_grp) != 2 or len(dfs_ctrl) != 2:
            print("Warning: Did not find pairs for both studies. Skipping.")
            continue
        
        path = plot_zToD(df_grp = dfs_grp, df_ctrl = dfs_ctrl, df_d = dfs_d,
                        d_mode = 'rect', xlbl="Vertex/Parcel", ylbl="Z-Score", dlab="Cohen's D", title=title,
                        save_path = f"{save_path}/raw", save_name = save_name, verbose = True)
        
        if test and counter == 1:
            break

    for commonName in commonNames:
        print(commonName)
        pngs2pdf(fig_dir = save_path, 
                        ptrn = commonName,
                        output = save_path)


### Between stduy D-score statistics
def plot_dD(df, 
            title=None, xlbl=None, ylbl=None, sorted=True, save_path=None, verbose = False):
    """
    Plot delta D.
    Plot d_3T, d_7T in a scatterplot (top) and a single heatmap (bottom, 3 rows).
    If sorted=True: do NOT rename dataframe columns, but remove trailing '_suffix'
    from tick labels and annotate suffix groups below the heatmap. Draw dotted
    vertical split lines between suffix groups.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least 3 rows:
        - Row 0: d values for study 1 (3T)
        - Row 1: d values for study 2 (7T)
        - Row 2: delta d (d_7T - d_3T)
        Optional additional rows:
        - Row 3: delta d normalized by d_3T (delta d / |d_3T|)
        - Row 4: delta d normalized by d_7T (delta d / |d_7T|)
    
    title : str, optional
        Title for the scatter plot (default: None).
    xlbl : str, optional
        Label for the x-axis (default: "Vertex/Parcel").
    ylbl : str, optional
        Label for the y-axis (default: "Within study Cohen's D (TLE vs CTRL)").
    sorted : bool, optional
        If True, group columns by suffix (e.g., '_lh', '_rh') and annotate groups below the heatmap.
        Default is True.
    save_path : str, optional
        Path to save the figure (default: None, which displays the plot instead).
    verbose : bool, optional
        If True, print progress messages (default: False).

    Returns
    -------
    str
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import pandas as pd

    # numeric conversion (preserve original column names order)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    cols_orig = list(df_numeric.columns)
    ncols = df_numeric.shape[1]
    length = min(75, max(6, int(ncols * 0.3)))

    # prepare display names (strip trailing suffix for labels) but DO NOT change df_numeric
    suffix_order = []
    base_cols = []
    suffixes_per_col = []
    if sorted:
        for c in cols_orig:
            if isinstance(c, str) and ('_' in c):
                base, suf = c.rsplit('_', 1)
                base_cols.append(base)
                suffixes_per_col.append(suf)
                if suf not in suffix_order:
                    suffix_order.append(suf)
            else:
                base_cols.append(c)
                suffixes_per_col.append(None)

    else:
        base_cols = cols_orig.copy()
        suffixes_per_col = [None] * len(base_cols)

    # collect the three rows (use NaN fallback)
    rows = []
    for idx in (2, 3, 4):
        if idx < len(df_numeric):
            rows.append(df_numeric.iloc[idx].values.astype(float))
        else:
            rows.append(np.full(ncols, np.nan))
    heat_data = np.vstack(rows)   # shape (3, ncols)

    # Hard-coded bounds for each row (tunable)
    bounds = np.array([1.0, 5.0, 5.0], dtype=float)

    # Scale each row into [-1,1] by its bound so we can display a single imshow
    scaled = np.empty_like(heat_data, dtype=float)
    for r in range(3):
        scaled[r] = heat_data[r] / bounds[r]

    # create figure with top scatter and bottom heatmap sharing x
    n_heat_rows = 3
    heat_ratio = max(1.8, 0.9 * n_heat_rows)
    top_ratio = 2.0
    total_height = 2 + heat_ratio * 2.0
    fig, (ax_main, ax_heat) = plt.subplots(
        2, 1,
        figsize=(length, total_height),
        gridspec_kw={'height_ratios': [top_ratio, heat_ratio]},
        sharex=True
    )

    x = np.arange(ncols)
    # scatter 3T / 7T if present (use original columns for values)
    if df_numeric.shape[0] >= 1:
        ax_main.scatter(x, df_numeric.iloc[0].values, label='3T', color='tab:blue', marker='o', s=40)
    if df_numeric.shape[0] >= 2:
        ax_main.scatter(x, df_numeric.iloc[1].values, label='7T', color='tab:orange', marker='s', s=40)
    # Draw dotted gridlines only at 0, 0.5, and 1
    for y in [0, 0.5, -0.5, 1, -1]:
        if y in [0.5, -0.5, 1, -1]:
            ax_main.axhline(y, color='black', linestyle=':', linewidth=1, alpha=0.6)
        elif y in [0]:
            ax_main.axhline(y, color='black', linestyle='-', linewidth=2, alpha=0.7)
    if ylbl:
        ax_main.set_ylabel(ylbl)
    else:
        ax_main.set_ylabel("Within study Cohen's D (TLE vs CTRL)")
    if title:
        ax_main.set_title(title)
    ax_main.legend()
    # Remove grid except for the custom lines above
    ax_main.grid(False)

    # single imshow of scaled data with common vmin/vmax = -1..1
    im = ax_heat.imshow(scaled, aspect='auto', cmap='bwr', vmin=-1.0, vmax=1.0)
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_yticklabels(['Δd = 7T - 3T', 'Δd/3T', 'Δd/7T'])
    if xlbl is not None:
        ax_heat.set_xlabel(xlbl)
    else:
        ax_heat.set_xlabel('Vertex/Parcel')

    # If sorted, identify suffix groups and compute boundaries / annotation positions
    ordered_suffixes = []
    suf_idxs = {}
    if sorted and any(s is not None for s in suffixes_per_col):
        for idx, s in enumerate(suffixes_per_col):
            suf_idxs.setdefault(s, []).append(idx)
        ordered_suffixes = [s for s in suffix_order if s in suf_idxs]

        # draw vertical dotted split lines between suffix groups (in data coordinates)
        # boundary between group A and B is midpoint between last index of A and first index of B
        for a, b in zip(ordered_suffixes[:-1], ordered_suffixes[1:]):
            last_a = suf_idxs[a][-1]
            first_b = suf_idxs[b][0]
            bx = 0.5 * (x[last_a] + x[first_b])
            # draw line on both axes for alignment
            for ax in (ax_main, ax_heat):
                ax.axvline(bx, linestyle=':', color='black', linewidth=3, alpha=1)

        # place suffix annotations BELOW the heatmap (use ax_heat transform; negative y in axis coords)
        for suf in ordered_suffixes:
            inds = np.array(suf_idxs[suf])
            mid = inds.mean() if inds.size else np.nan
            label = str(suf).replace('L', 'LEFT').replace('R', 'RIGHT').replace('ipsi', 'IPSILATERAL').replace('contra', 'CONTRALATERAL')
            ax_heat.text(mid, -0.12, label, transform=ax_heat.get_xaxis_transform(),
                         ha='center', va='top', fontsize=10, fontweight='bold', color='black',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Finally set xticks and xtick labels WITHOUT suffix (use base_cols)
    ax_heat.set_xticks(x)
    ax_heat.set_xticklabels(base_cols, rotation=45, ha='right')

    # create one small colorbar per row mapped to original (unscaled) values,
    # positioned to the right of the heatmap aligned with each row.
    pos = ax_heat.get_position()  # Bbox in figure coords
    row_h = pos.height / 3.0
    cbar_width = 0.0025
    pad = 0.02

    for r in range(3):
        v_pad = 0.025 * r
        y0 = pos.y0 + (2 - r) * row_h
        cax = fig.add_axes([pos.x1 + pad, y0 + 0.05 - v_pad, cbar_width, row_h - 0.002])
        sm = ScalarMappable(norm=Normalize(vmin=-bounds[r], vmax=bounds[r]), cmap='bwr')
        sm.set_array([])  # required for colorbar
        cb = fig.colorbar(sm, cax=cax, orientation='vertical')
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')

    plt.tight_layout(rect=[0, 0, 0.92, 1])  # leave space for colorbars

    # save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        file_size = os.path.getsize(save_path) / 1e6  # size in MB
        if verbose:
            print(f"Figure saved to ({file_size:0.2f} MB): {save_path}")

    return save_path

def plot_dByStudy(d_3T, d_7T,
                  title=None, scatter_lbl = None, xlbl="Cohen's D 3T", ylbl="Cohen's D 7T", save_path=None, verbose = True):
    """
    Plot cohen's D at 3T as a function of cohen's D at 7T. Each point is a parcel/vertex.
    Adds marginal histograms for each axis and annotate the means of both distributions.

    Parameters:
    d_3T : pandas.DataFrame
        DataFrame of Cohen's D values for 3T study (single row, columns are parcels/vertices).
    d_7T : pandas.DataFrame
        DataFrame of Cohen's D values for 7T study (single row, columns are parcels/vertices).
    
    title : str, optional
        Title for the plot.
    scatter_lbl : str, optional
        Label for the scatter points in the legend (default: 'vertex/parcel').
    xlbl : str, optional
        Label for the x-axis (default: "Cohen's D 3T").
    ylbl : str, optional
        Label for the y-axis (default: "Cohen's D 7T").
    sa  ve_path : str, optional
        Path to save the figure (including filename, without extension). If None, the figure is not saved.
    verbose : bool, optional
        If True, prints the path and file size of the saved figure.

    Returns:   
    str or None
        Path to the saved PNG figure, or None if not saved.
    """

    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import pandas as pd
    from matplotlib import use
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn as sns
    use('Agg') # non-interactive backend

    # numeric conversion (preserve original column names order)
    d3 = d_3T.apply(pd.to_numeric, errors='coerce')
    d7 = d_7T.apply(pd.to_numeric, errors='coerce')
    # Remove NaN values by keeping only indices where both d3 and d7 are not NaN
    valid_idx = d3.index[d3.notna() & d7.notna()]
    d3 = d3.loc[valid_idx]
    d7 = d7.loc[valid_idx]
    dif = d7 - d3

    # align indices (intersection only)
    common_idx = d3.index.intersection(d7.index)
    d3 = d3.reindex(common_idx)
    d7 = d7.reindex(common_idx)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Add marginal histograms using axes_grid1
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", size="20%", pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)

    # Hide tick labels for histograms
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    # Scale alpha by the absolute difference between d7 and d3 (dif)
    # Normalize dif to [0.2, 1.0] for alpha (avoid zero alpha)
    dif_abs = np.abs(dif.values)
    if len(dif_abs) > 0 and np.nanmax(dif_abs) > 0:
        dif_norm = (dif_abs - np.nanmin(dif_abs)) / (np.nanmax(dif_abs) - np.nanmin(dif_abs) + 1e-8)
        alpha_vals = 0.2 + 0.8 * dif_norm  # alpha in [0.2, 1.0]
    else:
        alpha_vals = np.full_like(dif_abs, 0.4)

    # Add legend label parameter (default: 'vertex/parcel')
    if scatter_lbl is None:
        scatter_lbl = 'vertex/parcel'
    sc = ax.scatter(d3.values, d7.values, color='black', marker='o', s=40, alpha=alpha_vals, label=scatter_lbl)
    ax.legend(loc='upper left')

    # Marginal KDE (thinner lines, less fill)
    sns.kdeplot(d3.values, ax=ax_histx, color='grey', fill=False, linewidth=1, common_norm=False)
    sns.kdeplot(y=d7.values, ax=ax_histy, color='grey', fill=False, linewidth=1, common_norm=False)
    # Remove borders and axis for the histograms
    for ax_hist in [ax_histx, ax_histy]:
        ax_hist.set_frame_on(False)
        ax_hist.axis('off')

    # Draw dotted gridlines only at 0, 0.5, and 1
    for v in [0, 0.5, -0.5, 1, -1]:
        if v in [1, -1]:
            ax.axhline(v, color='black', linestyle=':', linewidth=1, alpha=0.6)
            ax.axvline(v, color='black', linestyle=':', linewidth=1, alpha=0.6)
        elif v in [0]:
            ax.axhline(v, color='black', linestyle='-', linewidth=2, alpha=0.7)
            ax.axvline(v, color='black', linestyle='-', linewidth=2, alpha=0.7)
    
    # add y-tick at the mean value of d3, d7. Do so opposite the other yticks
    d3_mean = np.mean(d3.values)
    d7_mean = np.mean(d7.values)
    
    # Annotate mean values on the marginal axes
    ax_histx.text(d3_mean, ax_histx.get_ylim()[1]*0.9, f"{d3_mean:.2f}", ha='center', va='bottom', fontsize=10)
    ax_histy.text(ax_histy.get_xlim()[1]*0.9, d7_mean, f"{d7_mean:.2f}", ha='right', va='center', fontsize=10)

    # Calculate % of points in each quadrant and annotate
    total = len(d3)
    if total > 0:
        q1 = np.sum((d3.values > 0) & (d7.values > 0))
        q2 = np.sum((d3.values < 0) & (d7.values > 0))
        q3 = np.sum((d3.values < 0) & (d7.values < 0))
        q4 = np.sum((d3.values > 0) & (d7.values < 0))
        pct_q1 = 100 * q1 / total
        pct_q2 = 100 * q2 / total
        pct_q3 = 100 * q3 / total
        pct_q4 = 100 * q4 / total

        # Place annotations in each quadrant
        ax.annotate(f"{pct_q1:.1f}%", xy=(0.7, 0.875), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='blue')
        ax.annotate(f"{pct_q2:.1f}%", xy=(0.3, 0.875), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='blue')
        ax.annotate(f"{pct_q3:.1f}%", xy=(0.3, 0.1), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='blue')
        ax.annotate(f"{pct_q4:.1f}%", xy=(0.7, 0.1), xycoords='axes fraction', ha='center', va='center', fontsize=10, color='blue')

    # identity line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if xlbl:
        ax.set_xlabel(xlbl)
    else:
        ax.set_xlabel("Cohen's D 3T")
    if ylbl:
        ax.set_ylabel(ylbl)
    else:
        ax.set_ylabel("Cohen's D 7T")
    if title:
        # Set title above the main scatter axis, not the marginal histograms
        fig.suptitle(title, y=0.98, fontsize=14)
        # Optionally reduce figure size for less tall/wide histograms
        fig.set_size_inches(6, 6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        file_size = os.path.getsize(save_path) / 1e6  # size in MB
        if verbose:
            print(f"Figure saved to ({file_size:0.2f} MB): {save_path}")

    return save_path

def btwD_vis(dl, save_path,
             key_winD = 'df_d_ic', idx_winD = ['d_TLE_ic_ipsiTo-L'], 
             key_btwD = 'comps_df_d_ic', idx_btwD = ['d_TLE_ic_ipsiTo-L_Δd', 'd_TLE_ic_ipsiTo-L_Δd_by3T', 'd_TLE_ic_ipsiTo-L_Δd_by7T'], 
             test=False):
    """
    Visualize between study D-score comparisons.

    Parameters
    ----------
    dl : list of dict
        List of dictionaries containing data and metadata for each analysis.
    save_path: str
    
    key_winD : str, optional
        Key for the dataframe with within-study D-score data in the dictionaries (default is 'df_d_ic').
    idx_winD : list of str, optional
        List of indices (i.e. row names) to extract from the within-study D-score dataframe (default is ['d_TLE_ic_ipsiTo-L']).
    key_btwD : str, optional
        Key for the dataframe with between-study D-score data in the dictionaries (default is 'comps_df_d_ic').
    idx_btwD : list of str, optional
        List of indices (i.e. row names) to extract from the between-study D-score dataframe (default is ['d_TLE_ic_ipsiTo-L_Δd', 'd_TLE_ic_ipsiTo-L_Δd_by3T', 'd_TLE_ic_ipsiTo-L_Δd_by7T']).
    
    test : bool, optional
        If True, runs in test mode (default is False).
    """
    import pandas as pd
    import datetime
    import tTsTGrpUtils as tsutil
    import numpy as np

    counter = 0
    now = datetime.datetime.now().strftime("%d%b%Y-%H%M%S")
    commonNames = set()

    for i, item in enumerate(dl):
        counter += 1
        #print(item.keys())
        
        title = tsutil.printItemMetadata(item, return_txt = True)
        
        commonName =  f"{item.get('region', None)}_{item.get('feature', None)}_{item.get('label', None)}_{item.get('smth', None)}mm"
        commonNames.add(commonName)
        
        print(f"\n\t{title}")
        region = item.get('region', None)
        if region is not None:
            if region == 'hippocampus':
                key_str = 'dk25'
            elif region == 'cortex':
                key_str = 'glsr'
            else:
                ValueError(f"Region not recognized: {region}")
        else:
            ValueError("Region key not found.")

        # load dataframes with raw D-scores per study
        df_D_ic = item.get(key_winD, None)
        if df_D_ic is not None and isinstance(df_D_ic, list):
            #print(f"{len(df_D_ic)}: {df_D_ic}")
            
            dfs = [] # initilaize
            for study, df in zip(item.get('studies', None), df_D_ic):
                
                df_in = tsutil.loadPickle(df, verbose = False)
                
                try:
                    df_in = df_in.loc[idx_winD] # extract index of interest
                except Exception as e:
                    print(f"WARNING. Could not extract index `{idx_winD}` from df_in [Type: {type(df_in)}]. Error: {e}")
                    continue
                
                if study == 'MICs':
                    # add suffix to index names
                    suf = '3T'
                elif study == 'PNI':
                    suf = '7T'
                else:
                    ValueError(f"Study not recognized: {study}")
                df_in.index = [f"{idx}_{suf}" for idx in df_in.index]
                
                #print(f"\t{study}: {df_in.index}")
                dfs.append(df_in)
            
            # combine D-scores from both studies
            if len(dfs) < 2:
                print(f"WARNING. Did not find both studies. Skipping.")
                continue
            df_D_ic = pd.concat(dfs, axis=0)

        elif df_D_ic is None:
            print("WARNING. Key not found: df_maps_parc_d_score")
            continue
        else:
            print(f"df_D_ic is of an unrecognized type: {type(df_D_ic)}. Skipping")
            continue
        
        try: # sort col names
            df_D_ic = tsutil.sortCols(df_D_ic)
        except Exception as e:
            print(f"WARNING. Could not sort df_D_ic [Type: {type(df_D_ic)}]. Error: {e}")
            continue

        #print(f"\t\tWithin study df <{df_D_ic.shape}>: {df_D_ic.index.tolist()}")

        df_dD_ic = item.get(key_btwD, None)
        if df_dD_ic is not None and isinstance(df_dD_ic, str):
            df_dD_ic = tsutil.loadPickle(df_dD_ic, verbose = False)

            try:
                df_dD_ic = df_dD_ic.loc[idx_btwD] # extract appropriate row from df_d
            except Exception as e:
                print(f"WARNING. Could not extract index `{idx_btwD}` from df_dD_ic [Type: {type(df_dD_ic)}]. Error: {e}")
                continue
        elif df_dD_ic is None:
            print("WARNING. Key not found: df_maps_parc_d_score")
            continue
        else:
            assert isinstance(df_dD_ic, pd.DataFrame), f"df_d not a dataframe nor string: {type(df_dD_ic)}"

        try:
            df_dD_ic_srt = tsutil.sortCols(df_dD_ic)
            #print("cols sorted")
        except Exception as e:
            print(f"WARNING. Could not sort `df_dD_ic` (Type: {type(df_dD_ic)}). Error: {e}")
            continue
        
        if item.get('region', None) == 'cortex':
            parc_lbl = 'GLASSER Parcel'
        elif item.get('region', None) == 'hippocampus':
            parc_lbl = 'DK25 Parcel'
        else:
            parc_lbl = 'Vertex/Parcel'

        # concat within study and between study D-statistics
        df = pd.concat([df_D_ic, df_dD_ic], axis=0)

        # plot scatter plot with difference statistics
        plt1 = plot_dD(df, title = title, 
                            xlbl = parc_lbl, 
                            sorted = True, 
                            save_path = f"{save_path}/01_{commonName}_{now}.png",
                            verbose = False)
        
        # plot 3T vs 7T cohen's D scatter plot
        plt2 = plot_dByStudy(d_3T = df_D_ic.loc['d_TLE_ic_ipsiTo-L_3T'],
                                d_7T = df_D_ic.loc['d_TLE_ic_ipsiTo-L_7T'],
                                ylbl = "Cohen's D 7T (TLE vs CTRL)",
                                xlbl = "Cohen's D 3T (TLE vs CTRL)",
                                scatter_lbl = parc_lbl,
                                title = title,
                                save_path = f"{save_path}/02_{commonName}_{now}.png",
                                verbose = False)

        # merge plots into pdfs
        pdf_pth = pngs2pdf(fig_dir = save_path, 
                        ptrn = commonName,
                        output = save_path,
                        cleanup = True,
                        verbose = True)

        if test:
            print("TEST IS TRUE?")
            if counter == 1:
                break
    
    print(f"\nCompleted btwD_vis for {counter} analyses.")
    return