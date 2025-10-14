# functions to create stacked histograms
def dCor(d_a, d_b, verbose = False):
    from scipy.stats import pearsonr
    import pandas as pd

    if not isinstance(d_a, pd.Series) or not isinstance(d_b, pd.Series):
        raise ValueError("Inputs must be pandas Series.")
    
    # Identify NaN indices in both series
    nan_indices = set(d_a[d_a.isna()].index) | set(d_b[d_b.isna()].index)
    n_nan = len(nan_indices)
    
    if len(d_a) != len(d_b):
        raise ValueError("Input Series must be of the same length.")
    
    if n_nan > 0:
        
        d_a = d_a.drop(index=nan_indices)
        d_b = d_b.drop(index=nan_indices)
        if verbose:
            print(f"[dCor] Removing {n_nan} NaN values from both series.")
            print(f"[dCor] New lengths: {len(d_a)}, {len(d_b)}")
    
    cor, _ = pearsonr(d_a, d_b)
    return cor, n_nan

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
    if len(x) > 0:
        step = max(1, len(cols) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([cols[i] for i in x[::step]], rotation=45, ha='right')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    return fig, ax, x, cols, df_ctrl


def plot_rawToZ(
    ctrl_fig_ax, df_grp, df_grp_z, id, save_pth, save_name, title=None,
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
        colors = cmap(norm(row_z))
        sc = ax_top.scatter(
            x_pos, row_raw, c=row_z, cmap='bwr', norm=norm,
            label=f'{id} Raw (coloured by Z)', marker='o', alpha=0.8
        )
        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax_top, orientation='vertical', pad=0.01)
        cbar.set_label('Z-Score')
        cbar.set_ticks([-4, -3, -2, -1,0,1, 2,3, 4])
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
        ax_z.set_ylim(-4, 4)
        ax_z.set_ylabel("Z-Score")
        ax_z.set_xlabel('Vertex/Parcel')
        # set x ticks on bottom axis only
        if len(x_pos) > 0:
            step = max(1, len(cols) // 10)
            ax_z.set_xticks(x_pos[::step])
            ax_z.set_xticklabels([cols[i] for i in x_pos[::step]], rotation=90, ha='right')

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
            print(f"[rawToZ_plot_ctrl_ax] Saved: {name}")
        return name


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
    d_cor, n_nan = dCor(df_d[0], df_d[1])
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