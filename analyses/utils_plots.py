# functions to create stacked histograms

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


def pngs2pdf(fig_dir, ptrn, output=None, verbose=False):
    """
    Combine PNGs in fig_dir whose filename contains `ptrn` into a single PDF.

    Input:
        fig_dir: Directory containing png files.
        ptrn: substring to match in filenames (only files containing this substring are included).
        output: Directory to save output pdf file. If None, saves in fig_dir.
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
        return output_pdf
    except Exception as e:
        if verbose:
            print(f"[pngs2pdf] Failed to save PDF {output_pdf}: {e}")
        return None


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
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
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

    # create three rows: top violin (3T), middle d scatter, bottom violin (7T)
    # make figure width scale with number of parcels so plots have enough horizontal room
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1,
        figsize=(10, 40),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 3]}
    )

    bins = 30  # density bins per parcel
    cmap = mpl_cm.get_cmap('bwr')  # for d-values
    # allow title to carry main title and/or study names:
    #   title can be:
    #     - a simple string => used as overall suptitle
    #     - a tuple/list (main_title, (study0_name, study1_name))
    # If study names are provided, use them for axis mini-titles and d-legend labels.
    main_title = None
    if isinstance(title, (list, tuple)) and len(title) == 2 and isinstance(title[1], (list, tuple)):
        main_title = title[0]
        # Use only common strings between both study label lists
        study_labels = list(set(title[1][0]).intersection(set(title[1][1])))
        if not study_labels:
            # fallback: use first label from each if no commonality
            study_labels = [title[1][0][0], title[1][1][0]]
    else:
        main_title = title
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

        # dotted zero line across full axis
        ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_ylabel(ylbl)
        ax.set_title(f"{study_labels[st_idx]}", loc='left')

    # Middle axis: Cohen's d scatter for both studies
    # prepare d arrays and plotting parameters
    d_vals_list = []
    for st_idx in (0, 1):
        d_df = df_d_list[st_idx]
        if d_df is None:
            d_vals_list.append(np.full(len(cols), np.nan))
            continue
        # attempt to coerce to numeric and align columns
        d_series = pd.to_numeric(d_df.iloc[0].reindex(cols), errors='coerce')
        d_vals_list.append(d_series.values.astype(float))

    d_all = np.vstack(d_vals_list)
    # force Cohen's d scale to +/-0.5 as requested
    vlim = 0.5
    norm = Normalize(vmin=-vlim, vmax=vlim)

    # plot scatter for both studies on same axis
    markers = ['o', 's']
    labels = [study_labels[0] + ' d', study_labels[1] + ' d']
    for st_idx in (0, 1):
        # ensure numeric numpy arrays (fixes boolean masking/index issues for first element)
        dv = np.asarray(d_vals_list[st_idx], dtype=float)
        xi_valid = np.asarray(x, dtype=float)
        # mask nans
        mask = np.isfinite(dv)
        if not mask.any():
            continue
        xi_plot = xi_valid[mask]
        dv_plot = dv[mask]
        # facecolors by d, edgecolor to distinguish study
        sc = ax_mid.scatter(xi_plot, dv_plot, c=dv_plot, cmap=cmap, norm=norm,
                            marker=markers[st_idx], edgecolor=study_edge[st_idx], linewidth=0.8, s=48, label=labels[st_idx])

    ax_mid.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax_mid.set_ylabel(dlab)
    # enforce y-limits for Cohen's D plot so scale is +/-0.5
    ax_mid.set_ylim(-vlim * 1.05, vlim * 1.05)

    # place Cohen's D legend horizontally to the left above the middle axis (in the gap between top and mid)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=markers[0], color='none', markerfacecolor='lightgray', markeredgecolor=study_edge[0], markersize=8, label=study_labels[0]),
        Line2D([0], [0], marker=markers[1], color='none', markerfacecolor='lightgray', markeredgecolor=study_edge[1], markersize=8, label=study_labels[1])
    ]
    ax_mid.legend(handles=legend_handles, frameon=False,
                  loc='upper left', bbox_to_anchor=(0.0, 1.12), ncol=2)

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
        ax_bot.set_xticks(ticks)
        ax_bot.set_xticklabels(tick_labels, rotation=90, ha='right')
        # keep tick marks on top/mid for alignment but hide their labels
        ax_top.set_xticks(ticks)
        ax_top.set_xticklabels(['' for _ in ticks])
        ax_mid.set_xticks(ticks)
        ax_mid.set_xticklabels(['' for _ in ticks])

    # overall title (keeps previous title text if provided)
    if main_title:
        fig.suptitle(main_title, fontsize=14)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pth = os.path.join(save_path, f"{save_name}.png")
    fig.savefig(pth, dpi=300, bbox_inches='tight')
    
    if verbose:
        file_size = os.path.getsize(pth) / 1e6
        print(f"[plot_zToD] Saved figure ({file_size:0.1f} MB): {pth}")
    plt.close(fig)
    return pth
