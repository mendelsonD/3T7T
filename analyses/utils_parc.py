def col_key(col):
    import re
    s = str(col)
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else float('inf')

def sort_columns_numeric(df):
    import numpy as np
    cols = list(df.columns)
    keys = [col_key(c) for c in cols]
    idx = np.argsort(keys, kind='stable')   # stable keeps original order among duplicates
    return df.iloc[:, idx]

def group_stats_by_colname(df, dropna=True):
    import numpy as np
    import pandas as pd
    # get unique column names sorted numerically by their key
    unique_names = sorted(pd.unique(list(df.columns)), key=col_key)
    out = {}
    for name in unique_names:
        # select all columns having this name (works with duplicate column labels)
        sub = df.loc[:, df.columns == name]
        vals = sub.to_numpy().ravel()
        if dropna:
            vals = vals[~np.isnan(vals)]
        series = pd.Series(vals)
        out[name] = series.describe()   # count, mean, std, min, 25%, 50%, 75%, max
        # place 50% (median) next to mean
        out[name]['mdn'] = out[name]['50%']
        out[name]['mn-mdn'] = abs(out[name]['mean'] - out[name]['mdn'])
        out[name] = out[name][['count', 'mn-mdn', 'mean', 'mdn', 'std', 'min', '25%', '75%', 'max']]  
    
    stats_df = pd.DataFrame(out).T   # rows = vertex name, cols = describe fields
    return stats_df

def plot_group_distributions(df, names=None, figsize=(8, 4), bins=50):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if names is None:
        import pandas as pd
        names = sorted(pd.unique(list(df.columns)), key=col_key)
    for name in names:
        sub = df.loc[:, df.columns == name]
        vals = sub.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        plt.figure(figsize=figsize)
        sns.boxplot(vals, bins=bins, kde=True)
        plt.title(f"Distribution for column group: {name}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_ridgeplot(matrix, matrix_df=None, Cmap='rocket', Range=(0, 5), Xlab="Map value",
                   save_path=None, title=None, Vline=None, VlineCol='darkred'):
    """
    Ridge plot: each row = individual (variable number), each column = vertex (variable).
    - matrix: numpy array or pandas DataFrame (rows: individuals, cols: vertices)
    - matrix_df: optional DataFrame with an 'id' column used for labeling rows
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib as mpl
    from matplotlib import use
    use('Agg')

    # convert to numpy
    matrix_np = matrix.values if hasattr(matrix, "values") else np.asarray(matrix)
    if matrix_np.ndim != 2:
        raise ValueError("matrix must be 2D (n_individuals x n_vertices)")

    n_individuals, n_vertices = matrix_np.shape

    # sort rows by mean value (nan-safe)
    mean_row_values = np.nanmean(matrix_np, axis=1)
    sorted_idx = np.argsort(mean_row_values)
    sorted_matrix = matrix_np[sorted_idx, :]

    # optional ids
    if matrix_df is not None and 'id' in matrix_df.columns:
        sorted_id_x = matrix_df['id'].values[sorted_idx]
    else:
        sorted_id_x = [str(i) for i in (sorted_idx + 1)]

    # build small dataframe for seaborn use (optional)
    ai = sorted_matrix.flatten()
    subject = np.repeat(np.arange(1, n_individuals + 1), n_vertices)
    id_x = np.repeat(sorted_id_x, n_vertices)
    df = pd.DataFrame({'feature': ai, 'subject': subject, 'id_x': id_x})

    # figure sizing: width fixed, height per row with cap
    per_row_in = 0.28
    max_height_in = 40
    fig_w = 8
    fig_h = min(max_height_in, max(1.0, per_row_in * n_individuals))
    fig, axs = plt.subplots(nrows=n_individuals, ncols=1,
                            figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    fig.patch.set_alpha(0)

    x = np.linspace(Range[0], Range[1], 300)

    for i, ax in enumerate(axs, start=1):
        # data for this subject
        vals = sorted_matrix[i - 1, :]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            # nothing to plot
            ax.set_xlim(Range)
            ax.set_yticks([])
            ax.set_facecolor("none")
            continue

        # kde: if too few unique points, fall back to histogram
        try:
            sns.kdeplot(vals, fill=True, color="w", alpha=0.25,
                        linewidth=1.5, legend=False, ax=ax)
        except Exception:
            ax.hist(vals, bins=30, color='w', alpha=0.25, density=True)

        ax.set_xlim(Range[0], Range[1])

        # color fill using imshow clipped to kde polygon (if available)
        try:
            im = ax.imshow(np.vstack([x, x]), cmap=Cmap, aspect="auto",
                           extent=[*ax.get_xlim(), *ax.get_ylim()], alpha=1.0)
            # get path from the PolyCollection created by kdeplot
            if ax.collections:
                path = ax.collections[0].get_paths()[0]
                patch = mpl.patches.PathPatch(path, transform=ax.transData)
                im.set_clip_path(patch)
        except Exception:
            pass

        # remove spines, y ticks
        for s in ['left', 'right', 'bottom', 'top']:
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.axhline(0, color="black")
        ax.set_facecolor("none")

        # x tick only on last axis
        if i != n_individuals:
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xlabel(Xlab)

    # xticks: only on last axis
    if n_individuals > 0:
        last_ax = axs[-1]
        last_ax.set_xticks([Range[0], Range[1]])
        last_ax.tick_params(axis='x', which='major', labelsize=9, rotation=0)

    # optional id labels at left of each row (small)
    print_labels = False
    if print_labels:
        for i, ax in enumerate(axs):
            ax.text(0.01, 0.02, sorted_id_x[i], transform=ax.transAxes,
                    fontsize=8, ha='left', va='bottom')

    # vertical reference line
    if Vline is not None:
        for ax in axs:
            ax.axvline(x=Vline, linestyle='dashed', color=VlineCol)

    # overlap rows slightly
    hspace = -0.7 if n_individuals > 1 else 0.2
    plt.subplots_adjust(hspace=hspace)

    if title:
        plt.suptitle(title, y=0.99, fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.04)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return fig


def pngs2pdf(fig_dir, output=None, verbose=False, cleanup=False):
    """
    Combine multiple pngs held in same folder to a single pdf.
    Groups pngs that differ only by the "_col-<...>" token (and also
    ignores _smth-, _stat- and time suffixes when grouping).

    If cleanup=True, delete the source PNG files that were merged once the PDF
    is successfully created.
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
            if verbose:
                print(f"Created output directory: {output}")

    # Find all PNG files in the directory
    files = [f for f in os.listdir(fig_dir)
             if os.path.isfile(os.path.join(fig_dir, f)) and f.lower().endswith('.png')]

    # Normalise filename for grouping: remove _col-..., _smth-...mm, _stat-..._ and time stamp -###### before extension
    def get_group_key(filename):
        s = filename
        s = re.sub(r"_col-[^_\.]+", "", s)                  # remove column token
        s = re.sub(r"_smth-(\d+|NA)mm", "", s)             # Remove smoothing pattern
        s = re.sub(r"_stat-([^_\.]+)_", "_", s)            # Remove stat pattern
        s = re.sub(r"-(\d{6})(?=\.\w+$)", "", s)           # Remove time pattern
        s = re.sub(r"_+", "_", s)                          # collapse repeated underscores
        s = re.sub(r"\.png$", "", s, flags=re.IGNORECASE)
        return s

    # Group files by base name (ignoring column token)
    file_groups = {}
    for f in files:
        key = get_group_key(f)
        file_groups.setdefault(key, []).append(f)

    if verbose:
        print(f"Found {len(files)} PNGs -> creating {len(file_groups)} PDFs")

    # Sorting key: prefer numeric _col-, then numeric _smth-, then _stat- name, then filename
    def sort_key(fname):
        mcol = re.search(r"_col-([0-9]+)", fname)
        colv = int(mcol.group(1)) if mcol else float('inf')
        msm = re.search(r"_smth-(\d+)mm", fname)
        smv = int(msm.group(1)) if msm else -1
        mstat = re.search(r"_stat-([^_\.]+)", fname)
        stat = mstat.group(1) if mstat else ""
        return (colv, smv, stat, fname)

    # Create a PDF for each group
    for base_name, group_files in file_groups.items():
        # sort group: primarily by col value (if present), then smoothing, then stat
        group_files_sorted = sorted(group_files, key=sort_key)

        # Output PDF path
        output_pdf = os.path.join(output, f"{base_name}_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.pdf")

        # Open images and save them directly to a PDF
        images = []
        added_files = []  # track files successfully opened/added
        for file in group_files_sorted:
            file_path = os.path.join(fig_dir, file)
            try:
                img = Image.open(file_path)
            except Exception as e:
                if verbose:
                    print(f"\tSkipping {file_path}: {e}")
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            added_files.append(file_path)

        # Save all images to a single PDF
        if images:
            saved_ok = False
            try:
                images[0].save(output_pdf, save_all=True, append_images=images[1:])
                saved_ok = True
                if verbose:
                    print(f"\tPDF created: {output_pdf} ({len(images)} pages)")
            except Exception as e:
                if verbose:
                    print(f"\tFailed to write PDF {output_pdf}: {e}")
            finally:
                # close image objects to release file handles
                for im in images:
                    try:
                        im.close()
                    except Exception:
                        pass

            # optionally remove source files that were merged
            if cleanup and saved_ok:
                for fp in added_files:
                    try:
                        os.remove(fp)
                        if verbose:
                            print(f"\tRemoved source file: {fp}")
                    except Exception as e:
                        if verbose:
                            print(f"\tFailed to remove {fp}: {e}")
    return output_pdf 


def parcel_stats(dl, key, sv_root, meth = 'perc', test = False):
    """
    Compute statistics for distribution of vertex values within parcels.
    Note, by default only processes items with 'smth'=='NA' (no smoothing).

    Input:
    - dl: list of dictionarry items with keys:
        - study
        - region
        - feature
        - label
        - surf
        - smth
    - key: str
        - key in dl items with dataframe of parcellated values
        NOTE. Parcellation type taken from this string
    - sv_root: str
        - root folder for saving output
    - meth: str
        - method for selecting parcels to plot histograms for:
            - 'perc': keep parcels present in at least perc% of subjects (default)
            - 'std': identify parcels with large differences between mean and median
    - test: bool
        - if True, only runs on a single randomly choosen index (for testing)

    Outputs:
    - <saved> stats_df:
        - descriptive statistics per parcel (column name)
    - <saved> pdf of histograms for parcels with large mean-median differences
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import tTsTGrpUtils as tsutil
    import datetime
    import pandas as pd
    import numpy as np

    print(f"Computing parcel stats for key '{key}' in {len(dl)} items, saving to {sv_root}")

    if test:
        import random
        test_item = {}
        counter = 0
        while counter < 10 and test_item.get(key, None) is None or test_item.get(key, None).shape[0] == 0 or test_item.get('smth', None) != 'NA':
            idx = random.randint(0, len(dl)-1)
            test_item = dl[idx]
            counter += 1
            #print(f"[{counter}] Trying index {idx}: smth={test_item.get('smth', None)}, key present={key in test_item}")
            
        dl = [dl[idx]]        
        print(f"Test mode: processing only index {idx}.")

    log = ""

    if meth == 'perc':
        perc = 2.5  # keep parcels present in at least perc% of subjects
        print(f"\tPlotting parcels with largest mean-median values (top {perc}% of parcels).")
    elif meth == 'std':
        std = 2
        print(f"\tPlotting parcels with differences between mean and median > {std}*standard deviations of all mean/mdn difs in this item.")
    else:
        raise ValueError(f"Unknown method '{meth}'. Select 'perc' or 'std'.")
    for idx, item in enumerate(dl):
        if item['smth'] != 'NA':
            continue

        txt = tsutil.printItemMetadata(item, return_txt = True)
        print(txt)
        df = item.get(key, None)
        log += f"\n\n{txt}\tShape: {df.shape}\n"
        
        
        if df is None:
            log += f"\tNo data found for key '{key}'. Skipping."
            continue
        if not isinstance(df, (pd.DataFrame, np.ndarray)):
            log += f"\tData for key '{key}' is not a DataFrame or ndarray. Skipping."
            continue
        
        parc = key.split('-')[-1]
        
        if df is not None:
            df_sorted = sort_columns_numeric(df)            # reorder columns numerically (0.360); assumes glasser parcellation
            stats = group_stats_by_colname(df_sorted)      # descriptive stats per column name, sorted
            if meth == 'perc':
                threshold = np.percentile(stats['mn-mdn'], 100 - perc)
                df_highVar = stats[stats['mn-mdn'] >= threshold]  # keep only top 'perc' percentile of mn-mdn differences
            elif meth == 'std':
                df_highVar = stats[stats['mn-mdn'] > stats['mn-mdn'].std() * std] # identify parcels with large differences between mean and median

            if parc in ['glsr', "glasser"] and 0 in df_highVar.index: # if parcellation = glasser and index 0 is in list, remove it
                df_highVar = df_highVar.drop(index=0)
           
            if df_highVar.shape[0] > 0:
                df_highVar = df_highVar.sort_values(by='mn-mdn', ascending=False)
                log += f"\nParcels with higher mean-median dif: {list(df_highVar.index)}\n"
                for parcel in df_highVar.index:
                    plt.figure(figsize=(8,4))
                    sns.histplot(df_sorted[parcel], kde=True)
                    plt.axvline(stats.at[parcel, 'mean'], color='r', linestyle='--', label='Mean')
                    plt.axvline(stats.at[parcel, 'mdn'], color='g', linestyle='-', label='Median')
                    plt.legend()
                    plt.title(f"Histogram of {parc}:{parcel} (mn-mdn={stats.at[parcel, 'mn-mdn']:0.2f})")
                    plt.xlabel(item['feature'])
                    if test:
                        hist_save_name = f"TEST_glsr_hist_{item['study']}_{item['region']}_{item['feature']}_{item['label']}_{item['surf']}_{item['smth']}_col-{parcel}.png"
                    else:
                        hist_save_name = f"glsr_hist_{item['study']}_{item['region']}_{item['feature']}_{item['label']}_{item['surf']}_{item['smth']}_col-{parcel}.png"
                    plt.savefig(f"{sv_root}/{hist_save_name}", dpi=300)
                    plt.close()
                    
                output_pdf = pngs2pdf(sv_root, cleanup=True)
                
            log += f"{stats.to_string()}\n"
            log += f"Histograms of values with large differences between mean and median saved to {output_pdf}\n"
            log += "-"*50

    save_log = True
    if save_log:
        if test:
            log_pth = f"{sv_root}/TEST_parcel_distr_stats_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.txt"
        else:
            log_pth = f"{sv_root}/parcel_distr_stats_{datetime.datetime.now().strftime('%d%b%Y-%H%M%S')}.txt"
        with open(log_pth, 'w') as f:
            f.write(log)
        print(f"Saved [{os.path.getsize(log_pth) / (1024 * 1024):0.1f}MB]: {log_pth}")