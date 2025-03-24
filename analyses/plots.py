# functions to create stacked histograms

def histStack(df):
    """
    Return a stacked histogram of the data in the input dataframe.

    import:
        df: pandas dataframe, each column a list of values to create a unique histogram from
            e.g.: each column is a single participant; or each column is a different smoothing kernel in same participant; or each column is a different image resoltuion
    
    return:
        fig: plotly figure object, a 3D stacked histogram of the input data

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