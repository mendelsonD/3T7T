import datetime
import pandas as pd

def tempCSV(df, subject, session=None, subject_colName="ID", session_colName=None, save=False, path_save=None):
    """
    Remove a subject from a csv file and returns a csv file with the subject removed.
    Optionally removes only a subject at a specified session.
    Optionally saves the new csv file to specified path; else, returns the new csv file.

    Parameters
        df: dataframe with subject and session IDs
        subject: subject ID to remove from list
        <optional> session: unique session ID for `subject` if want to remove only this subject's session
        subject_colName (default: "ID"): name of column in `csv` that contains subject IDs
        <optional> session_colName: name of column in `csv` that contains session IDs. Must be provided if `session` is not None
        <optional> save (default: false): if True, save the new csv to `path` and does not return the temporary ddf
        <optional> path: path to save the new csv. Must be provided if `save` is True.

    Returns
        tmp_csv: csv with row corresponding to `subject`-`session` removed

    """

    if subject_colName not in df.columns:
        raise ValueError(f"[tempCSV] ERROR. Provided subject column name `{subject_colName}` is not in columns of provided df.")

    # remove subject
    if session is None:
        tmp_df = df[df[subject_colName] != subject]

    else:
        if session_colName is None:
            raise ValueError("[tempCSV] ERROR. `session_colName` must be provided if session is specified.")
        else:

            if session_colName not in df.columns:
                raise ValueError(f"[tempCSV] ERROR. `{session_colName}` is not a column of the input csv.")
            else:
                tmp_df = df[(df[subject_colName] != subject) & (df[session_colName] != session)]

    if save:
        if path_save is None:
            raise ValueError("[tempCSV] ERROR. Unable to save output as no path provided in `path_save` parameter.")
        else:
            tmp_df.to_csv(path_save, index=False)
            print(f"[tempCSV] {subject}: saved temporary csv to {path_save}")
    else:
        return tmp_df


sheet_path = "/host/verges/tank/data/daniel/zBrain_3T7T/data/pt_2Jan2025_manual.csv"
subject_colName = "tT_ID"
session_colName = "tT_ses_num"

#out_path = "/Users/danielmendelson/Documents/Boris_projects/code/output/tmp/" # path should end with `/`
out_path = "/host/verges/tank/data/daniel/zBrain_3T7T/data/ctrl_lists/"

# read file
#df = pd.read_excel(sheet_path)
df = pd.read_csv(sheet_path, converters={subject_colName: str, session_colName: str})
# df.head()

df_short = pd.concat(["sub-" + df[[subject_colName]], "ses-" + df[[session_colName]]], axis=1)
df_short.columns = ["ID", "SES"]

# Create control csv files for each participant
for subject in df_short["ID"]:

    date = datetime.datetime.now().strftime("%d%b%Y-%Hh%Mm%Ss")
    out_path_subj = out_path + f"{subject}_{date}.csv"
    #print(out_path_subj)

    tempCSV(df=df_short, subject=subject, subject_colName="ID", save=True, path_save=out_path_subj)
