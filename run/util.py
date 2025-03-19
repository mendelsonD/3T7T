def runZBrain(df_path, study, ctx, hip, res, demo, score = "z", dir_software="/host/verges/tank/data/daniel/z-brains/zbrains"):
    """
    Call zBrains software.
    Supports z-score (default) or w-scores. 
    For w-score, requires path to csv with participant_id, session_id, age, sex columns.
    """
    
    import subprocess
    if score == "z":
        cmd = ["bash", dir_software, 
                            str(df_path),
                            str(study["dir_root"]),
                            str(study["dir_mp"]),
                            str(study["dir_hu"]),
                            str(study["zb"]),
                            str(study["ctrl_ptrn"]),
                            str(ctx),
                            str(hip),
                            str(res)
                            ]
    elif score == "w":
                cmd = ["bash", dir_software, 
                            str(df_path),
                            str(study["dir_root"]),
                            str(study["dir_mp"]),
                            str(study["dir_hu"]),
                            str(study["zb"]),
                            str(study["ctrl_ptrn"]),
                            str(ctx),
                            str(hip),
                            str(res),
                            str(demo)
                            ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result