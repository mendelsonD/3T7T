def runZBrain(study, zb_dir, ctx, hip, res, demo_ref, demo, score = "z", run_arg="proc analysis",dir_software="/host/verges/tank/data/daniel/z-brains/zbrains"):
    """
    Call zBrains software.
    Supports z-score (default) or w-scores. 
    For w-score, requires path to csv with participant_id, session_id, age, sex columns.

    run_arg <optional> : arguments to send to zBrains software. Default is "proc analysis". Options
        "proc" : for post-processing only
        "analysis" : for analysis only
    """
    
    import subprocess
    if score == "z":
        cmd = ["bash", dir_software, 
                            str(study["dir_root"]),
                            str(study["dir_mp"]),
                            str(study["dir_hu"]),
                            zb_dir,
                            str(study["ctrl_ptrn"]),
                            demo_ref,
                            demo,
                            str(ctx),
                            str(hip),
                            str(res)
                            ]
    elif score == "w":
        cmd = ["bash", dir_software, 
                            str(study["dir_root"]),
                            str(study["dir_mp"]),
                            str(study["dir_hu"]),
                            zb_dir,
                            str(study["ctrl_ptrn"]),
                            demo_ref,
                            demo,
                            str(ctx),
                            str(hip),
                            str(res)
                            ]
    else:
        raise ValueError("[runZBrain] Error. Invalid score type: %s" %score)

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result