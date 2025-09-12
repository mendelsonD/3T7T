# functions from z-brains



##### FROM z-brains/src/functions/utils_analysis.py
def map_resolution(struct: Structure, res: Resolution):
    # Resolution # from constants.py
    LOW_RESOLUTION_CTX = "5k"
    HIGH_RESOLUTION_CTX = "32k"
    LOW_RESOLUTION_HIP = "2mm"
    HIGH_RESOLUTION_HIP = "0p5mm"
    ##############################

    if struct == "cortex":
        return HIGH_RESOLUTION_CTX if res == "high" else LOW_RESOLUTION_CTX
    if struct == "hippocampus":
        return HIGH_RESOLUTION_HIP if res == "high" else LOW_RESOLUTION_HIP
    raise ValueError(f"Mapping resolution for unknown structure: {struct}")



###### FROM clinical_reports.py

def _make_png_hip(
    *,
    analysis,
    data_lh: np.ndarray,
    data_rh: np.ndarray,
    out_png: PathType,
    res: Resolution = "high",
    cmap="cmo.balance",
    color_range=(-2, 2),
    color_bar="bottom",
):

    lat_lh, mid_lh, unf_lh, unf_rh, mid_rh, lat_rh = _load_surfaces_hip(res=res)
    if analysis == "asymmetry":
        kwds = dict()
        kwds["text__textproperty"] = {"fontSize": 50}
        plot_surfs(
            surfaces=[lat_lh, mid_lh, unf_lh],
            values=[data_lh, data_lh, data_lh],
            views=["dorsal", "dorsal", "lateral"],
            color_bar=color_bar,
            zoom=1.75,
            cmap=cmap,
            color_range=color_range,
            interactive=False,
            screenshot=True,
            filename=out_png,
        )

    else:
        plot_surfs(
            surfaces=[lat_lh, mid_lh, unf_lh, unf_rh, mid_rh, lat_rh],
            values=[data_lh, data_lh, data_lh, data_rh, data_rh, data_rh],
            views=["dorsal", "dorsal", "lateral", "lateral", "dorsal", "dorsal"],
            color_bar=color_bar,
            zoom=1.75,
            cmap=cmap,
            color_range=color_range,
            interactive=False,
            screenshot=True,
            filename=out_png,
        )

    return (
        f'<p style="text-align:center;margin-left=0px;"> '
        f'<a href="{out_png}" target="_blank">'
        f'<img style="height:175px;margin-top:-100px;" src="{out_png}"> '
        f"</a> "
        f"</p>"
    )

def _load_surfaces_hip(res: Resolution = "high"):
    res_hip = map_resolution("hippocampus", res)
    label = "midthickness"

    pth_canonical = (
        f"{DATA_PATH}/tpl-avg_space-canonical_den-{res_hip}"
        f"_label-hipp_{label}.surf.gii"
    )
    pth_unfold = (
        f"{DATA_PATH}/tpl-avg_space-unfold_den-{res_hip}"
        f"_label-hipp_{label}.surf.gii"
    )

    mid_rh = read_surface(pth_canonical)
    unf_rh = read_surface(pth_unfold)
    mid_lh = read_surface(pth_canonical)
    unf_lh = read_surface(pth_unfold)

    # Flip right to left surface
    mid_lh.Points[:, 0] *= -1
    unf_lh.Points[:, 0] *= -1

    # vflip = np.ones(hipp_mid_l.Points.shape)
    # vflip[:, 0] = -1
    # hipp_mid_l.Points = hipp_mid_l.Points * vflip
    # hipp_unf_l.Points = hipp_unf_l.Points * vflip

    # Rotate surfaces because Reinder didn't accept my pull request

    # Rotate around Y-axis 270
    rot_y270 = Rotation.from_rotvec(3 * np.pi / 2 * np.array([0, 1, 0]))
    unf_rh.Points = rot_y270.apply(unf_rh.Points)

    # Rotate around X-axis 90
    rot_y90 = Rotation.from_rotvec(np.pi / 2 * np.array([0, 1, 0]))
    unf_lh.Points = rot_y90.apply(unf_lh.Points)

    # Rotate around Z-axis 180
    rot_z = Rotation.from_rotvec(np.pi * np.array([0, 0, 1]))
    unf_rh.Points = rot_z.apply(unf_rh.Points)

    # Right Antero-posterior lateral
    lat_rh = read_surface(pth_canonical)
    lat_rh.Points = rot_y270.apply(lat_rh.Points)

    # Left Antero-posterior lateral
    lat_lh = read_surface(pth_canonical)
    lat_lh.Points = rot_y90.apply(
        mid_lh.Points
    )  # TODO: should it be lat_lh instead of mid_lh?

    return lat_lh, mid_lh, unf_lh, unf_rh, mid_rh, lat_rh

def plot_surfs(
    surfaces,
    values: List[np.ndarray],
    views: Union[List[str], None] = None,
    size: Union[int, Tuple[int, int], None] = None,
    zoom: Union[float, List[float]] = 1.75,
    color_bar="bottom",
    share="both",
    color_range=(-2, 2),
    cmap="cmo.balance",
    transparent_bg=False,
    **kwargs,
):
    """
    surfaces = [hip_mid_l, hip_unf_l, hip_unf_r, hip_mid_r]  Can be 1 or more
    views = ['dorsal', 'lateral', 'lateral', 'dorsal'] Can be 1 or more
    """

    # Append values to surfaces
    my_surfs = {}
    array_names = []
    for i, surf in enumerate(surfaces):
        name = f"surf{i + 1}"
        surf.append_array(values[i], name=name)

        my_surfs[name] = surf
        array_names.append(name)

    # Set the layout as the list of keys from my_surfs
    layout = [list(my_surfs.keys())]

    if size is None:
        size = (200 * len(surfaces), 350)

    return plot_surf(
        my_surfs,
        layout,
        array_name=array_names,
        view=views,
        color_bar=color_bar,
        color_range=color_range,
        share=share,
        cmap=cmap,
        zoom=zoom,
        size=size,
        transparent_bg=transparent_bg,
        **kwargs,
    )