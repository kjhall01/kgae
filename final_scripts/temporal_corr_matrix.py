import numpy as np
import xarray as xr

def corr_matrix(da: xr.DataArray, time_dim: str = "time", mode_dim: str = "mode") -> xr.DataArray:
    """
    Compute the Pearson correlation matrix across `time_dim` for each pair of `mode_dim`.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dims (time_dim, mode_dim)
    time_dim : str
        Name of time dimension
    mode_dim : str
        Name of mode dimension

    Returns
    -------
    xr.DataArray
        Correlation matrix with dims (mode_dim, mode_dim_2)
    """
    if set([time_dim, mode_dim]) - set(da.dims):
        raise ValueError(f"da must have dims ({time_dim}, {mode_dim}); got {da.dims}")

    # Ensure float + common time axis; xr.corr handles NaNs pairwise.
    da = da.transpose(time_dim, mode_dim).astype(float)

    modes = da[mode_dim].values
    out = np.full((len(modes), len(modes)), np.nan, dtype=float)

    for i, mi in enumerate(modes):
        xi = da.sel({mode_dim: mi})
        for j, mj in enumerate(modes):
            xj = da.sel({mode_dim: mj})
            out[i, j] = float(xr.corr(xi, xj, dim=time_dim))

    return xr.DataArray(
        out,
        coords={mode_dim: modes, f"{mode_dim}_2": modes},
        dims=(mode_dim, f"{mode_dim}_2"),
        name="corr",
    )


def corr_to_latex_table(
    corr: xr.DataArray,
    row_labels=None,
    col_labels=None,
    *,
    caption: str = "",
    label: str = "tab:corr",
    fmt: str = "{:.2f}",
    dash: str = "-",
    bold_pairs=None,
    bold_thresh: float | None = None,
) -> str:
    """
    Render a lower-triangular correlation matrix as LaTeX (booktabs), with '-' above diagonal.

    Parameters
    ----------
    corr : xr.DataArray
        dims (mode, mode_2), square
    row_labels, col_labels : list[str] | None
        Display labels. Defaults to corr coords.
    caption : str
    label : str
        LaTeX label WITHOUT backslashes, e.g. "tab:Atable"
    fmt : str
        Numeric format
    dash : str
        What to print above diagonal
    bold_pairs : set[tuple[str,str]] | list[tuple[str,str]] | None
        Pairs (row_label, col_label) to bold (lower triangle entries).
    bold_thresh : float | None
        If set, bold any |corr| >= bold_thresh (lower triangle, excluding diagonal).
    """
    mode = corr.dims[0]
    mode2 = corr.dims[1]
    if corr.sizes[mode] != corr.sizes[mode2]:
        raise ValueError("corr must be square")

    n = corr.sizes[mode]
    default_labels = [str(x) for x in corr[mode].values]
    row_labels = default_labels if row_labels is None else list(row_labels)
    col_labels = default_labels if col_labels is None else list(col_labels)

    if len(row_labels) != n or len(col_labels) != n:
        raise ValueError("row_labels/col_labels must match corr size")

    bold_pairs = set(bold_pairs) if bold_pairs is not None else set()

    def fmt_cell(i, j):
        if j > i:
            return dash
        v = float(corr.values[i, j])
        if np.isnan(v):
            s = dash
        else:
            s = fmt.format(v)

        if i == j or s == dash:
            return s

        rl, cl = row_labels[i], col_labels[j]
        do_bold = (rl, cl) in bold_pairs or (cl, rl) in bold_pairs
        if (bold_thresh is not None) and (not do_bold) and (not np.isnan(v)):
            do_bold = abs(v) >= bold_thresh

        return r"\\textbf{" + s + "}" if do_bold else s

    # Build LaTeX
    col_spec = "c|" + ("c" * n)
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"    \centering")
    lines.append(rf"    \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append("     & " + " & ".join(col_labels) + r" \\")
    lines.append(r"    \midrule")

    for i in range(n):
        row = [row_labels[i]] + [fmt_cell(i, j) for j in range(n)]
        lines.append("    " + " & ".join(row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    if caption:
        lines.append(rf"    \caption{{{caption}}}")
    lines.append(rf"\end{{table}}\label{{{label}}}")
    return "\n".join(lines)


if __name__ == "__main__":
    import xarray as xr 
    import kgae 

    modes = xr.open_dataset('/Users/kylehall/Desktop/kgae/final_scripts/climate_mode_indices.nc').magnitude
    latents = kgae.open_latents('xval', experiment='large-ensemble-2-1940-2014', n_ensemble=100).mean('seed')#['__xarray_dataarray_variable__']
    latents = latents.sel(mode=[5,4,3]).assign_coords({'mode': ['DM', 'IA', 'QB']})
    modes = xr.concat([latents, modes], 'mode')
    corr = corr_matrix(modes)
    tex = corr_to_latex_table(corr)
    print(tex)


# -----------------------------
# Example usage
# -----------------------------
# da: xr.DataArray with dims (time, mode)
# corr = corr_matrix(da)

# Option A: bold specific entries you reference in text
# bold = {("PDO", "BDM"), ("ONI", "IA"), ("ONI", "QB")}
# tex = corr_to_latex_table(
#     corr,
#     row_labels=["DM","IA","QB","PDO","ONI"],   # <-- display labels (optional)
#     col_labels=["BDM","IA","QB","PDO","ONI"],  # <-- display labels (optional)
#     bold_pairs=bold,
#     caption="Time-series correlation matrix, where DM indicates ...",
#     label="tab:Atable",
# )

# Option B: bold everything above a threshold (e.g., |r| >= 0.65)
# tex = corr_to_latex_table(corr, bold_thresh=0.65, caption="...", label="tab:Atable")

# print(tex)