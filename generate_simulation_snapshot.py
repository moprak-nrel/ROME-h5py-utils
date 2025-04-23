import glob
import re
from pathlib import Path

import h5py as h5
import numpy as np
import yt


def read_single_snapshot(fname, fields, hdf5_cache_path):
    hfname = f"{hdf5_cache_path}/U_{fname[-6:]}.hdf5"
    if not Path(hfname).is_file():
        field_map = {
            "u": "velocityx",
            "v": "velocityy",
            "w": "velocityz",
            "p": "p",
        }
        hfields = [field_map[f] for f in fields]
        ds = yt.load(fname)
        spatial_dims = ds.domain_dimensions
        t = float(ds.current_time)
        data = ds.covering_grid(
            left_edge=ds.domain_left_edge.d,
            fields=hfields,
            dims=ds.domain_dimensions,
            level=ds.max_level,
        )
        snapshot = np.empty((len(fields), *spatial_dims))

        for i, f in enumerate(fields):
            snapshot[i] = data[field_map[f]][()]
    else:
        hfile = h5.File(hfname, "r")
        spatial_dims = hfile[fields[0]].shape
        t = hfile["t"][0]
        snapshot = np.empty((len(fields), *spatial_dims))
        for i, f in enumerate(fields):
            snapshot[i] = hfile[f][()]
        hfile.close()
    return (snapshot, t)


def create_snapshot(plt_path, hdf5_cache_path, start_idx=0, N=1000):
    lexsort = lambda s: [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
    ]

    ## Get list of files, and sort them
    files = glob.glob(f"{plt_path}/plt*")
    files.sort(key=lexsort)
    files = files[start_idx : start_idx + N]
    nt = len(files)
    print(f"Processing {nt} files")
    fields = ["u", "v", "w", "p"]
    units = {"times": "s", "fields": "", "x": "m", "y": "m", "z": "m"}
    dim_order = ["times", "fields", "x", "y", "z"]

    ds = yt.load(files[0])
    spatial_dims = ds.domain_dimensions
    plo = ds.domain_left_edge.d
    phi = ds.domain_right_edge.d
    x, y, z = [
        np.linspace(plo[i], phi[i], spatial_dims[i], endpoint=False) for i in range(3)
    ]
    times = np.zeros(nt)

    snapshot, _ = read_single_snapshot(files[0], fields, hdf5_cache_path)
    snapshot_dims = snapshot.shape
    full_dims = (nt, *snapshot_dims)

    # H5 file creation
    hfile = h5.File(outfile, "w")
    dset = hfile.create_dataset(
        "data",
        full_dims,
        chunks=(1, 1, *spatial_dims),
        dtype=np.float64,
    )
    for i, fname in enumerate(files):
        if i % 10 == 0:
            print(f"Read {i} of {nt} snapshots")
        snapshot, t = read_single_snapshot(fname, fields, hdf5_cache_path)
        times[i] = t
        dset[i] = snapshot

    data = {
        "x": x,
        "y": y,
        "z": z,
        "times": times,
        "fields": fields,
        "units": units,
        "dim_order": dim_order,
    }

    for i, dim in enumerate(dim_order):
        dim_dataset = hfile.create_dataset(dim, data=data[dim])
        dim_dataset.attrs["units"] = data["units"][dim]
        dim_dataset.make_scale(dim)
        dset.dims[i].label = dim
        dset.dims[i].attach_scale(dim_dataset)

    hfile.close()
    print(f"Successfully created h5 file: {outfile} with dimensions: {full_dims}")


if __name__ == "__main__":
    prefix = "aligned"
    plt_path = f"/scratch/pmohan/two_turbine/{prefix}"
    hdf5_cache_path = f"/scratch/pmohan/two_turbine/hdf5_cache/{prefix}"
    N = 10
    start = 13900
    end = start + N
    outfile = f"/scratch/pmohan/two_turbine/data_{prefix}_{start}_{end}.h5"
    print(f"Processing files from start: {start} to end: {end} into {outfile}")
    if not Path(outfile).is_file():
        data = create_snapshot(plt_path, hdf5_cache_path, start, end - start)
