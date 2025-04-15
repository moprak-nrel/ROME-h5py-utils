
import argparse

import h5py as h5
import numpy as np


def write_test_h5_file(outfile):
    """Creates a test hdf5 file which has the dimensions of {time, field, x, y, z}
       and appropriate hdf5 attributes/scales for each dimension
    """
    nx, ny, nz =  32, 32, 32
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    z = np.linspace(0, 1, nz, endpoint=False)

    spatial_dims = (nx,ny,nz)
    fields = ["u", "v", "w", "p"]

    def get_snapshot(spatial_dims, time):
        res = np.empty((len(fields), *spatial_dims))
        snapshot = {}
        for i, f in enumerate(fields):
            snapshot[f] = np.random.rand(*spatial_dims)
            res[i] = snapshot[f]
        return res
    snapshot_dims = (len(fields), *spatial_dims)
    times = np.arange(0,10)
    full_dims = (len(times), *snapshot_dims)

    full_data = np.empty(full_dims)
    for i, t in enumerate(times):
        full_data[i] = get_snapshot(spatial_dims, t)

    hfile = h5.File(outfile, "w")
    dset = hfile.create_dataset("data", data=full_data, chunks = (1, 1, *spatial_dims))
    dimensions = {"x" : x, "y" : y, "z" : z, "times" : times, "fields": fields}
    dim_order = ["times", "fields", "x", "y", "z"]
    dim_units = {"times": "s", "fields": "", "x": "m", "y": "m", "z": "m"}
    for i, dim in enumerate(dim_order):
        dim_dataset = hfile.create_dataset(dim, data = dimensions[dim])
        dim_dataset.attrs["units"] = dim_units[dim]
        dim_dataset.make_scale(dim)
        dset.dims[i].label = dim
        dset.dims[i].attach_scale(dim_dataset)
    hfile.close()
    print(f"Successfully created h5 file: {outfile} with dimensions: {full_dims}")

if __name__ == "__main__":
    """Create test hdf5 file"""
    parser = argparse.ArgumentParser(
        description=
        """Creates a test hdf5 file which has the dimensions of {time, field, x, y, z}
           and appropriate hdf5 attributes/scales for each dimension
        """
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="name of test file",
        required=False,
        type=str,
        default = "./test.h5"
    )
    args = parser.parse_args()
    write_test_h5_file(args.outfile)
