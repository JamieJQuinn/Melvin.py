import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot Melvin dump files")
parser.add_argument("filename", help="file to plot")
parser.add_argument("--var", required=True, help="variable to plot")
parser.add_argument("--output", help="output file")
parser.add_argument("--slice", nargs=2, type=int, help="axis to slice along")
args = parser.parse_args()

filename = args.filename
var = args.var

npzdata = np.load(filename)
data = np.absolute(npzdata[var])

if args.slice:
    slice_axis = args.slice[0]
    slice_idx = args.slice[1]
    data_slice = data.take(slice_idx, axis=slice_axis)
    plt.plot(data_slice)
else:
    img = plt.imshow(data.T, origin="lower")
    plt.colorbar(img)

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
