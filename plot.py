import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot Melvin save files')
parser.add_argument('filename', help='file to plot')
parser.add_argument('--aspect_ratio', default=2,  help='aspect ratio of physical data')
parser.add_argument('--contour', action='store_true', help='enables contour plotting')
parser.add_argument('--output', help='output file')
parser.add_argument('--slice', nargs=2, type=int, help='axis to slice along')
args = parser.parse_args()

filename = args.filename

data = np.load(filename)
if data.dtype == 'complex128':
    mode = 'spectral'
    data = np.absolute(data)
else:
    mode = 'physical'

if args.slice:
    slice_axis = args.slice[0]
    slice_idx = args.slice[1]
    data_slice = data.take(slice_idx, axis=slice_axis)
    plt.plot(data_slice)
else:
    if mode == 'spectral':
        img = plt.imshow(data.T, origin='lower')
        plt.colorbar(img)
    elif mode == 'physical':
        extent = [0, 1, 0, args.aspect_ratio]
        if args.contour:
            plt.contour(data.T, origin='lower', extent=extent)
        else:
            img = plt.imshow(data.T, origin='lower', extent=extent)
            plt.colorbar(img)

plt.axis('off')
plt.tight_layout()

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
