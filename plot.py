import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path
import cmocean
from multiprocessing import Pool

MOVING_AVERAGE_N = 10

parser = argparse.ArgumentParser(description='Plot Melvin save files')
parser.add_argument('filenames', nargs='+', help='file to plot')
parser.add_argument('--aspect_ratio', default=0.5, type=float, help='aspect ratio of physical data')
parser.add_argument('--slice', nargs=2, type=int, help='axis to slice along')
parser.add_argument('--ncores', type=int, default=1, help='number of cores to use in parallel')
parser.add_argument('--figsize', nargs=2, type=int, default=[5, 10], help='figsize in inches')
parser.add_argument('--contour', action='store_true', help='enables contour plotting')
parser.add_argument('--colorbar', action='store_true', help='add colorbar')
parser.add_argument('--balance_cmap', action='store_true', help='ensure "middle" of colourmap represents value of 0')
parser.add_argument('--cmap', default=cmocean.cm.deep, help='matplotlib colormap')
parser.add_argument('--pretty', action='store_true', help='make pretty')
parser.add_argument('--smooth', action='store_true', help='smooth cmap changes using moving average')
parser.add_argument('--save', action='store_true', help='save to <inname>.png')
parser.add_argument('--replace', action='store_true', help='overwrite existing outputs')
args = parser.parse_args()

def calc_vmaxes(filenames):
    vmaxes = []
    for filename in filenames:
        data = np.load(filename)
        vmaxes.append(np.max(np.absolute(data)))
    return vmaxes

def calc_moving_average(data, window_width=MOVING_AVERAGE_N):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = np.zeros_like(data)
    ma_vec[:-window_width+1] = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    ma_vec[-window_width+1:] = ma_vec[-window_width]
    return ma_vec

def get_initial_vmax(filename):
    data = np.load(filename)
    data = np.absolute(data)
    return data.max()

def form_output_fname(filename):
    return filename + '.png'

def plot(v):
    i = v[0]
    filename = v[1]
    print(filename)
    output_fname = form_output_fname(filename)
    if os.path.isfile(output_fname) and not args.replace:
        print("Not overwriting", output_fname)
        return

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
            img = plt.imshow(data.T, origin='lower', cmap=args.cmap)
            if args.colorbar:
                plt.colorbar(img)
        elif mode == 'physical':
            extent = [0, args.aspect_ratio, 0, 1]
            if args.contour:
                plt.contour(data.T, origin='lower', extent=extent, cmap=args.cmap)
            else:
                if args.pretty:
                    fig, ax = plt.subplots(figsize=args.figsize)
                if args.balance_cmap:
                    vmax = np.absolute(data).max()
                    vmin = -vmax
                else:
                    vmax = data.max()
                    vmin = data.min()
                if args.smooth:
                    vmax = ma_vec[i]
                    vmin = -vmax
                img = plt.imshow(data.T, origin='lower', extent=extent,
                                 cmap=args.cmap, vmax=vmax, vmin=vmin)
                if args.colorbar:
                    plt.colorbar(img)

    if args.pretty:
        plt.axis('off')
    plt.tight_layout()

    if args.save:
        plt.savefig(output_fname)
    else:
        plt.show()

    plt.close()

if __name__ == '__main__':
    if args.smooth:
        vmaxes = calc_vmaxes(args.filenames)
        ma_vec = calc_moving_average(vmaxes)
    with Pool(args.ncores) as p:
        print(p.map(plot, enumerate(args.filenames)))
