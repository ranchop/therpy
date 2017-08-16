# Imagedata : manipulating image data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.special import erf


def get_cropi(data, center=None, width=None, height=None, point1=None, point2=None, point=None, **kwargs):
    # Prepare Output
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    [XX, YY] = np.meshgrid(x, y)
    cropi = (slice(None, None), slice(None, None))
    # Option 1 -- center width and height
    if center is not None:
        if width is None and height is None: return cropi
        if width is not None and height is None: height = width
        if width is None and height is not None: width = height
        xmin = max(0, center[0] - int(width / 2.0))
        xmax = min(x[-1], xmin + width)
        ymin = max(0, center[1] - int(height / 2.0))
        ymax = min(y[-1], ymin + height)
    # Option 2 -- point1 and point 2
    elif point1 is not None and point2 is not None:
        xmin = max(min(point1[0], point2[0]), 0)
        xmax = min(max(point1[0], point2[0]), x[-1]) + 1
        ymin = max(min(point1[1], point2[1]), 0)
        ymax = min(max(point1[1], point2[1]), y[-1]) + 1
    # Option 3 -- point and width and height
    elif point is not None:
        if width is None and height is None: return cropi
        if width is not None and height is None: height = width
        if width is None and height is not None: width = height
        xmin = max(point[0], 0)
        ymin = max(point[1], 0)
        xmax = min(point[0] + width, x[-1])
        ymax = min(point[1] + height, y[-1])
    else:
        return cropi
    # Return a np array of true false
    return (slice(ymin, ymax), slice(xmin, xmax))


def get_od(wa, woa, **kwargs):
    # Inputs
    width = kwargs.get('width', 5)
    rawod = kwargs.get('rawod', False)
    # Useful quantities
    X, Y = np.meshgrid(np.arange(wa.shape[1]), np.arange(wa.shape[0]))
    with np.errstate(divide='ignore', invalid='ignore'):
        od = np.log(woa / wa)
    od[np.logical_not(np.isfinite(od))] = np.nan
    # Return raw if asked
    if rawod: return od
    # Find trouble points
    pts = np.logical_not(np.isfinite(od))
    Xp, Yp = X[pts], Y[pts]
    # Average neighbors
    for i, x in enumerate(Xp):
        cropi = get_cropi(X, center=(Xp[i], Yp[i]), width=width)
        replace = od[cropi].flatten()
        replace = replace[np.isfinite(replace)]
        if replace.size == 0:
            replace = 0
        else:
            replace = np.mean(replace)
        od[Yp[i], Xp[i]] = replace
    # return fixed od
    return od

def fix_od(odIn, width = 5):
    # Inputs
    od = odIn.copy()
    # Find trouble points
    X, Y = np.meshgrid(np.arange(od.shape[1]), np.arange(od.shape[0]))
    od[np.logical_not(np.isfinite(od))] = np.nan
    pts = np.logical_not(np.isfinite(od))
    Xp, Yp = X[pts], Y[pts]
    # Average neighbors
    for i, x in enumerate(Xp):
        cropi = get_cropi(X, center=(Xp[i], Yp[i]), width=width)
        replace = od[cropi].flatten()
        replace = replace[np.isfinite(replace)]
        if replace.size == 0:
            replace = 0
        else:
            replace = np.mean(replace)
        od[Yp[i], Xp[i]] = replace
    # return fixed od
    return od

# usable pixels in an image
def get_usable_pixels(wa, woa, **kwargs):
    '''
	Inputs:
		threshold => (wa-dark) must be >= threshold for pixel to be usable; default 25

	Output:
		use
		True only if (wa-dark) >= threshold and rawod is a finite number
	'''
    # Input parser
    threshold = kwargs.get('threshold', 25)
    # Fill in
    rawod_finite = np.isfinite(get_od(wa, woa, rawod=True))
    threshold_condition = np.greater_equal(wa, threshold)
    return np.logical_and(rawod_finite, threshold_condition)


def com(data):
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    total = np.sum(data)
    x0 = np.sum(data * X) / total
    y0 = np.sum(data * Y) / total
    return (x0, y0)


def box_sharpness(data, **kwargs):
    # Treating data
    data[np.logical_not(np.isfinite(data))] = 0
    # Inputs
    plot = kwargs.get('plot', True)
    thickness = kwargs.get('thickness', 10)
    length = kwargs.get('length', 200)
    center = kwargs.get('center', com(data))
    using = kwargs.get('using', 'May16')
    guess_side = kwargs.get('guess_side', None)
    guess_top = kwargs.get('guess_top', None)
    threshold = kwargs.get('threshold', np.inf)
    # Default values
    if using == 'May16':
        thickness = 10
        length = 200
        guess_side = (0.1, 55, length / 2)
        guess_top = (2.5, length / 2 - 26, length / 2 + 26, 3, 3)
    # Get crop regions
    crop1 = get_cropi(data, center=center, width=length, height=thickness)
    crop2 = get_cropi(data, center=center, width=thickness, height=length)
    crop3 = get_cropi(data, center=center, width=length, height=length)
    # Cuts
    y1o = np.sum(data[crop1], axis=0)
    x1o = np.arange(y1o.size)
    y2o = np.sum(data[crop2], axis=1)
    x2o = np.arange(y2o.size)
    # Thresholds
    x1 = x1o[np.less(y1o, threshold)]
    y1 = y1o[np.less(y1o, threshold)]
    x2 = x2o[np.less(y2o, threshold)]
    y2 = y2o[np.less(y2o, threshold)]

    # Fitting functions
    def side_circle(x, amp, rad, x0):
        y = amp * np.real(np.sqrt(rad ** 2 - (x - x0) ** 2))
        y[np.isnan(y)] = 0
        return y

    def top_erf(x, amp, x1, x2, s1, s2):
        return amp * (erf((x - x1) / (np.sqrt(2) * s1)) + erf(-(x - x2) / (np.sqrt(2) * s2)))

    # Fits
    res1, _ = curve_fit(side_circle, x1, y1, p0=guess_side)
    res2, _ = curve_fit(top_erf, x2, y2, p0=guess_top)
    # Prepare outputs
    outp = {'res1': res1, 'res2': res2, 'thickness': res2[2] - res2[1], 'radius': res1[1]}
    outp['center'] = center
    outp['guess_side'] = guess_side
    outp['guess_top'] = guess_top
    # Plots
    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 3))
        axes[0].plot(x1o, y1o, x1, y1, x1o, side_circle(x1o, *res1))
        axes[1].plot(x2o, y2o, x2, y2, x2o, top_erf(x2o, *res2))
        axes[2].imshow(data)
        axes[2].scatter(*center)
        axes[2].add_patch(
            patches.Rectangle((center[0] - length / 2, center[1] - length / 2), length, length, fill=False))
        outp['plot'] = fig
    # return
    return outp


def plot_crop(data, *args, **kwargs):
    # Get cropi
    if len(args) > 0:
        if type(args[0]) is slice:
            cropi = args[0]
            center = None
        else:
            cropi = get_cropi(data, **kwargs)
            center = kwargs.get('center', None)
    # Plot
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.imshow(data)
    if center is not None: axes.scatter(*center)
    axes.add_patch






def main():
    # Tests of get_cropi
    data = np.ones((512, 256))
    cropi = get_cropi(data, center=(50, 100), width=10, height=50)
    print('center width height:', data[cropi].shape)
    cropi = get_cropi(data)
    print('nothing', data[cropi].shape)
    cropi = get_cropi(data, point1=(50, 50), point2=(100, 100))
    print('two points', data[cropi].shape)
    cropi = get_cropi(data, point=(50, 50), width=50)
    print('point and width', data[cropi].shape)
    plt.imshow(data[cropi], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
