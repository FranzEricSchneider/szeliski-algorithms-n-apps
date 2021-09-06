"""
Best guess: Extract SIFT features to build a Bag-of-Words representation of an
image for classification
"""

import argparse
import cProfile
from matplotlib import pyplot
from pathlib import Path
import time

import cv2
import numpy
from skimage.feature import hessian_matrix


# Always...
numpy.set_printoptions(suppress=True, precision=6)


# TODO: Explain. Test?
# DIVISIONS_PER_OCTAVE = 5  # TODO: Why is this causing a crash??? Suspicious
DIVISIONS_PER_OCTAVE = 4
SCALES = [2**(i/DIVISIONS_PER_OCTAVE) for i in range(DIVISIONS_PER_OCTAVE)]

# WTF size should the gaussian kernel be? This is never stated. Should it be
# dependent on the size of the stddev? How about 16x16 because of the SIFT
# sizing later? Actually how about 15x15 so that there's a defined central
# pixel. This is totally ad hoc.
KERNEL_SIDE_LEN = 15
# While we're making ad-hoc decisions, what should the initial stddev be? I
# chose a value based on the side length, where a value of 2*BASE approximately
# covered the gaussian with meaningful values, and 1*BASE was somewhat confined
# to the center. Arbitrary.
BASE_STDDEV = 1.2
KERNELS = [cv2.getGaussianKernel(KERNEL_SIDE_LEN, scale * BASE_STDDEV)
           for scale in SCALES]

NUM_OCTAVES = 4

# Global cache of image derivatives, taken on the diffed (DoG) images. Stored
# by an int tuple of (octave index, image index within the octave).
DIFF_DERIVATIVES = {}
# Same story but with hessians (double derivatives)
DIFF_HESSIANS = {}
# Then the same thing again but caching the blurred images (no hessian needed)
BLUR_DERIVATIVES = {}

# This is "the standard deviation used for the Gaussian kernel, which is used
# as weighting function for the auto-correlation matrix." It's absolutely
# arbitrary at the moment, this is what was in the example. I found someone
# online setting it to 3.0 as well.
HESSIAN_SIGMA = 0.1

# Taken straight from the paper, could be tested
# CONTRAST_THRESHOLD = 0.03
CONTRAST_THRESHOLD = 0.01
PRINCIPAL_RATIO = 10
PRINCIPAL_THESHOLD = (PRINCIPAL_RATIO + 1)**2 / PRINCIPAL_RATIO

# Fractional threshold where we count multiple competing orientation options.
# E.g. if the threshold is 0.8, then all orientation peaks above 80% of the
# max peak are all counted as possible orientations
ORIENT_THRESHOLD = 0.8


def main(image, profile, plot_update, plot_filtered, plot_keypoints):

    # Start profiling if the flag is set
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # There are many possible keypoint detectors. Let's use the one described
    # in the original SIFT paper, Difference of Gaussians (DoG). Apparently
    # this is a computationally efficient version of Laplacian of Gaussians.
    keypoints = detect_features(image, plot_update, plot_filtered)
    if plot_keypoints:
        display_keypoints(image, keypoints)

    # This process is fairly simple, and is described well in Szeliski
    # sifted = siftify(keypoints)

    # codebook = find_words(sifted)
    # distribution = find_distribution(codebook, sifted)

    # Write out profile messages for speed-up attempts if the flag is set
    if profile:
        profiler.disable()
        filename = f"profile_{int(time.time()*1e6)}.snakeviz"
        profiler.dump_stats(filename)
        print(f"Wrote profile stats to {filename}")


def detect_features(original, plot_update, plot_filtered):
    """
    Inspired by Szeliski section 7.1.1, as well as:
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
    """

    # Scale the image down to 0-1 for two reasons:
    # 1) It matches the paper
    # 2) It prevents any uint8 step aliasing along the way
    image = original.astype(float) / numpy.max(original)

    downsampled = downsample(image, NUM_OCTAVES)
    blurred = blur(downsampled, KERNELS)
    diffed = differences(blurred)
    extrema = list(extremities(diffed))
    valid, low_contrast, edge = adjust_and_filter(
        extrema, diffed, blurred, plot_update
    )
    if plot_filtered:
        print(f"valid: {len(valid)}")
        print(f"low_contrast: {len(low_contrast)}")
        print(f"edge: {len(edge)}")
        figure, axes = pyplot.subplots(1, 3)
        display_keypoints(original, valid, axis=axes[0], color=(0, 255, 0),
                          show=False, title="Valid Keypoints")
        display_keypoints(original, low_contrast, axis=axes[1], color=(255, 0, 0),
                          show=False, title="Low-Contrast Filtered Keypoints")
        display_keypoints(original, edge, axis=axes[2], color=(0, 0, 255),
                          show=False, title="Edge Filtered Keypoints")
        pyplot.show()
    return valid


def downsample(image, num_octaves):
    """Downsample an image recursively until we have the right number.

    Arguments:
        image: Grayscale image, (n, m) matrix of undetermined dtype
        num_octaves: (int) How many octaves we want to end up with. Octave 0
            will just be the image again, then 1..N-1 will each be downsampled
            by a factor of 2.

    Returns: List of images of len num_octaves, where the first element is the
        original image and they will get smaller from there.
    """
    downsampled = [image.copy()]
    for i in range(1, num_octaves):
        downsampled.append(image[::2**i, ::2**i])
    return downsampled


def blur(images, kernels):
    """Blur all incoming images by a set of kernels.

    Arguments:
        images: A list of grayscale images of undetermined dtype
        kernels: A list of kernels, e.g. from cv2.getGaussianKernel. These
            should be (n, 1) gaussian arrays, type float.

    Returns: A list of len(images), where each list element has another list of
        len(kernels). At each level i the list of kernels will have been
        convolved with images[i]
    """
    blurred = []
    for image in images:
        octave = []
        for kernel in kernels:
            # BTW, the border behavior is likely cv2.BORDER_REFLECT101, this
            # has the same value as cv2.BORDER_DEFAULT
            octave.append(cv2.sepFilter2D(
                src=image, ddepth=-1, kernelX=kernel, kernelY=kernel,
            ))
        blurred.append(octave)
    return blurred


def differences(blurred):
    """Get differences between each image in the octaves.

    Arguments:
        blurred: See blur() docstring for output details. Dtype must be float

    Returns: almost the same as before, but with length 1 less within each
        octave. That's for the simple reason that getting deltas between
        elements in a list reduces the length by 1.
    """
    diffed = []
    for images in blurred:
        # Check the dtype
        for image in images:
            assert image.dtype == float
        # Then collect the diffed images
        octave = []
        for i in range(len(images) - 1):
            octave.append(images[i + 1] - images[i])
        diffed.append(octave)
    return diffed


def extremities(diffed):
    """
    Get extrema sandwiched between the outer layers. Something is marked as
    extrema if it is higher OR lower than ALL its neighbors.

    Note that extrema are determined by their 26 neighbors (9 on upper/lower
    images, 8 around the pixel in its image). This takes 3 images to find, and
    means that extrema cannot come from the outer images. This also means that
    extrema cannot come from the other ring of pixels.

    Arguments:
        diffed: See the docstring of differences for the output. That's
            basically a series of image-like arrays, arranged in groups by
            downsample size. So the first group will be (n, n), the next
            (n/2, n/2), etc.

    Yields: Elements in a list like [[o, k, i, j], ...] where o is the index of
        the current octave (e.g. 0 for the first group of images), k is the
        index of the image within the group (note that this is limited between
        1 and len(octave) - 2) (also not sure if this will be used later, but
        hey bookkeeping), and (i, j) is the pixel location (x/y index) of the
        extrema.
    """

    # TODO: Would it be faster to construct or modify a mask based on i, j?
    def neighbors_minmax(o, k, i, j):
        """Helper function to find the min/max values around a given pixel."""
        neighbors = []
        for image_index in [k-1, k+1]:
            neighbors.extend(diffed[o][image_index][i-1:i+2, j-1:j+2].flatten())
        # Avoid the central pixel
        central = diffed[o][k][i-1:i+2, j-1:j+2].flatten()
        neighbors.extend(central[:4])
        neighbors.extend(central[5:])
        return numpy.min(neighbors), numpy.max(neighbors)

    # Go through the octaves and diffed images to find extrema. An important
    # note is that
    # 1) We go from [1:-1] in the diffed images, so that we always have a
    #    diffed image on either side to compare to
    # 2) We go from [8:-8] on the pixel counts, so that there's always an 8
    #    pixel buffer around our selected pixels. This is important for SIFT,
    #    which requires a 16x16 grid
    for octave_index, octave in enumerate(diffed):
        # This takes advantage of the fact that zip stops when the shorter
        # list (octave[1:-1]) ends, so I didn't bother making range() cover
        # the same area.
        for image_index, image in zip(range(1, len(octave)), octave[1:-1]):
            # Go through each pixel
            # TODO: Is this too slow? Do we have a faster alternative?
            for i in range(8, image.shape[0] - 8):
                for j in range(8, image.shape[1] - 8):
                    # If we have an extreme point, yield
                    vector = [octave_index, image_index, i, j]
                    nmin, nmax = neighbors_minmax(*vector)
                    if (image[i, j] < nmin) or (image[i, j] > nmax):
                        yield vector


def adjust_and_filter(extrema, diffed, blurred, plot_update):

    valid = []
    low_contrast = []
    edge = []

    for o, k, i, j in extrema:

        # TODO: Refactor this shit, too many inputs and outputs
        try:
            d, dx, ddx, x_adjusted, i_adjusted, j_adjusted = \
                localize(diffed, o, k, i, j, plot_update)
        except numpy.linalg.LinAlgError:
            # In the case of a singular matrix, let's toss out the extremity.
            # I'm not 100% sure, but I think that cases where the matrix would
            # be singular lines up with times where we don't care (e.g. if the
            # rows are not independent that means the axes of the transform
            # overlap and... maybe we're on a line?). I suspect it won't happen
            # much in real life.
            continue

        # Save the adjusted position, as well as the addition of the detected
        # orientation of the keypoint
        new_extrema = [
            [o, k, i_adjusted, j_adjusted, x_adjusted, orientation]
            for orientation in detect_orientation(
                blurred, o, k, i_adjusted, j_adjusted, x_adjusted
            )
        ]

        # Calculate an interpolated value and filter out if the abs value is
        # too low (formula straight from the paper). This SHOULD weed out low
        # contrast points
        d_adjusted = d + 0.5*dx.dot(x_adjusted)
        if abs(d_adjusted) < CONTRAST_THRESHOLD:
            low_contrast.extend(new_extrema)
            continue

        # Calculate the ratio of the principal axes of the hessian. The
        # threshold SHOULD weed out edge-only points
        if numpy.trace(ddx)**2 / numpy.linalg.det(ddx) > PRINCIPAL_THESHOLD:
            edge.extend(new_extrema)
            continue

        # Hey, we're here! Past the filter steps! Hypothetically this is a good
        # keypoint
        valid.extend(new_extrema)

    return valid, low_contrast, edge


def localize(diffed, o, k, i, j, plot_update):

    def get_gradients(o, k, i, j):
        """Helper function to recalculate gradients."""
        dx = get_cache(diffed, o, k, (i, j), DIFF_DERIVATIVES, derivative)
        ddx = get_cache(diffed, o, k, (i, j), DIFF_HESSIANS, hessian)
        inv_ddx = numpy.linalg.inv(ddx)
        # This is straight from the paper. We can calculate the updated
        # position where the derivative against x is 0
        x_adjusted = -inv_ddx.dot(dx)
        return dx, ddx, x_adjusted

    # Get cached gradient info, then localize the point until we've hit the
    # best spot
    dx, ddx, x_adjusted = get_gradients(o, k, i, j)
    i_adjusted_int = i
    j_adjusted_int = j
    # TODO: Investigate the difference between if/while, and which is a better
    # idea. I think it got stuck in an infinite loop with while where some
    # values kept tracking around?
    # # For now, make a count of repeated updates. I'm curious
    # counter = 0
    # while (abs(x_adjusted) > 0.5).any():
    if (abs(x_adjusted) > 0.5).any():
        # counter += 1
        i_adjusted_int = int(numpy.round(i + x_adjusted[0]))
        j_adjusted_int = int(numpy.round(j + x_adjusted[1]))
        dx, ddx, x_adjusted = get_gradients(o, k, i_adjusted_int, j_adjusted_int)
    # if counter > 1:
    #     print(f"Hit counter {counter} on original {o, k, i, j}, final"
    #           f" {o, k, i_adjusted_int, j_adjusted_int}")
    # # I think this is the case, examine any case where it's false and rework
    # assert (abs(x_adjusted) < 0.5).all()

    if plot_update:
        diff_image = diff_to_uint8(diffed[o][k])
        image = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2RGB)
        image[i_adjusted_int, j_adjusted_int] = (0, 255, 0)
        image[i, j] = (255, 0, 0)

        radius = 5
        low_i = max(i_adjusted_int - radius, 0)
        low_j = max(j_adjusted_int - radius, 0)

        figure, axes = pyplot.subplots(1, 2)
        for axis, plot_image in zip(axes, (diff_image, image)):
            axis.imshow(plot_image[low_i:low_i + 2*radius + 1,
                                   low_j:low_j + 2*radius + 1])
        pyplot.title(f"Localization ({i}, {j}) -> ({i_adjusted}, {j_adjusted})")
        pyplot.show()

    return (diffed[o][k][i_adjusted_int, j_adjusted_int],
            dx,
            ddx,
            x_adjusted,
            i_adjusted_int,
            j_adjusted_int)


def diff_to_uint8(diff_image, washout=20):
    """Turn a 1-channel diff image (float, possible negatives) to 0-255.

    NOTE: for visual purposes, we will try to make a certain percentage of the
    image pure black and white. The original images are very blah and grey.
    """
    uint8 = diff_image - numpy.percentile(diff_image, washout)
    uint8 *= 255 / numpy.percentile(uint8, 100 - washout)
    return numpy.clip(uint8, 0, 255).astype(numpy.uint8)


def get_cache(matrices, octave_index, image_index, pixel, cache, function):
    """Helper function to help cache and retrieve derivative calculations."""
    key = (octave_index, image_index)
    if key not in cache:
        cache[key] = function(matrices[octave_index][image_index])
    if pixel is None:
        return cache[key]
    else:
        return cache[key][pixel]


def derivative(image):
    """Apply the Scharr operator to the image, getting (dx, dy) for each pixel.

    Inspired by this. Apparently Scharr is more accurate than Sobel for (3, 3)
    kernels? That said, maybe I should be using something more than (3, 3)?
        https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html

    This was a very nice explanation of how to derivatize images with some code:
    https://towardsdatascience.com/image-derivative-8a07a4118550

    Arguments:
        image: Greyscale image, theoretically should be one of the DoG images

    Returns: Approximately same size image out, but instead of an (n, m) image
        it is (n, m, 2), where the last two elements are the x (axis 0) and
        y (axis 1) derivative values.
    """
    src = image.astype(float)
    # The Scharr kernal apparently needs 1/32 in order to be normalized. I
    # tested this by making a simple image that should have slope 1 (pixel
    # values [1, 2, 3, 4, 5]) and checking that. The Scharr kernel apparently
    # looks something like [[-3, -10, -3], [0, 0, 0], [3, 10, 3]] (abs sum 32)
    scale_factor = 1 / 32
    return numpy.dstack((
        cv2.Scharr(src=src, ddepth=-1, dx=1, dy=0, scale=scale_factor),
        cv2.Scharr(src=src, ddepth=-1, dx=0, dy=1, scale=scale_factor),
    ))


def hessian(image):
    """Get the hessian (double derivative) for each pixel in the given image.

    Working from this:
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hessian_matrix

    Arguments:
        image: Greyscale image, theoretically should be one of the DoG images

    Returns: Approximately same size image out, but instead of an (n, m) image
        it is (n, m, 2, 2), where each pixel is mapped to a
        [[dxx, dxy], [dxy, dyy]] matrix. Note that x is along axis 0, y is
        along axis 1.
    """
    rr, rc, cc = hessian_matrix(image,
                                sigma=HESSIAN_SIGMA,
                                order="rc",
                                mode="mirror")
    # Turn dxx, dxy, dyy into a (2, 2) matrix at each pixel
    return numpy.stack([numpy.dstack([rr, rc]),
                        numpy.dstack([rc, cc])],
                       axis=3)


def detect_orientation(blurred, o, k, i, j, x_adjusted):
    """TODO."""

    # First get the gradients of the entire image
    gradients = get_cache(blurred, o, k, None, BLUR_DERIVATIVES, derivative)
    # Then select down to just the square section we need, around the pixel of
    # interest
    gradients = get_quadrants(gradients, i, j, x_adjusted, side_len=16)
    # Turn into magnitudes and angles so we can bin by angle
    magnitudes = numpy.linalg.norm(gradients, axis=2)
    angles = numpy.arctan2(gradients[:, :, 1], gradients[:, :, 0])
    # Coerce angles to 0-2pi
    angles %= 2 * numpy.pi
    # Scale the weights by a gaussian
    kernel_1d = cv2.getGaussianKernel(16, SCALES[k] * BASE_STDDEV * 1.5)
    kernel_2d = numpy.outer(kernel_1d, kernel_1d)
    magnitudes_scaled = magnitudes * kernel_2d
    # Bin according to angle, in 36 bins
    histogram, bin_edges = numpy.histogram(
        a=angles.flatten(),
        bins=36,
        range=(0, 2*numpy.pi),
        weights=magnitudes_scaled.flatten()
    )
    # Detect all regions greater than X% of the maximum
    for index in numpy.argwhere(
                histogram > (ORIENT_THRESHOLD * numpy.max(histogram))
            ).flatten():
        yield numpy.average(bin_edges[index:index+2])

    # TODO: TEST!
    # TODO: Display!


def get_quadrants(matrix, i, j, adjustment, side_len=16):
    """
    Select a square of pixels, sub-dividable into quadrants (assuming a
    reasonable side_len is used). There is no central pixel here, instead we
    base the slice on the chosen pixel and the adjustment. The slice is
    constructed so that (i, j) + adjustment is approximately in the center of
    the slice.

    Arguments:
        matrix: ndarray of shape (m, n, ...). It doesn't matter what the shape
            is past the first two, as we select the pixels whatever they are.
        i, j: int, pixel indices into (axis0, axis1) of the matrix
        adjustment: two-element ndarray, float
        side_len: int, side length of the selected square. Must be divisible
            by 4. It will likely just always stay at the default

    Returns: matrix of shape (side_len, side_len, ...), selected from the
        incoming matrix.
    """
    radius = side_len // 2

    if adjustment[0] < 0:
        i -= 1
    if adjustment[1] < 0:
        j -= 1
    return matrix[i-radius+1:i+radius+1, j-radius+1:j+radius+1]


def display_keypoints(image, keypoints, axis=None, color=(255, 0, 0), show=True,
                      title="Keypoints of varying scales"):
    plot_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for o, k, i, j, _, theta in keypoints:
        scalar = (2**o)
        cv2.circle(img=plot_image,
                   center=(scalar * j, scalar * i),
                   radius=2,
                   color=color,
                   thickness=-1)
        cv2.rectangle(img=plot_image,
                      pt1=(scalar * (j - 8), scalar * (i - 8)),
                      pt2=(scalar * (j + 8), scalar * (i + 8)),
                      color=color,
                      thickness=1)
    if axis is None:
        pyplot.imshow(plot_image)
        pyplot.title(title)
    else:
        axis.imshow(plot_image)
        axis.set_title(title)

    if show:
        pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("image",
                        help="Path to image we want to bag of words.",
                        type=Path)
    parser.add_argument("-f", "--plot-filtered",
                        help="Whether to display the filtering process.",
                        action="store_true")
    parser.add_argument("-k", "--plot-keypoints",
                        help="Whether to display image keypoints.",
                        action="store_true")
    parser.add_argument("-u", "--plot-update",
                        help="Whether to display the point update process.",
                        action="store_true")
    parser.add_argument("-p", "--profile",
                        help="Capture profile information of the process.",
                        action="store_true")
    args = parser.parse_args()

    assert args.image.is_file()
    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2GRAY)

    main(
        image=image,
        profile=args.profile,
        plot_update=args.plot_update,
        plot_filtered=args.plot_filtered,
        plot_keypoints=args.plot_keypoints,
    )
