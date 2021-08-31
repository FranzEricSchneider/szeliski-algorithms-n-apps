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
DIVISIONS_PER_OCTAVE = 5
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

# Global cache of image derivatives, stored by an int tuple of (octave index,
# image index within the octave).
DERIVATIVES = {}
# Same story but with hessians (double derivatives)
HESSIANS = {}

# This is "the standard deviation used for the Gaussian kernel, which is used
# as weighting function for the auto-correlation matrix." It's absolutely
# arbitrary at the moment, this is what was in the example. I found someone
# online setting it to 3.0 as well.
HESSIAN_SIGMA = 0.1


def main(image, profile, plot_keypoints):

    # Start profiling if the flag is set
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # There are many possible keypoint detectors. Let's use the one described
    # in the original SIFT paper, Difference of Gaussians (DoG). Apparently
    # this is a computationally efficient version of Laplacian of Gaussians.
    keypoints = detect_features(image)
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


def detect_features(image):
    """
    Inspired by Szeliski section 7.1.1, as well as:
        https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        https://en.wikipedia.org/wiki/Scale-invariant_feature_transform
    """

    # Scale the image down to 0-1 for two reasons:
    # 1) It matches the paper
    # 2) It prevents any uint8 step aliasing along the way
    image = image.astype(float) / numpy.max(image)

    downsampled = downsample(image, NUM_OCTAVES)
    blurred = blur(downsampled, KERNELS)
    diffed = differences(blurred)
    extrema = list(extremities(diffed))
    filtered = adjust_and_filter(extrema, diffed)
    # TODO: Add a filtering step
    return extrema


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


def adjust_and_filter(extrema, diffed):

    for o, k, i, j in extrema:

        dx = get_cache(diffed, o, k, (i, j), DERIVATIVES, derivative)
        ddx = get_cache(diffed, o, k, (i, j), HESSIANS, hessian)

        try:
            inv_ddx = numpy.linalg.inv(ddx)
        except numpy.linalg.LinAlgError:
            # In the case if a singular matrix, let's toss out the extremity.
            # I'm not 100% sure, but I think that cases where the matrix would
            # be singular lines up with times where we don't care (e.g. if the
            # rows are not independent that means the axes of the transform
            # overlap and... maybe we're on a line?). I suspect it won't happen
            # much in real life.
            continue

        # I've got a couple of problems here
        # 1) x_adjusted = -inv_ddx.dot(dx) appears to give a huge value for
        #    x_adjusted, in the range of 10-30 pixels. This seems very wrong.
        #    Maybe one of the derivative methods is scaled inappropriately?
        # 2) Just playing with D + dx.dot(x) + x.dot((ddx.dot(x))) for small
        #    x values like numpy.array([1, 1]) appears not to be a very good
        #    approximation. Maybe one of the derivative methods is scaled
        #    inappropriately? This may be a good time for some simple test
        #    images.
        import ipdb; ipdb.set_trace()


def get_cache(diffed, octave_index, image_index, pixel, cache, function):
    """Helper function to help cache and retrieve derivative calculations."""
    key = (octave_index, image_index)
    if key not in cache:
        cache[key] = function(diffed[octave_index][image_index])
    return cache[key][pixel]


def derivative(image):
    """Apply the Scharr operator to the image, getting (dx, dy) for each pixel.

    Inspired by this. Apparently Scharr is more accurate than Sobel for (3, 3)
    kernels? That said, maybe I should be using something more than (3, 3)?
        https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html

    Arguments:
        image: Greyscale image, theoretically should be one of the DoG images

    Returns: Approximately same size image out, but instead of an (n, m) image
        it is (n, m, 2), where the last two elements are the x (axis 0) and
        y (axis 1) derivative values.
    """
    return numpy.dstack((
        cv2.Scharr(src=image.astype(float), ddepth=-1, dx=1, dy=0),
        cv2.Scharr(src=image.astype(float), ddepth=-1, dx=0, dy=1),
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


def filter_low_contrast(downsampled, extrema):
    """Discard low-contrast keypoints

    Extremely helpful and clear for filtering:
    https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Keypoint_localization

    Hessian (double-derivative) is defined in my Terms notes. Here's some talk
    about how it would be computed theoretically:
    https://www.quora.com/What-are-the-ways-of-calculating-2-x-2-Hessian-matrix-for-2D-image-of-pixel-at-x-y-position

    I think we may need to calculate the derivative (convolution) and then the
    hessian (convolutions) on every pixel of every downsample

    This was a very nice explanation of how to derivatize images with some code:
    https://towardsdatascience.com/image-derivative-8a07a4118550
    """
    pass


def display_keypoints(image, keypoints):
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for o, k, i, j in keypoints:
        scalar = (2**o)
        cv2.circle(img=color,
                   center=(scalar * j, scalar * i),
                   radius=2,
                   color=(255, 0, 0),
                   thickness=-1)
        cv2.rectangle(img=color,
                      pt1=(scalar * (j - 8), scalar * (i - 8)),
                      pt2=(scalar * (j + 8), scalar * (i + 8)),
                      color=(255, 0, 0),
                      thickness=1)
    pyplot.imshow(color)
    pyplot.title("Keypoints of varying scales")
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("image",
                        help="Path to image we want to bag of words.",
                        type=Path)
    parser.add_argument("-k", "--plot-keypoints",
                        help="Whether to display image keypoints.",
                        action="store_true")
    parser.add_argument("-p", "--profile",
                        help="Capture profile information of the process.",
                        action="store_true")
    args = parser.parse_args()

    assert args.image.is_file()
    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2GRAY)

    main(image, args.profile, args.plot_keypoints)
