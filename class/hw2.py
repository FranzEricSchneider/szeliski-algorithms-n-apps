"""
Best guess: Extract SIFT features to build a Bag-of-Words representation of an
image for classification
"""

import argparse
import cProfile
from pathlib import Path
import time

import cv2
import numpy


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



def main(image, profile):

    # Start profiling if the flag is set
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # There are many possible keypoint detectors. Let's use the one described
    # in the original SIFT paper, Difference of Gaussians (DoG). Apparently
    # this is a computationally efficient version of Laplacian of Gaussians.
    keypoints = detect_features(image)

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
    """
    downsampled = downsample(image, NUM_OCTAVES)
    blurred = blur(downsampled, KERNELS)
    differences = difference(blurred)


def downsample(image, num_octaves):
    """Downsample an image recursively until we have the right number.

    Arguments:
        image: Regular grayscale image, np.uint8 (n, m) matrix
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
        images: A list of grayscale images
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


def difference(blurred):
    """Get differences between each image in the octaves.

    Arguments:
        blurred: See blur() docstring for output details

    Returns: almost the same as before, but with length 1 less within each
        octave. That's for the simple reason that getting deltas between
        elements in a list reduces the length by 1.
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("image",
                        help="Path to image we want to bag of words.",
                        type=Path)
    parser.add_argument("-p", "--profile",
                        help="Capture profile information of the process.",
                        action="store_true")
    args = parser.parse_args()
    import ipdb; ipdb.set_trace()
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2GRAY)
    main(image, args.profile)
