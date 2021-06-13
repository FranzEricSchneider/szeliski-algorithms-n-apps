import argparse
import cv2
from glob import glob
from os.path import basename, join

from matplotlib import pyplot
from matplotlib.patches import Ellipse
import numpy


# Always
numpy.set_printoptions(suppress=True)

PATH_MAPPING = {
    "people_1.jpeg": "PROCESSED_people_1.jpeg",
    "people_2.jpg": "PROCESSED_people_2.jpg",
    "people_3.jpeg": "PROCESSED_people_3.jpeg",
    "people_4.jpg": "PROCESSED_people_4.jpg",
    "people_5.jpg": "PROCESSED_people_5.jpg",
    "people_6.jpg": "PROCESSED_people_6.jpg",
    "people_7.jpg": "PROCESSED_people_7.jpg",
    "people_8.jpeg": "PROCESSED_people_8.jpeg",
    "people_9.jpg": "PROCESSED_people_9.jpg",
}
VISUALIZE_PATH = "ex_2_8_images/DETECTED"


def get_samples(search_path, use_chromaticity=True):
    image_paths = glob(search_path)
    images = [cv2.imread(path) for path in image_paths]
    if use_chromaticity:
        sample_images = [pixel_chromaticity(image) for image in images]
    else:
        sample_images = [color_ratio(image) for image in images]
    # TODO: Consider overweighting samples from small images
    samples, masks = flatten_nonzero(sample_images)
    return samples, image_paths, masks


def from_masks(paths, masks, invert=False):
    """Returns the raw sample values either in or out of the masks.

    Arguments:
        paths: list of paths to processed images, each of which should have a
            unprocessed counterpart
        masks: list of (n, m) boolean matrices indicating skin locations in the
            processed image. There should be a 1:1 relationship with paths
        invert: boolean. When false, return the samples within the masks. When
            true, return the samples outside the masks

    Returns: (q, 2) array of xy sampled pixel values
    """
    samples = None
    for path, mask in zip(paths, masks):
        raw_path = get_unprocessed(path)
        image = cv2.imread(raw_path)
        xy_image = pixel_chromaticity(image)
        if invert:
            mask = numpy.logical_not(mask)
        if samples is None:
            samples = xy_image[mask]
        else:
            samples = numpy.vstack((samples, xy_image[mask]))
    return samples


# Eq 2.104 on page 88
RGB_to_XYZ = numpy.array([
    [0.49,    0.3,    0.21   ],
    [0.17697, 0.8124, 0.01063],
    [0,       0.01,   0.99   ],
])
XYZ_to_RGB = numpy.linalg.inv(RGB_to_XYZ)

def pixel_chromaticity(image):
    """
    Arguments:
        images: (n, m, 3) numpy array, dtype uint8. Color image of the scene.

    Returns: (n, m, 2) numpy array, dtype uint8. (x, y) chromaticity values,
        which were first transferred to (X, Y, Z) space, then normalized to get
        chromaticity coordinates.
    """
    assert len(image.shape) == 3, "Needs width, height, and color channels"
    assert image.shape[-1] == 3, "Color channel wasn't 3D"
    assert image.dtype == numpy.uint8, "Given image wasn't uint8"

    # Get XYZ coordinates by applying RGB_to_XYZ matrix to each pixel
    # numpy.einsum is quite complicated and seems quite useful, as a reminder:
    # https://ajcr.net/Basic-guide-to-einsum/
    # https://stackoverflow.com/questions/25922212/element-wise-matrix-multiplication-in-numpy
    XYZ = numpy.einsum("ij,nmj->nmi", RGB_to_XYZ, image)
    # Normalize into xyz coordinates
    # TODO: Figure out how to remove the 0/0 warnings
    xyz = numpy.einsum("ijk,ij->ijk", XYZ, 1 / numpy.sum(XYZ, axis=2))
    # And remove the nans of pure black
    xyz[numpy.isnan(xyz)] = 0.0

    # It's not clear to me why chromaticity is xy instead of xyz, but hey
    return xyz[:, :, :2]


def color_ratio(image):
    # Normalize
    # TODO: Figure out how to remove the 0/0 warnings
    normalized = numpy.einsum("ijk,ij->ijk", image, 1 / numpy.sum(image, axis=2))
    # And remove the nans of pure black
    normalized[numpy.isnan(normalized)] = 0.0
    return normalized


def flatten_nonzero(images):
    """Select the non-zero pixels and flatten it all.

    The pixels that aren't of interest have been painted out using pure block,
    which is why this is done.

    Arguments:
        images: list of (n, m, 2 or 3) images in xy coordinates.

    Returns: two-element tuple of
        [0] (p, 2 or 3) vector flattened down from all of the images.
        [1] Lists of the masks the the samples were extracted with,
            corresponding to the incoming images
    """
    flattened = None
    masks = []
    for image in images:
        # Get the locations where either x or y pixels have any color value
        x0 = image[:, :, 0] != 0.0
        y0 = image[:, :, 1] != 0.0
        if image.shape[2] == 2:
            nonzeros = x0 | y0
        elif image.shape[2] == 3:
            z0 = image[:, :, 2] != 0.0
            nonzeros = x0 | y0 | z0
        else:
            raise ValueError(f"Image had shape {image.shape}, should be (n, m, 2/3)")

        # Then add the latest nonzero pixels to the growing list
        if flattened is None:
            flattened = image[nonzeros]
        else:
            flattened = numpy.vstack((flattened, image[nonzeros]))
        # Capture the masks
        masks.append(nonzeros.copy())
    return flattened, masks


def plot(samples, covariance=None, background=None, title=""):

    # Set up the shared axes
    left = bottom = 0.1
    width = height = 0.65
    spacing = 0.005
    figure = pyplot.figure(figsize=(8, 8))
    axis = figure.add_axes([left, bottom, width, height])
    axis_histx = figure.add_axes([left, bottom + height + spacing, width, 0.2],
                                 sharex=axis)
    axis_histy = figure.add_axes([left + width + spacing, bottom, 0.2, height],
                                 sharey=axis)
    axis_histx.tick_params(axis="x", labelbottom=False)
    axis_histy.tick_params(axis="y", labelleft=False)

    # Plot the distribution
    if background is not None:
        axis.scatter(background[:, 0], background[:, 1], s=0.2, color="k",
                     label="background")
    axis.scatter(samples[:, 0], samples[:, 1], s=0.3, label="samples")
    axis_histx.hist(samples[:, 0], bins=100)
    axis_histy.hist(samples[:, 1], bins=100, orientation="horizontal")

    # And the covariance on top of it
    if covariance is not None:
        means = numpy.mean(samples, axis=0)
        axis.scatter(means[0], means[1], s=10, color="r", label="mean")
        # Get the main axes
        values, vectors = numpy.linalg.eig(covariance)
        std_x, std_y = numpy.sqrt(values)
        angle = numpy.arctan2(vectors[1, 0], vectors[0, 0])
        axis.add_artist(Ellipse(means, values[0], values[1], angle, color="c",
                                fill=False, label="cov"))
        axis.add_artist(Ellipse(means, std_x, std_y, angle, color="r",
                                fill=False, label="sqrt(cov)"))

    # Add labels
    axis.legend()
    axis.set_xlabel("x value (normalized X)")
    axis.set_ylabel("y value (normalized Y)")
    axis_histx.set_title(title)
    pyplot.show()


def get_unprocessed(path):
    """Get the unprocessed version of the given processed path."""
    for raw_path in glob("ex_2_8_images/*jp*g"):
        if PATH_MAPPING.get(basename(raw_path), None) == basename(path):
            break
    else:
        raise RuntimeError(f"Couldn't find raw image for {path}")
    return raw_path


def detect_skin(image, mean, axes, stddev_values, scalar=1.0,
                use_chromaticity=True):
    """Checks whether each xy pixel is within the detected range for "skin".

    In practice this makes a rectangle around the mean, along the axes. Maybe
    it should be changed to an ellipse?

    Arguments:
        image: (n, m, 2) numpy array, dtype float. This should contain the xy
            units for the image
        mean: length-2 vector in the xy units, indicating the center of the
            rectangle
        axes: (2, 2) matrix, where each column is an eigenvector indicating a
            major axis of the detected covariance
        stddev_values: length of the rectangle sides, corresponding in order to
            the axes.
        scalar: float, default 1. This is multiplied by the stddev to either
            grow or shrink the rectangle from being one standard deviation.

    Returns: (n, m) boolean array, indicating whether that pixel is in the
        indicated rectangle.
    """

    # Get the full image in xy coordinates
    if use_chromaticity:
        processed_image = pixel_chromaticity(image)
    else:
        processed_image = color_ratio(image)
    # Center the values around the mean
    processed_image -= mean
    # Check the distance along the major and minor axes
    within_axes0 = numpy.abs(
        numpy.einsum("ijk,k->ij", processed_image, axes[:, 0])
    ) < scalar * stddev_values[0]
    within_axes1 = numpy.abs(
        numpy.einsum("ijk,k->ij", processed_image, axes[:, 1])
    ) < scalar * stddev_values[1]
    # "Skin" are those values within both axes directions
    return within_axes0 & within_axes1


def visualize_mask(path, mask, suffix):
    # To visualize the mask just write the True values to red
    image = cv2.imread(path)
    if mask is not None:
        image[mask] = [0, 0, 255]
    # Then save the file with an altered name
    filename = basename(path)
    filename = filename.split(".")
    filename = f"{filename[0]}_{suffix}.{filename[-1]}"
    cv2.imwrite(join(VISUALIZE_PATH, filename), image)


def main():
    parser = argparse.ArgumentParser(description='Bad skin detector.')
    parser.add_argument(
        '-c', '--color-ratio',
        action='store_true',
        help='Use color ratio instead of chromaticity'
    )
    args = parser.parse_args()

    samples, paths, masks = get_samples(
        "ex_2_8_images/PROCESSED/*jp*g",
        use_chromaticity=not args.color_ratio,
    )
    mean = numpy.mean(samples, axis=0)
    # Get covariance and the main axes
    covariance = numpy.cov(samples.T)
    cov_values, vectors = numpy.linalg.eig(covariance)
    # Hand-tuned scalar around the covariance (very unscientific)
    scalar = 2

    # THIS ENDED UP BEING TOO BROAD
    # Sqrt of variance is std deviation, so we can express ourselves in
    # multiples of stddev
    # stddev_values = numpy.sqrt(cov_values)

    # # Optional visualization
    # all_pixels, _, _ = get_samples("ex_2_8_images/*jp*g")
    # plot(samples, covariance, all_pixels,
    #      title="x,y values for skin pixel samples on background")

    # # Alternate visualization
    # plot(from_masks(paths, masks, invert=False),
    #      title="x,y values for skin pixel samples")
    # plot(from_masks(paths, masks, invert=True),
    #      title="x,y values for BACKGROUND samples")

    # Visualize images for which there is ground truth
    for path, mask in zip(paths, masks):
        raw_path = get_unprocessed(path)
        detected = detect_skin(
            cv2.imread(raw_path), mean, vectors, cov_values, scalar,
            use_chromaticity=not args.color_ratio
        )
        visualize_mask(raw_path, None, "ORIGINAL")
        visualize_mask(raw_path, detected, "DETECTED")
        visualize_mask(raw_path, mask, "GROUND_TRUTH")

    # And those without ground truth
    for raw_path in glob("ex_2_8_images/raw_only*jp*g"):
        detected = detect_skin(
            cv2.imread(raw_path), mean, vectors, cov_values, scalar,
            use_chromaticity=not args.color_ratio
        )
        visualize_mask(raw_path, None, "ORIGINAL")
        visualize_mask(raw_path, detected, "DETECTED")



if __name__ == "__main__":
    main()
