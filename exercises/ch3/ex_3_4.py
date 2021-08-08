import argparse
import cv2
from matplotlib import pyplot
import numpy


# Always
numpy.set_printoptions(suppress=True)


# TODO: Try this again with a single object?
def main(show_border, show_scale, show_delta, show_close, show_alpha):
    """
    Take a picture of the empty background, and then of the background with a
    new object in front of it. Pull the matte using the difference between each
    colored pixel and its assumed corresponding background pixel.

    Known difficulty - the phone appears to have color balanced between the two
    images, so we're going to have to do something to match up "background"
    between the two images.

    As a test, I want to display the matte on black and then on something else.
    """

    # Load the starter images
    still = cv2.cvtColor(cv2.imread("ex_3_4/still_life.jpg"),
                         cv2.COLOR_BGR2RGB)
    background = cv2.cvtColor(cv2.imread("ex_3_4/background.jpg"),
                              cv2.COLOR_BGR2RGB)

    # Get the border (known background) from the still image
    border = still.copy()
    height, width, _ = border.shape
    border_mask = numpy.ones((height, width), dtype=bool)
    border_mask[int(0.15 * height):int(0.9 * height),
                int(0.1 * width):int(0.95 * width)] = False
    border[numpy.logical_not(border_mask)] = 0
    if show_border:
        pyplot.imshow(border)
        pyplot.title("Shows the region we are designating as border")
        pyplot.show()

    # Clearly the relationship between still and background isn't a straight
    # color scale, but that may be good enough to work
    # R - mean: 0.9645, std: 1.4281
    # G - mean: 0.8418, std: 0.0795
    # B - mean: 0.7422, std: 0.0827
    printout, stats, recreated = scalar_relationship(
        background, still, border_mask
    )
    if show_scale:
        print(printout)
        pyplot.imshow(numpy.hstack((background, recreated, still)))
        pyplot.title("Original background | Color corrected background we will"
                     " compare to still | Original still")
        pyplot.show()

    # Get difference between still and recreated images, low deltas will
    # indicate background. It's important to cast one of the images as a float,
    # otherwise we will get wrap-around with negative numbers.
    delta = still.astype(float) - recreated
    distance = numpy.linalg.norm(delta, axis=2)
    if show_delta:
        greyscale = distance.copy()
        greyscale[greyscale > 255] = 255
        pyplot.imshow(greyscale.astype(numpy.uint8))
        pyplot.colorbar()
        pyplot.title("EUCLIDEAN DISTANCE delta between still, recreated images")
        pyplot.show()

        pyplot.imshow(numpy.hstack([
            numpy.abs(delta[:, :, i]).astype(numpy.uint8)
            for i in range(3)
        ]))
        pyplot.colorbar()
        pyplot.title("R, G, and B deltas, ABS()")
        pyplot.show()

    # Now make a simple mask! Hand-chosen threshold here
    mask = distance < 50
    # Agressively close small holes (value chosen via the show_close display)
    def close_image(image, size, n=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        closed = cv2.morphologyEx(image.astype(numpy.uint8), cv2.MORPH_CLOSE, kernel)
        for i in range(n - 1):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
        return closed
    closed = close_image(mask, 8)

    if show_close:
        pyplot.imshow(numpy.vstack((
            numpy.hstack((
                mask,
                close_image(mask, 3),
                close_image(mask, 5),
                close_image(mask, 8),
                close_image(mask, 10),
            )),
            numpy.hstack((
                close_image(mask, 12),
                close_image(mask, 14),
                close_image(mask, 16),
                close_image(mask, 20),
                close_image(mask, 25),
            )),
        )))
        pyplot.title("Closing with kernels of\noriginal, 3, 5, 8, 10\n12, 14,"
                     " 16, 20, 25")
        pyplot.show()

    # Now make an alpha channel based on that mask, smoothed along the edges.
    # Note that we are doing it on a 0-1 scale, not 0-255! This is because the
    # cv2.addWeighted function seems to expect this
    alpha = numpy.logical_not(closed).astype(float)
    if show_alpha:
        pyplot.imshow(numpy.hstack((
            alpha,
            cv2.GaussianBlur(src=alpha, ksize=(5, 5), sigmaX=0),
            cv2.GaussianBlur(src=alpha, ksize=(15, 15), sigmaX=0),
            cv2.GaussianBlur(src=alpha, ksize=(35, 35), sigmaX=0)
        )))
        pyplot.title("Smoothed alpha, kernel size: original, 5, 15, 35")
        pyplot.colorbar()
        pyplot.show()
    # Choose an aggressive blur because of the size of the image
    alpha = cv2.GaussianBlur(src=alpha, ksize=(15, 15), sigmaX=0)

    # And we're done! It's been matted! Make some display images
    # On black
    black = numpy.zeros(still.shape, dtype=numpy.uint8)
    pyplot.imshow(alpha_matte(still, black, alpha))
    pyplot.title("Matte on black")
    pyplot.show()
    # On a random image
    random = cv2.cvtColor(cv2.imread("ex_3_2/comparison.jpg"),
                          cv2.COLOR_BGR2RGB)
    pyplot.imshow(alpha_matte(still, random, alpha))
    pyplot.title("Matte on arbitrary image")
    pyplot.show()

    # Results: not amazing, but I purposefully included the shadowed regions
    # in the matting because they seemed clearly changed from the background.
    # The shadows don't look good on the composite images, but I think you'd
    # have to do something fancy to account for shadowing in particular.


def scalar_relationship(im1, im2, mask=None):
    assert im1.dtype == numpy.uint8
    assert im2.dtype == numpy.uint8

    # Avoid /0 errors
    im1 = im1.astype(float)
    im1[im1 == 0] = 1e-6

    # Make an all-good mask if one doesn't exist
    if mask is None:
        mask = numpy.ones(im1.shape[:2], dtype=bool)

    # Calculate scalar relationship and characterize, but only where the values
    # are high enough not to get blown all out of proportion
    min_level = 2
    scale = im2[mask] / im1[mask]
    stats = [
        (
            numpy.mean(scale[:, i][im1[mask][:, i] > min_level]),
            numpy.std(scale[:, i][im1[mask][:, i] > min_level]),
        )
        for i in range(3)
    ]

    # Save the characterization as a printable string
    printout = ""
    for color, (mean, std) in zip("RGB", stats):
        printout += f"{color} - mean: {mean:.4f}, std: {std:.4f}\n"

    # Recreate im2 as best as possible from im2, then cast to uint8 and clamp
    recreated = im1.copy()
    for i, (mean, _) in enumerate(stats):
        recreated[:, :, i] *= mean
    recreated = recreated.astype(numpy.uint8)
    recreated[recreated > 255] = 255

    return printout, stats, recreated


def alpha_matte(im1, im2, alpha):
    """Put im1 over im2 based on the given alpha channel."""
    # Multiply alpha by each color
    overlaid = numpy.zeros(im1.shape, dtype=im1.dtype)
    for i in range(3):
        overlaid[:, :, i] = alpha * im1[:, :, i] + (1 - alpha) * im2[:, :, i]
    # Clamp to [0-255] just in case, not sure this is necessary
    overlaid = numpy.clip(overlaid, 0, 255)
    return overlaid.astype(numpy.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--show-alpha",
        action="store_true",
        help="Display the alpha blurring process",
    )
    parser.add_argument(
        "-b", "--show-border",
        action="store_true",
        help="Display the border-only image",
    )
    parser.add_argument(
        "-c", "--show-close",
        action="store_true",
        help="Display a variety of close options on the mask",
    )
    parser.add_argument(
        "-d", "--show-delta",
        action="store_true",
        help="Display the deltas between the two images",
    )
    parser.add_argument(
        "-s", "--show-scale",
        action="store_true",
        help="Display the scale effects between the two images",
    )
    args = parser.parse_args()

    main(show_border=args.show_border,
         show_scale=args.show_scale,
         show_delta=args.show_delta,
         show_close=args.show_close,
         show_alpha=args.show_alpha)
