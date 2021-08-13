import argparse
from collections import deque
from pathlib import Path

import cv2
from matplotlib import pyplot
import numpy


"""
Working from section 2.1 here:
http://users.eecs.northwestern.edu/~yingwu/teaching/EECS432/Reading/Toyama_ICCV99.pdf

Which then directs you here:
https://www.commsp.ee.ic.ac.uk/~xl404/papers/Linear%20prediction%20A%20tutorial%20review.pdf
    Particularly II)C) on page 5, where it references eq. (18) on page 4
"""


# Choose some arbitrary height and width limits, video will be scaled to
# whichever is more extreme. These need to be floats for division's sake.
HEIGHT = 70.0
WIDTH = 100.0


# Choose these Wiener filter variables, somewhat arbitrarily
P = 10
N = 20


# TODO: Can we make these not greyscale in the future? I think the original
# paper was greyscale...
def pre_process(video_path):
    """Turn given video into a series of shrunken greyscale images."""

    # Starting assertion
    assert video_path.suffix == ".mp4"

    # Create a place for our processed images to go, based on the given path.
    # Given "example.mp4", the stem will just be "example". Assert implicitly
    # that this directory cannot already exist.
    new_dir = Path(video_path.parent, video_path.stem)
    new_dir.mkdir()

    # Look at the first image and see what the scaling factor should be
    capture = cv2.VideoCapture(str(video_path))
    success, image = capture.read()
    if not success:
        raise RuntimeError(f"Couldn't even get one frame from {video_path}")
    scale = max(image.shape[0] / HEIGHT, image.shape[1] / WIDTH)
    desired_size = (int(image.shape[1] / scale), int(image.shape[0] / scale))

    # Go through and write each frame as a PNG
    counter = 0
    while success:
        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(Path(new_dir, f"frame_{counter}.png")),
                    cv2.resize(greyscale, dsize=desired_size))
        counter += 1
        success, image = capture.read()

    return new_dir, counter, desired_size[::-1]


def frame_number(image_path):
    """Helper function to get frame number as an int. Useful for sorting."""
    return int(str(image_path).replace(".png", "").split("_")[-1])


def process(image_paths):

    # Set the image history size, see phi() docstring for why we need P+N
    history = deque(maxlen=P+N)
    coefficients = None

    for i, path in enumerate(image_paths):
        # Read each image. If we don't have enough history, store and move on
        image = cv2.imread(str(path), flags=cv2.IMREAD_GRAYSCALE).astype(int)
        if len(history) < history.maxlen:
            history.appendleft(image)
            continue

        # Once we have enough history, estimate the pixel-by-pixel coefficients
        if coefficients is None:
            import cProfile
            prof = cProfile.Profile()
            prof.enable()
            coefficients = instantiate_coefficients(history, p=P, N=N)
            prof.disable()
            prof.dump_stats("stats_no_loop.profile")

        # Then use that history (without image in it) to make a prediction
        # we can compare to image
        predicted = predixel(history, coefficients)

        # TODO
        expected_error = predixel_error(image, history, coefficients)

        foreground = numpy.abs(image - predicted) > (0.5 * expected_error)
        print(numpy.sum(foreground))

        display(image, foreground, i)

        # coefficients = update_coefficients(coefficients, history)


# TODO: This has GOT to go faster
def instantiate_coefficients(history, p, N):
    """Calculate historical coefficients for each pixel."""

    # Instantiate the matrix to get the shape right (note that it's (n, m, p))
    # instead of (n, m, 3) or something for a color image
    coefficients = numpy.zeros((history[0].shape[0], history[0].shape[1], p),
                               dtype=float)
    histarray = numpy.array(history)

    # Go through each pixel and set the coefficients
    for i in range(history[0].shape[0]):
        for j in range(history[0].shape[1]):
            matrix, vector = construct_equations(s=histarray[:, i, j],
                                                 p=p,
                                                 N=N)
            try:
                coefficients[i, j] = numpy.linalg.solve(matrix, vector)
            except numpy.linalg.LinAlgError:
                # Sometimes the matrix will be singular, this is apparently
                # the next best solution. This seems to be less than 5%
                coefficients[i, j], _, _, _ = numpy.linalg.lstsq(matrix,
                                                                 vector,
                                                                 rcond=None)
        print(i, f"0:{history[0].shape[1]}")

    return coefficients


def predixel(history, coefficients):
    """Predict the upcoming pixel values.

    According to the Toyama paper, the predicted value sp_t is equal to
        -1 * sum {k from 1 to p} (a_k * s_{t-k})
    Where a_k is the coefficient and s_{t-k} is the historical value.

    Arguments:
        history: length p+N list of greyscale images, where the most recent
            images MUST COME FIRST
        coefficients: matrix of shape (n, m, p), where the greyscale images are
            of shape (n, m)
    """

    # We know this must be the case
    p = coefficients.shape[2]
    # Get inputs as numpy arrays, only back as far as p
    history = numpy.array([history[i] for i in range(p)])
    # Map the historical axis (k in kij) to the coefficient axis (k in ijk)
    return -numpy.einsum("kij,ijk->ij", history, coefficients)


# TODO: CLEAN UP, TEST
def predixel_error(image, history, coefficients):

    p = coefficients.shape[2]

    st_sq = image**2

    ak_st = coefficients * numpy.repeat(
        image.reshape(image.shape[0], image.shape[1], 1),
        repeats=p,
        axis=2,
    )

    history = numpy.array([history[i] for i in range(p)])

    # TODO: Why is abs necessary?
    return numpy.sqrt(numpy.abs(st_sq + numpy.einsum("kij,ijk->ij", history, ak_st)))


def display(image, mask, i):
    desired_size = (int(image.shape[1] * 5), int(image.shape[0] * 5))
    color = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)
    if i < 100:
        return
    # import ipdb; ipdb.set_trace()
    color[mask] += numpy.array([0, 0, 80], dtype=numpy.uint8)
    # color[color > 255] = 255
    color = cv2.resize(color.astype(numpy.uint8), dsize=desired_size)
    cv2.imshow(f"YOLO_{i}", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def construct_equations(s, p, N):
    """As shown in eq. (18) we need to make an aX=b matrix and results vector.

    This is the prelude to computing the p different coefficients by solving
    for p separate linear equations. As eq. (18) states, we want to calculate
    each row like so:
        sum {k from 1 to p} a_k * phi(k, i) = -phi(0, i)
    And then i is defined from 1 to p as well, each i forming a new row. It's
    pretty clear that we can separate this out into a matrix X of all the LHS
    phi() components, an "a" vector of a_1 to a_p, then a "b" vector on the RHS
    that represents the result in this solution.

    Arguments:
        s: A list of historical "output signal" values, which I am interpreting
            as the actually-seen pixel values. Which makes this a time series
            of uint8.
        p: (int) Number of steps that we need to know "in the future", see
            docstring equation discussion of phi() function
        N: (int) Number of steps to look back in the past, needed by phi()
    """

    # As discussed in the phi() docstring, this is a requirement
    assert len(s) >= p + N

    # TODO: Make assertions about dtype=int

    matrix = []
    vector = []

    # Form the rows
    for i in range(1, p + 1):
        row = []
        # And the columns within it. Note the switching of i and k, that was
        # on purpose and matches the notation in the paper. Not strictly
        # necessary but it's nice to keep everything consistent.
        for k in range(1, p + 1):
            row.append(phi(s=s, p=p, N=N, i=k, k=i))
        matrix.append(row)
        vector.append(-1 * phi(s=s, p=p, N=N, i=0, k=i))

    # And put these in numpy
    return (numpy.array(matrix), numpy.array(vector))


def phi(s, p, N, i, k):
    """Calculates phi_{i,k} as from eq. (20) in the linear prediction primer.

    From eq. (20) we know that
        phi_{i,k} = sum {n from 0 to N-1} ( s[n-i] - s[n-k] )

    It's important to note that i and k can go from 0 to a value called p. So
    in the case where (n=0, i=p), then we have the case s[-p]. This does NOT
    mean the python thing where we wrap around to the beginning, instead the
    text says "the values from -p to N-1 must all be known: a total of p+N
    samples". To account for that, I'm going to take p as an input and then
    treat the [p] index as 0, like so:
        p = 5
        -5, -4, -3, -2, -1, 0  # We know s must be defined from -p to N-1
         0,  1,  2,  3,  4, 5  # So when we zero-index the array [p] hits our
                               # desired 0 point

    It's important to note that the choices here have been made such that
    new values should show up in the FRONT of the array, which is a little
    unusual but I think matches well with the paper's layout.

    Arguments:
        s: A numpy array of historical "output signal" values, which I am
            interpreting as the actually-seen pixel values. Which makes this a
            time series of uint8.
        p: (int) Number of steps that we need to know "in the future", see
            docstring equation discussion
        N: (int) Number of steps to look back in the past
        i: (int) Index offset for the first term
        k: (int) Index offset for the second term

    Returns: (int but NOT uint8) sum of a variety of products, as defined by
        the equation.
    """
    # Perform the sum with the offset to make p the "0" point in the array as
    # discussed.
    return numpy.sum(s[p-i:p+N-i] * s[p-k:p+N-k])


# Do some fiddling around by hand
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pre-process",
        help="If a path to a video is given here, DO NOT do any real"
             " processing, just turn the video into a directory of images of"
             " the right size constraints.",
        type=Path,
    )
    group.add_argument(
        "--process",
        help="If a path to a directory of correctly sized images is given,"
             " process those images to determine fore/background.",
        type=Path,
    )
    args = parser.parse_args()

    if args.pre_process:
        out_path, number, size = pre_process(video_path=args.pre_process)
        print(f"\n{number} images of size {size} have been written to"
               " {out_path}. Run again with")
        print(f"\tpython ex_3_5.py --process {out_path}")
        print("to process them.")
    else:
        process(sorted(args.process.glob("*.png"),
                       key=lambda x: frame_number(x)))
