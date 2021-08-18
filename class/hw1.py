"""
Best guess: Implement the Hough Transform to detect lines in an image

Inspired by Ex 7.12: Hough transform line detector in Szeliski

Looked at the explanations here but didn't use the code:
    https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549

Also this:
    https://medium.com/@tomasz.kacmajor/hough-lines-transform-explained-645feda072ab
"""


# OVERALL TODOS
# 1) Figure out a mechanism for "connectedness" checks for messy backgrounds
# 2) Clean the heck up


import argparse
from collections import namedtuple
from matplotlib import pyplot
from pathlib import Path

import cv2
import numpy
from scipy.stats import linregress


Result = namedtuple("Result", ["gray", "edge", "lined"])


BLUR_KERNEL_SIZE = (3, 3)
CANNY_THRESH1 = 120
CANNY_THRESH2 = 180

THETA_N_STEPS = 50
DIST_N_STEPS = 100

# Fraction of the image's pixels that it takes to turn it into a line from
# the accumulator
LINE_THRESH_FRAC = 0.005

ENDPOINT_FRAC = 0.02


def main(image_dir):

    # TODO
    results = []
    parameters = {
        "blurnel": BLUR_KERNEL_SIZE,
        "canny_thresh1": CANNY_THRESH1,
        "canny_thresh2": CANNY_THRESH2,
        "theta_n": THETA_N_STEPS,
        "dist_n": DIST_N_STEPS,
        "line_thresh_frac": LINE_THRESH_FRAC,
        "endpoint_frac": ENDPOINT_FRAC,
    }

    for image_path in image_dir.glob("*.jp*g"):
        print(f"{image_path}...")
        # Read image and find the edges
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
        # According to this it"s standard to blur before finding edges:
        # https://datacarpentry.org/image-processing/06-blurring/
        blurred = cv2.GaussianBlur(image, parameters["blurnel"], sigmaX=0)
        edge = cv2.Canny(blurred,
                         parameters["canny_thresh1"],
                         parameters["canny_thresh2"])

        lines = hough_linefinder(blurred, edge, parameters)
        print(f"Found {len(lines)} lines")
        lined = render_lines(image, lines)

        results.append(Result(gray=blurred, edge=edge, lined=lined))

    display(results)


# TODO: Try to speed this up
def hough_linefinder(image, edge_image, parameters):

    accumulator = numpy.empty((parameters["theta_n"],
                               parameters["dist_n"]), dtype=object)
    for i in numpy.ndindex(accumulator.shape):
        accumulator[i] = []

    thetas = [
        (theta, numpy.cos(theta), numpy.sin(theta))
        for theta in numpy.linspace(
            -numpy.pi/2,
            numpy.pi/2,
            parameters["theta_n"]
        )
    ]
    thetas[parameters["theta_n"] // 2] = (0.0, 1.0, 0.0)

    # TODO
    max_dist = int(numpy.linalg.norm(edge_image.shape))
    offset = max_dist / parameters["dist_n"]
    dist_values = numpy.linspace(
        -max_dist + offset,
        max_dist - offset,
        parameters["dist_n"],
    )

    # TODO: Explain
    all_values = []
    for edgel in numpy.argwhere(edge_image):
        for i, (theta, cos, sin) in enumerate(thetas):
            # This took me a while to get, but it works if you define theta
            # starting from +x to the *perpendicular* line, then you see that
            # (y = -x cos / sin + dist / sin) works out exactly like mx + b,
            # where a 90deg triangle is formed with th y-axis, making
            # b = dist / sin. Then you rearrange that line formula to get dist.
            dist = edgel[0] * cos + edgel[1] * sin
            # Figure out which discretized dist value is closest to this edgel
            j = numpy.argmin(numpy.abs(dist_values - dist))
            accumulator[i, j].append(edgel)

            all_values.append([theta, dist])

            # display_pix_line(image, edgel, theta, dist)

    all_values = numpy.array(all_values)

    # TODO: Explain
    # TODO: Function?
    num_accumulated = numpy.zeros(accumulator.shape, dtype=int)
    for i in numpy.ndindex(accumulator.shape):
        num_accumulated[i] = len(accumulator[i])

    # pyplot.imshow((num_accumulated * 255.0 / numpy.max(num_accumulated)).astype(numpy.uint8))
    # pyplot.show()

    # pyplot.plot(all_values[:, 0], all_values[:, 1], 'o', markersize=2, alpha=0.05)
    # pyplot.xlabel("Theta (rad)")
    # pyplot.ylabel("Dist (px)")
    # pyplot.show()

    # TODO: This is trash
    threshold = numpy.max([numpy.min([
        0.9 * numpy.max(num_accumulated),
        numpy.product(edge_image.shape) * parameters["line_thresh_frac"],
    ]), sorted(num_accumulated.flatten())[-10]])
    roi = num_accumulated >= threshold

    lines = []
    class Line:
        def __init__(self, pt1, pt2):
            self.pt1 = pt1
            self.pt2 = pt2

        def __repr__(self):
            return f"{self.pt1}, {self.pt2}"

    for line_TODO in numpy.argwhere(roi):

        # TODO: Talk about 90 degree shift, and probably clean up
        cos, sin = thetas[line_TODO[0]][1:]
        direction = numpy.array([sin, -cos])

        edgels = numpy.array(accumulator[tuple(line_TODO)])
        ordered = numpy.argsort(direction.dot(edgels.T))
        group_size = int(numpy.ceil(len(edgels) * parameters["endpoint_frac"]))
        end1 = numpy.mean(edgels[ordered[:group_size]], axis=0)
        end2 = numpy.mean(edgels[ordered[-group_size:]], axis=0)
        center = numpy.mean(edgels, axis=0)
        pt1 = numpy.round(center + direction * direction.dot(end1 - center)).astype(int)
        pt2 = numpy.round(center + direction * direction.dot(end2 - center)).astype(int)

        line = Line(pt1, pt2)

        # display_px_forming_line(image, edgels, line, thetas[line_TODO[0]], dist_values[line_TODO[1]])

        # # I had this baffling thing where you'd have this vertical string of
        # # points separated by a small x value, e.g. [(200, 100), (200, 150),
        # # (200, 200), (202, 100), (202, 150), (202, 200)] and the best line
        # # for it clearly would be a vertical line down the center, however, it
        # # was fit essentially horizontal (m=0.6, b=50). Trash. But somehow it
        # # works for horizontal collections?
        # line = linregress(edgels[:, 0], edgels[:, 1])
        # pyplot.plot(edgels[:, 0], edgels[:, 1], 'ro')
        # pyplot.title(f"{line}")
        # pyplot.show()

        lines.append(line)

    return lines


def display_pix_line(image, edgel, theta, dist):
    # Just give up on these for now
    if abs(numpy.sin(theta)) < 1e-9:
        return
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.circle(img=image, center=tuple(reversed(edgel)), radius=3, color=(255, 0, 0), thickness=-1)
    m = -numpy.cos(theta) / numpy.sin(theta)
    b = dist / numpy.sin(theta)
    # Remember, cv2.line defines (x, y) backwards from how I do
    image = cv2.line(
        image,
        pt1=(int(b), 0),
        pt2=(int(m * image.shape[0] + b), image.shape[0]),
        color=(0, 255, 0),
        thickness=1,
    )
    pyplot.imshow(image)
    pyplot.title(f"pixel: {edgel}, theta: {theta} rad, dist: {dist} px")
    pyplot.show()


def display_px_forming_line(image, edgels, line, theta, dist):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for edgel in edgels:
        cv2.circle(img=image, center=tuple(reversed(edgel)), radius=1, color=(255, 0, 0), thickness=-1)
    for pt in [line.pt1, line.pt2]:
        cv2.circle(img=image, center=tuple(reversed(pt)), radius=2, color=(0, 0, 255), thickness=-1)
    pyplot.imshow(image)
    print(f"theta: {theta} rad, dist: {dist} px")
    pyplot.title(f"theta: {theta} rad, dist: {dist} px")
    pyplot.show()


def render_lines(image, lines):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for line in lines:
        # Remember, cv2.line defines (x, y) backwards from how I do
        image = cv2.line(
            image,
            pt1=tuple(reversed(line.pt1)),
            pt2=tuple(reversed(line.pt2)),
            color=(255, 0, 0),
            thickness=1,
        )
        # image = cv2.line(
        #     image,
        #     pt1=(int(line.intercept), 0),
        #     pt2=(int(image.shape[0] * line.slope + line.intercept), image.shape[0]),
        #     color=(255, 0, 0),
        #     thickness=2,
        # )
    return image


def display(results):
    # TODO: Resolve with color
    max_0 = max([result.gray.shape[0] for result in results])
    max_1 = max([result.gray.shape[1] for result in results])
    stacks = [
        numpy.vstack([result.gray, result.edge])
        for result in results
    ]
    for i in range(len(stacks)):
        stack = stacks[i]
        fullsize = numpy.zeros((2 * max_0, max_1), dtype=numpy.uint8)
        fullsize[:stack.shape[0], :stack.shape[1]] = stack
        stacks[i] = fullsize
    pyplot.imshow(numpy.hstack(stacks))
    pyplot.show()

    # TODO: resolve with black and white
    stacks = [
        numpy.vstack([result.lined])
        for result in results
    ]
    for i in range(len(stacks)):
        stack = stacks[i]
        fullsize = numpy.zeros((max_0, max_1, 3), dtype=numpy.uint8)
        fullsize[:stack.shape[0], :stack.shape[1]] = stack
        stacks[i] = fullsize
    pyplot.imshow(numpy.hstack(stacks))
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("image",
                        help="Path to the image as want to detect lines in.",
                        type=Path)
    args = parser.parse_args()
    main(image_dir=args.image)
