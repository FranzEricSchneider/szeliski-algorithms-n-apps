"""
Best guess: Implement the Hough Transform to detect lines in an image

Inspired by Ex 7.12: Hough transform line detector in Szeliski

Looked at the explanations here but didn't use the code:
    https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549

Also this:
    https://medium.com/@tomasz.kacmajor/hough-lines-transform-explained-645feda072ab
"""


# OVERALL TODOS
# 1) Get this running a lot faster
# 2) Clean the heck up
# 3) Make a mechanism to loop with different parameters


import argparse
from collections import namedtuple
import cProfile
from matplotlib import pyplot
from pathlib import Path
import time

import cv2
import numpy
from scipy.stats import linregress


Result = namedtuple("Result", ["gray", "edge", "lined"])


BLUR_KERNEL_SIZE = (3, 3)
CANNY_THRESH1 = 150
CANNY_THRESH2 = 220
CLOSE_SIZE = (5, 5)

THETA_N_STEPS = 50
DIST_N_STEPS = 150

# Fraction of the image's pixels that it takes to turn it into a line from
# the accumulator
LINE_THRESH_FRAC = 0.005

ENDPOINT_FRAC = 0.02

WIPE_SIZE = 3
LINE_CLOSE_SIZE = (7, 7)
MIN_LINE_FRAC = 0.1


def main(image_dir, profile):

    # TODO
    results = []
    parameters = {
        "blurnel": BLUR_KERNEL_SIZE,
        "canny_thresh1": CANNY_THRESH1,
        "canny_thresh2": CANNY_THRESH2,
        "close_size": CLOSE_SIZE,
        "theta_n": THETA_N_STEPS,
        "dist_n": DIST_N_STEPS,
        "line_thresh_frac": LINE_THRESH_FRAC,
        "endpoint_frac": ENDPOINT_FRAC,
        "wipe_size": WIPE_SIZE,
        "line_close_size": LINE_CLOSE_SIZE,
        "min_line_frac": MIN_LINE_FRAC,
    }

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

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
        # Try to clean up the edge image as well and make it a little simpler
        edge = close(edge, parameters["close_size"])

        lines = hough_linefinder(blurred, edge, parameters)
        print(f"Found {len(lines)} lines")
        lined = render_lines(image, lines)

        results.append(Result(gray=blurred, edge=edge, lined=lined))

    if profile:
        profiler.disable()
        filename = f"profile_{int(time.time()*1e6)}.snakeviz"
        profiler.dump_stats(filename)
        print(f"Wrote profile stats to {filename}")

    display(results)


def close(image, size):
    """Helper function to close holes (erode then dilate)."""
    image = cv2.dilate(image,
                       cv2.getStructuringElement(cv2.MORPH_RECT, size))
    image = cv2.erode(image,
                      cv2.getStructuringElement(cv2.MORPH_RECT, size))
    return image


# TODO: Try to speed this up
def hough_linefinder(image, edge_image, parameters):

    accumulator = numpy.empty((parameters["theta_n"],
                               parameters["dist_n"]), dtype=object)
    for i in numpy.ndindex(accumulator.shape):
        accumulator[i] = []

    thetas = [
        (theta, numpy.cos(theta), numpy.sin(theta))
        for theta in numpy.linspace(-numpy.pi/2,
                                    numpy.pi/2,
                                    parameters["theta_n"])
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
    dist_slope = (parameters["dist_n"] - 1) / (2 * max_dist)

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
            j = int(dist_slope * (dist + max_dist) + 0.5)

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
    ]), sorted(num_accumulated.flatten())[-20]])
    roi = num_accumulated >= threshold

    # Go through and only take peaks in the center of a given area
    peaks = []
    # Approximation of a circular wipe region, this takes a grid size, e.g.
    # 3-pixel-on-a-side square, and finds the distance to the corner.
    radius = numpy.sqrt(2 * ((parameters["wipe_size"] - 1) / 2)**2) + 0.1
    for peak in numpy.argwhere(roi):
        if not any([numpy.linalg.norm(peak - existing) < radius
                    for existing in peaks]):
            peaks.append(peak)

    lines = []
    class Line:
        def __init__(self, pt1, pt2):
            self.pt1 = pt1
            self.pt2 = pt2

        def __repr__(self):
            return f"{self.pt1}, {self.pt2}"

    for peak in peaks:

        # TODO: Talk about 90 degree shift, and probably clean up
        cos, sin = thetas[peak[0]][1:]
        direction = numpy.array([sin, -cos])

        edgels = numpy.array(accumulator[tuple(peak)])

        TODO1 = numpy.zeros(image.shape, dtype=numpy.uint8)
        TODO1[edgels[:, 0], edgels[:, 1]] = 255
        TODO1 = close(TODO1, parameters["line_close_size"])
        (num_components,
         labels,
         stats,
         centroids) = cv2.connectedComponentsWithStats(TODO1,
                                                       connectivity=4,
                                                       ltype=cv2.CV_32S)

        # plot_image = TODO1.copy()
        # # Columns go (left, top, width, height, area). Skip the first label
        # # because it is background
        # for left, top, width, height, _ in stats[1:]:
        #     cv2.rectangle(plot_image,
        #                   pt1=(left, top),
        #                   pt2=(left + width, top + height),
        #                   color=175,
        #                   thickness=1)
        # pyplot.imshow(plot_image)
        # pyplot.show()

        edgel_groups = []
        min_line_length = max_dist * parameters["min_line_frac"]
        for label, stat in zip(range(1, len(stats)), stats[1:]):
            if numpy.linalg.norm(stat[2:4]) > min_line_length:
                edgel_groups.append(numpy.argwhere(labels == label))

        for group in edgel_groups:
            ordered = numpy.argsort(direction.dot(group.T))
            group_size = int(numpy.ceil(len(group) * parameters["endpoint_frac"]))
            end1 = numpy.mean(group[ordered[:group_size]], axis=0)
            end2 = numpy.mean(group[ordered[-group_size:]], axis=0)
            center = numpy.mean(group, axis=0)
            pt1 = numpy.round(center + direction * direction.dot(end1 - center)).astype(int)
            pt2 = numpy.round(center + direction * direction.dot(end2 - center)).astype(int)

            line = Line(pt1, pt2)

            # display_px_forming_line(image, group, line, thetas[peak[0]], dist_values[peak[1]])

            # # I had this baffling thing where you'd have this vertical string of
            # # points separated by a small x value, e.g. [(200, 100), (200, 150),
            # # (200, 200), (202, 100), (202, 150), (202, 200)] and the best line
            # # for it clearly would be a vertical line down the center, however, it
            # # was fit essentially horizontal (m=0.6, b=50). Trash. But somehow it
            # # works for horizontal collections?
            # line = linregress(group[:, 0], group[:, 1])
            # pyplot.plot(group[:, 0], group[:, 1], 'ro')
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
        cv2.circle(img=image,
                   center=tuple(reversed(edgel)),
                   radius=1,
                   color=(255, 0, 0),
                   thickness=-1)
    for pt in [line.pt1, line.pt2]:
        cv2.circle(img=image,
                   center=tuple(reversed(pt)),
                   radius=numpy.max([int(0.01 * numpy.max(image.shape)), 2]),
                   color=(0, 0, 255),
                   thickness=-1)
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
    figure1, axis1 = pyplot.subplots(1, 1)
    axis1.imshow(numpy.hstack(stacks))

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
    figure2, axis2 = pyplot.subplots(1, 1)
    axis2.imshow(numpy.hstack(stacks))

    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("image",
                        help="Path to folder with imX.jpg.",
                        type=Path)
    parser.add_argument("-p", "--profile",
                        help="Capture profile information of the process.",
                        action="store_true")
    args = parser.parse_args()
    main(image_dir=args.image, profile=args.profile)
