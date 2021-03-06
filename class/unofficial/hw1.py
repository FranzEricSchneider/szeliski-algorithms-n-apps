"""
Best guess: Implement the Hough Transform to detect lines in an image

Inspired by Ex 7.12: Hough transform line detector in Szeliski

Looked at the explanations here but didn't use the code:
    https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549

Also this:
    https://medium.com/@tomasz.kacmajor/hough-lines-transform-explained-645feda072ab

I think this is at a pretty good point. A couple of things that could be
improved:
1) The thresholding is nonsense and could use a legitimate mechanism
2) Some sort of notion of "side to side points are BAD, not good" would be
   extremely helpful in finding the thin lines that the human eye wants
3) Some sort of splitting-then-reweighting would help fight the issue of "find
   a mass of points across the whole image, call it a peak, then split it into
   multiple shitty lines". So much time is spent on the accumulator that I bet
   an extra processing loop would hardly be noticed. Would interact with the
   thresholding.
4) Do a better job of explaining and tuning the parameters would be nice
5) Maybe a way to loop on a specific image?
6) Doing smarter edge detection (e.g. based on the image) could be extremely
   helpful (garbage in garbage out)
"""


import argparse
from collections import namedtuple
import cProfile
from matplotlib import pyplot
from matplotlib.widgets import Slider, Button
from pathlib import Path
import time

import cv2
import numpy
from scipy.stats import linregress
import sys


Result = namedtuple("Result", ["gray", "edge", "lined"])
Line = namedtuple("Line", ["pt1", "pt2"])

# Set all figures to be large
pyplot.rcParams["figure.figsize"] = (20, 10)

# Size of kernel to blur with before calling canny
BLUR_KERNEL_SIZE = (3, 3)
# Canny parameter 1. Used to extend identified edges
CANNY_THRESH1 = 150
# Canny parameter 2. Used to initially find strong edges
CANNY_THRESH2 = 220
# Size of kernel used to close the initial edge-detected image
CLOSE_SIZE = (5, 5)

# Discretization of theta
THETA_N_STEPS = 40
# Discretization of distance (rho)
DIST_N_STEPS = 120

# Fraction of the image's pixels that it takes to turn it into a line from
# the accumulator
# TODO: Clean up entire thresholding process
LINE_THRESH_FRAC = 0.005

# Side-length of the square of spaces to ignore around each peak. Should be
# odd, so the peak can stay in the middle
WIPE_SIZE = 3
# Size of kernel used to close potential lines before checking connected
# components. If it's smaller or equal to CLOSE_SIZE then it's useless
LINE_CLOSE_SIZE = (7, 7)
# Fraction of the corner-to-corner size of the image that a post connected
# components line must be to keep
MIN_LINE_FRAC = 0.1


def main(image_dir, profile):

    # Save these algorithm parameters so that they can be easy tweaked in one
    # unified place
    parameters = {
        "blurnel": BLUR_KERNEL_SIZE,
        "canny_thresh1": CANNY_THRESH1,
        "canny_thresh2": CANNY_THRESH2,
        "close_size": CLOSE_SIZE,
        "theta_n": THETA_N_STEPS,
        "dist_n": DIST_N_STEPS,
        "line_thresh_frac": LINE_THRESH_FRAC,
        "wipe_size": WIPE_SIZE,
        "line_close_size": LINE_CLOSE_SIZE,
        "min_line_frac": MIN_LINE_FRAC,
        "plot_histogram": False,
        "plot_connected": False,
        "plot_px_in_line": False,
    }

    while True:

        # Start profiling if the flag is set
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        results = []
        for image_path in sorted(image_dir.glob("*.jp*g")):
            print(f"Processing {image_path}...")

            # Read image and find the edges
            image = cv2.cvtColor(cv2.imread(str(image_path)),
                                 cv2.COLOR_BGR2GRAY)
            # According to this it"s standard to blur before finding edges:
            # https://datacarpentry.org/image-processing/06-blurring/
            blurred = cv2.GaussianBlur(image, parameters["blurnel"], sigmaX=0)
            edge = cv2.Canny(blurred,
                             parameters["canny_thresh1"],
                             parameters["canny_thresh2"])

            # Try to clean up the edge image and make the image a bit simpler
            edge = close(edge, parameters["close_size"])
            # Then pull out the edge pixels that have been identified
            edgels = numpy.argwhere(edge)

            # Try and find lines, then display them on a rendered image. The
            # original image is just passed through for optional visualizations
            lines = hough_linefinder(image, edgels, edge.shape, parameters)
            lined = render_lines(image, lines)

            results.append(Result(gray=image, edge=edge, lined=lined))
            print(f"Found {len(lines)} lines")

        # Write out profile messages for speed-up attempts if the flag is set
        if profile:
            profiler.disable()
            filename = f"profile_{int(time.time()*1e6)}.snakeviz"
            profiler.dump_stats(filename)
            print(f"Wrote profile stats to {filename}")

        # Display the image and ask whether the user wants to try again with
        # any altered parameters
        parameters = display(results, parameters)


def close(image, size):
    """Helper function to close holes (erode then dilate)."""
    image = cv2.dilate(image,
                       cv2.getStructuringElement(cv2.MORPH_RECT, size))
    image = cv2.erode(image,
                      cv2.getStructuringElement(cv2.MORPH_RECT, size))
    return image


def hough_linefinder(image, edgels, image_shape, parameters):

    # Create the subdivision of the angle values. Pre-calculate the cos and
    # sine values to save replication later.
    thetas = numpy.array([(theta, numpy.cos(theta), numpy.sin(theta))
                          for theta in numpy.linspace(-numpy.pi/2,
                                                      numpy.pi/2,
                                                      parameters["theta_n"])])
    # Don't create subdivided distance values, instead create the values
    # necessary to take a calculated distance and figure out how far along the
    # subdivided vector it should land.
    max_dist = int(numpy.linalg.norm(image_shape))
    dist_slope = (parameters["dist_n"] - 1) / (2 * max_dist)

    # Fill the discfretized accumulator based on angle/distance representations
    # for all of the edge pixels.
    accumulator = sort_into_bins(
        edgels, thetas, dist_slope, max_dist, parameters
    )

    # Take the accumulator, where specific pixels are saved, and make a more
    # succint representation where we just count occurences. They'll both be
    # useful.
    num_accumulated = numpy.zeros(accumulator.shape, dtype=int)
    for i in numpy.ndindex(accumulator.shape):
        num_accumulated[i] = len(accumulator[i])

    # Plots the (theta, distance) accumulation distribution
    if parameters["plot_histogram"]:
        scalar = 255.0 / numpy.max(num_accumulated)
        pyplot.imshow((num_accumulated * scalar).astype(numpy.uint8))
        pyplot.xlabel("Theta (-90 to 90 deg)")
        pyplot.ylabel(f"Distance / rho (0 to {max_dist} pixels)")
        pyplot.title("Theta vs. Distance distribution")
        pyplot.show()

    # TODO: This is trash. Come up with another mechanism. Maybe step through
    # things in a sorted manner and don't use roi at all?
    threshold = numpy.max([numpy.min([
        0.9 * numpy.max(num_accumulated),
        numpy.product(image_shape) * parameters["line_thresh_frac"],
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
    for peak in peaks:

        # Grab the pixels that were associated with this peak
        edgels = numpy.array(accumulator[tuple(peak)])

        # IMPORTANT: Theta was initially defined from the X axis (axis 0)
        # relative to the PERPENDICULAR line, not to the actual line across the
        # image. In order to start reasoning about the line across the image we
        # need to do a little 90 degree rotation from theta.
        cos, sin = thetas[peak[0]][1:]
        direction = numpy.array([sin, -cos])

        # Create a mask of the identified line (peak in the histogram) and
        # close holes to a parameter specified degree. Then see which
        # components are connected, and only save/use the large ones.
        line_mask = numpy.zeros(image_shape, dtype=numpy.uint8)
        line_mask[edgels[:, 0], edgels[:, 1]] = 255
        line_mask = close(line_mask, parameters["line_close_size"])
        (num_components,
         labels,
         stats,
         centroids) = cv2.connectedComponentsWithStats(line_mask,
                                                       connectivity=4,
                                                       ltype=cv2.CV_32S)

        # Visualize the connected component process
        if parameters["plot_connected"]:
            plot_image = line_mask.copy()
            # Columns go (left, top, width, height, area). Skip the first label
            # because it is background
            for left, top, width, height, _ in stats[1:]:
                cv2.rectangle(plot_image,
                              pt1=(left, top),
                              pt2=(left + width, top + height),
                              color=175,
                              thickness=1)
            pyplot.imshow(plot_image)
            pyplot.title("Connected components for this peak/line candidate")
            pyplot.show()

        # Go through connected components and only keep components that are
        # long enough
        edgel_groups = []
        min_line_length = max_dist * parameters["min_line_frac"]
        for i, stat in enumerate(stats[1:]):
            # Columns go (left, top, width, height, area), so norm([2:4]) is
            # the corner to corner distance of (width, height)
            if numpy.linalg.norm(stat[2:4]) > min_line_length:
                edgel_groups.append(numpy.argwhere(labels == i+1))

        # Make each connected group into a separate line
        for group in edgel_groups:
            # Find the center of the group
            center = numpy.mean(group, axis=0)
            # Then find the furthest points in both directions *along* the
            # identified direction
            ordered = numpy.argsort(direction.dot(group.T))
            # Use the center and the extents to construct a line that reaches
            # to the furthest extents while still staying along the identified
            # angle and not getting pulled sideways by area points
            points = [
                numpy.round(
                    center + \
                    direction * direction.dot(group[ordered[i]] - center)
                ).astype(int)
                for i in [0, -1]
            ]
            line = Line(*points)
            lines.append(line)

            # Display which pixels ended up in this line
            if parameters["plot_px_in_line"]:
                display_px_forming_line(
                    image,
                    group,
                    line,
                    thetas[peak[0]][0],
                    j_to_dist(peak[1], dist_slope, max_dist),
                )

    return lines


def j_to_dist(j, dist_slope, max_dist):
    """Figure out what the approx distance value would be for this index."""
    return (j / dist_slope) - max_dist


def sort_into_bins(edgels, thetas, dist_slope, max_dist, parameters):
    """
    Fill the discfretized accumulator based on angle/distance representations
    for all of the edge pixels.
    """

    # Although this is a complicated line, it's important to keep as one line
    # because of how large the matrices are. We don't want to save intermediate
    # values.
    # First off, it's important to know that thetas[:, 1:] is an (n, 2) array
    # of (cos(theta), sin(theta)) values. distance = edgels.dot([cos, sin]) took
    # me a while to get, but it works if you define theta starting from +x to
    # the *perpendicular* line, then you see that
    # (y = -x cos / sin + dist / sin) works out exactly like mx + b, where a
    # 90deg triangle is formed with th y-axis, making b = dist / sin. Then you
    # rearrange that line formula to get dist.
    # Then slope and max_dist stuff is to figure out which discretized dist
    # value is most appropriate for this line.
    j_matrix = (
        dist_slope * (edgels.dot(thetas[:, 1:].T) + max_dist) + 0.5
    ).astype(int)

    accumulator = numpy.empty((parameters["theta_n"],
                               parameters["dist_n"]), dtype=object)
    for i in numpy.ndindex(accumulator.shape):
        accumulator[i] = []
    # TODO: This takes all the time in the process, basically. Maybe speed it
    # up later if possible? Perhaps by only saving a count and re-extracting
    # edgels as needed?
    for edgel, j_row in zip(edgels, j_matrix):
        for i, j in enumerate(j_row):
            accumulator[i, j].append(edgel)

    return accumulator


def display_px_forming_line(image, edgels, line, theta, dist):
    """Display the pixels that went into this line candidate."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Color each consituent pixel
    for edgel in edgels:
        cv2.circle(img=image,
                   center=tuple(reversed(edgel)),
                   radius=1,
                   color=(255, 0, 0),
                   thickness=-1)
    # Then create larger circles for the actually-chosen line endpoints
    for pt in [line.pt1, line.pt2]:
        cv2.circle(img=image,
                   center=tuple(reversed(pt)),
                   radius=numpy.max([int(0.01 * numpy.max(image.shape)), 2]),
                   color=(0, 0, 255),
                   thickness=-1)
    pyplot.imshow(image)
    pyplot.title(f"theta: {theta:.3f} rad, dist: {dist:.1f} px")
    pyplot.show()


def render_lines(image, lines):
    """Render the given lines onto an image."""
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for line in lines:
        # Remember, cv2.line defines (x, y) backwards from how I do
        image = cv2.line(
            image,
            pt1=tuple(reversed(line.pt1)),
            pt2=tuple(reversed(line.pt2)),
            color=(255, 0, 0),
            thickness=2,
        )
    return image


def display(results, parameters):
    """
    This will be a bit unusual, but display all processed images onto one big
    stiched together picture board. The goal is to be able to easily zoom in
    and see whatever, while having all output accessible at once.
    """

    figure, axis = pyplot.subplots()
    axis.margins(x=0)
    pyplot.subplots_adjust(left=0.02, right=1.0, top=1, bottom=0.3)

    # Stack each set of results vertically
    stacks = [
        numpy.vstack([result.lined,
                      cv2.cvtColor(result.edge, cv2.COLOR_GRAY2RGB)])
        for result in results
    ]

    # Use the largest image in each direction as a template and overlay each of
    # these stacks onto a black background
    max_0 = max([result.gray.shape[0] for result in results])
    max_1 = max([result.gray.shape[1] for result in results])
    for i in range(len(stacks)):
        stack = stacks[i]
        fullsize = numpy.zeros((2 * max_0, max_1, 3), dtype=numpy.uint8)
        fullsize[:stack.shape[0], :stack.shape[1]] = stack
        stacks[i] = fullsize

    pyplot.imshow(numpy.hstack(stacks))

    # Then go through a complicated series of definitions for sliders that can
    # Be used to adjust parameters. When the button is clicked we will update
    # the parameters according to slider values, close the window, and loop
    color = 'lightgoldenrodyellow'

    # TODO: Refactor
    x0 = 0.08
    x1 = 0.43
    x2 = 0.78
    width = 0.2
    height = 0.04
    blur = Slider(
        ax=pyplot.axes([x0, 0.2, width, height], facecolor=color),
        label="Blur Kernel (3-9)",
        valmin=3,
        valmax=9,
        valfmt="%1.0f",
        valinit=parameters["blurnel"][0],
        dragging=False,
        valstep=2,
    )
    canny1 = Slider(
        ax=pyplot.axes([x0, 0.15, width, height], facecolor=color),
        label="Canny1 (20-250)",
        valmin=20,
        valmax=250,
        valfmt="%1.0f",
        valinit=parameters["canny_thresh1"],
        dragging=False,
        valstep=10,
    )
    canny2 = Slider(
        ax=pyplot.axes([x0, 0.1, width, height], facecolor=color),
        label="Canny2 (20-250)",
        valmin=20,
        valmax=250,
        valfmt="%1.0f",
        valinit=parameters["canny_thresh2"],
        dragging=False,
        valstep=10,
    )
    n_theta = Slider(
        ax=pyplot.axes([x0, 0.05, width, height], facecolor=color),
        label="Theta N (20-300)",
        valmin=20,
        valmax=300,
        valfmt="%1.0f",
        valinit=parameters["theta_n"],
        dragging=False,
        valstep=10,
    )
    n_dist = Slider(
        ax=pyplot.axes([x0, 0, width, height], facecolor=color),
        label="Dist N (80-500)",
        valmin=80,
        valmax=500,
        valfmt="%1.0f",
        valinit=parameters["dist_n"],
        dragging=False,
        valstep=10,
    )
    close_kern = Slider(
        ax=pyplot.axes([x1, 0.2, width, height], facecolor=color),
        label="Close Kernel (3-9)",
        valmin=3,
        valmax=9,
        valfmt="%1.0f",
        valinit=parameters["close_size"][0],
        dragging=False,
        valstep=2,
    )
    thresh = Slider(
        ax=pyplot.axes([x1, 0.15, width, height], facecolor=color),
        label="~Line Thresh~ (0-0.05)",
        valmin=0,
        valmax=0.05,
        valfmt="%1.3f",
        valinit=parameters["line_thresh_frac"],
        dragging=True,
    )
    wipe_kern = Slider(
        ax=pyplot.axes([x1, 0.1, width, height], facecolor=color),
        label="Wipe Area (3-11)",
        valmin=3,
        valmax=11,
        valfmt="%1.0f",
        valinit=parameters["wipe_size"],
        dragging=False,
        valstep=2,
    )
    line_close_kern = Slider(
        ax=pyplot.axes([x1, 0.05, width, height], facecolor=color),
        label="Line Close (3-11)",
        valmin=3,
        valmax=11,
        valfmt="%1.0f",
        valinit=parameters["line_close_size"][0],
        dragging=False,
        valstep=2,
    )
    min_frac = Slider(
        ax=pyplot.axes([x1, 0, width, height], facecolor=color),
        label="Min Frac (0-0.5)",
        valmin=0,
        valmax=0.5,
        valfmt="%1.3f",
        valinit=parameters["min_line_frac"],
    )
    plot_hist = Slider(
        ax=pyplot.axes([x2, 0.2, width, height], facecolor=color),
        label="Plot Hist (n/y)",
        valmin=0,
        valmax=1,
        valfmt="%1.0f",
        valinit=parameters["plot_histogram"],
        valstep=1,
    )
    plot_conn = Slider(
        ax=pyplot.axes([x2, 0.15, width, height], facecolor=color),
        label="Plot Connected (n/y)",
        valmin=0,
        valmax=1,
        valfmt="%1.0f",
        valinit=parameters["plot_connected"],
        valstep=1,
    )
    plot_line = Slider(
        ax=pyplot.axes([x2, 0.1, width, height], facecolor=color),
        label="Plot Line (n/y)",
        valmin=0,
        valmax=1,
        valfmt="%1.0f",
        valinit=parameters["plot_px_in_line"],
        valstep=1,
    )

    def recreate(_):
        parameters["blurnel"] = (int(blur.val), ) * 2
        parameters["canny_thresh1"] = int(canny1.val)
        parameters["canny_thresh2"] = int(canny2.val)
        parameters["close_size"] = (int(close_kern.val), ) * 2
        parameters["theta_n"] = int(n_theta.val)
        parameters["dist_n"] = int(n_dist.val)
        parameters["line_thresh_frac"] = thresh.val
        parameters["wipe_size"] = int(wipe_kern.val)
        parameters["line_close_size"] = (int(line_close_kern.val), ) * 2
        parameters["min_line_frac"] = min_frac.val
        parameters["plot_histogram"] = bool(plot_hist.val)
        parameters["plot_connected"] = bool(plot_conn.val)
        parameters["plot_px_in_line"] = bool(plot_line.val)
        pyplot.close()
        print("=" * 80)
    loop_button = Button(
        pyplot.axes([0.7, 0.025, 0.08, 0.04]), "Loop", color=color
    )
    loop_button.on_clicked(recreate)

    def stop(_):
        pyplot.close()
        sys.exit(0)
    stop_button = Button(
        pyplot.axes([0.8, 0.025, 0.08, 0.04]), "Stop", color="r"
    )
    stop_button.on_clicked(stop)

    pyplot.show()
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get foreground from video.")
    parser.add_argument("directory",
                        help="Path to directory with imX.jpg images.",
                        type=Path)
    parser.add_argument("-p", "--profile",
                        help="Capture profile information of the process.",
                        action="store_true")
    args = parser.parse_args()
    main(image_dir=args.directory, profile=args.profile)
