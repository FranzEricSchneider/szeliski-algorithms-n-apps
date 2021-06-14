from matplotlib import pyplot
import numpy


# Example focal lengths, in mm
F = [60, 80, 100, 120, 140]
# TODO
OFFSET = 5


# Compute and plot the focus distance z_o as a function of the distance
# traveled from the focal length ∆z_i = f − z_i for a lens of focal length f
# (say, 100mm)
def part_1():
    """The governing equation is
        1/z_o + 1/z_i = 1/f
    where z_o is the distance from lens to object, z_i is the distance back
    from the lens where the image is actually cast, and f is the camera focal
    length. Note that f is not quite the same as z_i (pg 74). The governing
    equation that I'll use is
        zo = 1 / (1/f - 1/zi)
    """

    # Plot lines for a variety of focal lengths
    for f in F:
        zi = numpy.arange(f - OFFSET, f + OFFSET, 0.25)
        x = f - zi
        # Cast the result to meters
        y = 1 / (((1 / f) - (1 / zi)) * 1e3)
        pyplot.plot(x, y, label=f"f:{f}mm", linewidth=2)

    # Then plot the infinity line and label everything
    pyplot.plot([0, 0], [-1e5, 1e5], 'k--', label="infinity line")
    pyplot.legend()
    pyplot.xlim(-OFFSET, OFFSET)
    pyplot.ylim(-50, 50)
    pyplot.xlabel("∆zi = f - zi (∆ between focal length and focused plane) (mm)")
    pyplot.ylabel("zo = distance at which object will be in focus (m)")
    pyplot.title("1) Object focus distance by distance from focal length")
    pyplot.show()


# Compute the depth of field (minimum and maximum focus distances) for a given
# focus setting zo as a function of the circle of confusion diameter c (make it
# a fraction of the sensor width), the focal length f, and the f-stop number N
# (which relates to the aperture diameter d).
def part_2(c_frac=0.001):
    """Used for inspiration:
    https://en.wikipedia.org/wiki/Circle_of_confusion

    Check out the wiki page for the derivation, but we get
        c = (|zo1 - zo2| / zo2) * (f^2 / N (zo1 - f))
    where zo1 is the correctly focused object distance, and zo2 is the object
    distance where the circle of confusion is being assessed at. From there:
        c N (zo1 - f) / f^2 = |zo1 - zo2| / zo2
    Both of these are valid (+-)
        zo2 =  zo1 / ([cN(zo1 - f) / f^2] + 1)
        zo2 = -zo1 / ([cN(zo1 - f) / f^2] - 1)

    Arguments:
        c_frac: fraction of the sensor width that would be allowable in the
            circle of confusion (unitless, 0-1)
    """

    # Get the circle of confusion in mm
    W = 35
    c = c_frac * W

    # Iterate over a few focal lengths (mm)
    for f in [60, 100, 140]:
        # And f-numbers (unitless)
        for N in [8, 4, 2]:
            x = []
            y = []
            # Object distances at which the system is tuned (mm)
            for zo1 in numpy.arange(1, 20, 0.1) * 1e3:
                # Calculate upper and lower boundaries for the desired circle
                # of confusion. Throw away all points that are less than 0,
                # as that would be behind the lens and doesn't make sense/
                for zo2 in good_enough_depths(c, N, zo1, f):
                    if zo2 > 0:
                        x.append(zo1)
                        y.append(zo2)

            # Cast the points as meters and plot
            x = numpy.array(x) * 1e-3
            y = numpy.array(y) * 1e-3
            pyplot.plot(x, y, "o", label=f"f:{f}mm, N:{N}")

    # Plot the actually focused line
    pyplot.plot([0, 25], [0, 25], "k--", label="1:1 line of focus")
    # Then do axes and labels
    pyplot.title("2) Circle of confusion is {:.1f}% of sensor width".format(c_frac * 100))
    pyplot.xlabel("Focused distance (zo1, m)")
    pyplot.ylabel("Upper/lower bounds, staying within the circle of confusion (m)")
    pyplot.xlim(0, 20)
    pyplot.ylim(0, 20)
    pyplot.legend()
    pyplot.show()


def good_enough_depths(c, N, zo, f):
    """
    Calculate the upper and lower boundaries for a certain circle of confusion
    size. These two boundaries represent "undershot" and overshot" focuses,
    both of which exist. See part_2() for equation.
    """
    return [
         zo / ((c*N*(zo - f) / f**2) + 1),
        -zo / ((c*N*(zo - f) / f**2) - 1),
    ]


# Now consider a zoom lens with a varying focal length f. Assume that as you
# zoom, the lens stays in focus, i.e., the distance from the rear nodal point
# to the sensor plane zi adjusts itself automatically for a fixed focus
# distance zo. How do the depth of field indicators vary as a function of focal
# length?
def part_3(c_frac=0.001):
    """See docstring for part_2, this is just a shuffling of for loops."""

    # Get the circle of confusion in mm
    W = 35
    c = c_frac * W

    # Set an arbitrary f-number
    N = 8

    # Iterate over object distances at which the system is tuned (mm)
    for zo1 in numpy.arange(3, 16, 3) * 1e3:
        x = []
        y = []
        # And dense focal lengths (mm)
        for f in numpy.arange(60, 140, 1):
            for zo2 in good_enough_depths(c, N, zo1, f):
                x.append(f)
                y.append(zo2 - zo1)
        # Cast the points as meters and plot
        x = numpy.array(x)
        y = numpy.array(y) * 1e-3
        pyplot.plot(x, y, "o", label=f"zo:{zo1*1e-3}m")

    # Then do axes and labels
    pyplot.title("3) Circle of confusion is {:.1f}% of sensor width, f-number N: {}".format(c_frac * 100, N))
    pyplot.xlabel("Focal length (f, mm)")
    pyplot.ylabel("Upper/lower bounds within the circle of confusion, zeroed on focus distance (m)")
    pyplot.ylim(-8, 30)
    pyplot.legend()
    pyplot.show()


def main():
    part_1()
    part_2()
    part_3()


if __name__ == "__main__":
    main()
