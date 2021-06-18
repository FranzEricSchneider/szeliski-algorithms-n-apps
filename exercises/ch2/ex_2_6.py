import cv2

from matplotlib import pyplot
import numpy


# Always
numpy.set_printoptions(suppress=True)

# Maps ISO number to image paths
PATH_MAPPING = {
    1600: [
        "ex_2_6_images/IMG_5985.JPG",
        "ex_2_6_images/IMG_5986.JPG",
        "ex_2_6_images/IMG_5987.JPG",
        "ex_2_6_images/IMG_5988.JPG",
        "ex_2_6_images/IMG_5989.JPG",
    ],
    800: [
        "ex_2_6_images/IMG_5990.JPG",
        "ex_2_6_images/IMG_5991.JPG",
        "ex_2_6_images/IMG_5992.JPG",
        "ex_2_6_images/IMG_5993.JPG",
        "ex_2_6_images/IMG_5994.JPG",
    ],
}


# Plot your estimated variance as a function of level for each of your color
# channels. Does the amount of noise vary a lot with ISO/gain?
def plot_color_noise(iso, image_paths, axes):
    # Make a big old array of all the images in the batch, numpy can deal with
    # the indices. The shape will be something like (5, 2848, 4272, 3).
    image_stack = numpy.array([cv2.imread(path) for path in image_paths])
    # We can't get the variance by just doing var(R) where R is all of the red
    # values, since that ignores the underlying color of each pixel. However,
    # I think we can get the variance for mean-adjusted pixels just fine?
    mean = numpy.mean(image_stack, axis=0)
    mean_adjusted = image_stack - mean

    # Work out the data as a function by level
    num_bins = 25
    levels = numpy.linspace(0, 255, num_bins)
    positions = [numpy.mean([levels[i], levels[i+1]])
                 for i in range(num_bins - 1)]
    # Get separate samples for B, G, R
    binned_dev = [[], [], []]
    prevalence = [[], [], []]
    total_samples = float(numpy.product(image_stack.shape[0:3]))
    for i in range(3):
        for j in range(num_bins - 1):
            in_range = (mean[:, :, i] >= levels[j]) & (mean[:, :, i] <= levels[j+1])
            samples = numpy.array([
                mean_adjusted[k, :, :, i][in_range]
                for k in range(mean_adjusted.shape[0])
            ])
            # Calculate the variance of selected (mean adjusted) samples. Note
            # that numpy.var naturally flattens the given matrices
            binned_dev[i].append(numpy.sqrt(numpy.var(samples)))
            # Then check what percentage of the total sample is found within
            # this bin, as size will have a large effect on variance
            prevalence[i].append(100 * numpy.product(samples.shape) / total_samples)

    # Plot the std deviation and prevalence by color
    for i, color in zip(range(3), list("bgr")):
        axes[0].plot(positions, prevalence[i], f"{color}")
        axes[1].plot(positions, binned_dev[i], f"{color}o")


# Plot your estimated variance as a function of level for each of your color
# channels. Does the amount of noise vary a lot with ISO/gain?
def plot_hists(iso, image_paths):
    # Gather the images
    images = [cv2.imread(path) for path in image_paths]
    for image in images:
        for i in range(3):
            # Take the BGR index and invert it to give pyplot RGB colors
            color = [0] * 3
            # Make lower gain (darker images) have the darker colors
            color[2-i] = iso / 3200.0 + 0.5
            # The random bin numbers is to jitter it a bit for better viewing
            pyplot.hist(image[:, :, i].flatten(),
                        bins=numpy.random.randint(85, 100),
                        color=color,
                        histtype="step")


def main():
    figure, axes = pyplot.subplots(len(PATH_MAPPING))
    for axis, (iso, image_paths) in zip(axes, PATH_MAPPING.items()):
        alternate_axis = axis.twinx()
        plot_color_noise(iso, image_paths, [axis, alternate_axis])

        axis.set_title(f"Variance by level at ISO: {iso}")
        axis.set_xlabel("Mean color value for a given set of pixels ('level', 0-255)")
        axis.set_ylabel("[Line] Percentage of points in this bin (%)")
        alternate_axis.set_ylabel("[Dot] Standard Deviation (color values, 0-255)")
    pyplot.show()

    # for iso, image_paths in PATH_MAPPING.items():
    #     plot_hists(iso, image_paths)
    # pyplot.xlabel("Color value (uint8)")
    # pyplot.ylabel("Number of pixels at that value")
    # pyplot.title("Distribution of color scores, lower ISO has darker colors (800/1600)")
    # pyplot.show()


if __name__ == "__main__":
    main()
