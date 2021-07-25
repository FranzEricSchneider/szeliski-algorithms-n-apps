import cv2
from matplotlib import pyplot
import numpy

from skimage import color


def part_0(step=0.1):
    """Interactively adjust colors with the keyboard.

    Write a simple application to change the color balance of an image by
    multiplying each color value by a different user-specified constant.
    """

    # Read in a BGR image and display it
    image = cv2.imread("ex_3_1/example_image.png")
    cv2.imshow("example image", image)

    # Helptext
    print("Press <q> to quit")
    print("<e/r> will -/+ the red values")
    print("<f/g> will -/+ the green values")
    print("<v/b> will -/+ the blue values")

    while True:
        key_press = cv2.waitKey(0)
        if key_press in map(ord, 'rgbefvq'):
            cv2.destroyAllWindows()

            # Bump values up and down
            image = image.astype(float)
            if key_press == ord('q'):
                break
            elif key_press == ord('r'):
                image[:, :, 2] *= (1.0 + step)
            elif key_press == ord('e'):
                image[:, :, 2] *= (1.0 - step)
            elif key_press == ord('g'):
                image[:, :, 1] *= (1.0 + step)
            elif key_press == ord('f'):
                image[:, :, 1] *= (1.0 - step)
            elif key_press == ord('b'):
                image[:, :, 0] *= (1.0 + step)
            elif key_press == ord('v'):
                image[:, :, 0] *= (1.0 - step)

            # Clamp to 0-255 and cast as int, the redisplay
            image[image > 255] = 255
            image = image.astype(numpy.uint8)
            cv2.imshow("example image", image)


def part_1():
    """
    Do you get different results if you take out the gamma transformation
    before or after doing the multiplication?
    """

    # Gather the three images we want to compare and store them as RGB, so we
    # can use matplotlib display down below
    original = cv2.cvtColor(cv2.imread("ex_3_1/example_image.png"),
                            cv2.COLOR_BGR2RGB)
    pregamma = original.astype(float)
    postgamma = original.astype(float)

    # Define gamma correction and an arbitrary scaling, then apply them in
    # varying orders
    def gamma_correction(image):
        """
        If no color profile is embedded, then a standard gamma of 1/2.2 is
        usually assumed.
        """
        return ((image / numpy.max(image)) ** 2.2) * 255

    def scale(image):
        image[:, :, 0] *= 0.6
        image[:, :, 1] *= 1.3
        image[:, :, 2] *= 1.7
        return image

    pregamma = scale(gamma_correction(pregamma))
    postgamma = gamma_correction(scale(postgamma))

    # Clamp everything and cast to int
    def clampcast(image):
        image[image > 255] = 255
        return image.astype(numpy.uint8)
    pregamma = clampcast(pregamma)
    postgamma = clampcast(postgamma)

    # Display the images
    _, axes = pyplot.subplots(1, 3)
    axes[0].set_title("Original")
    axes[0].imshow(original)
    axes[1].set_title("Gamma correction then multiply")
    axes[1].imshow(pregamma)
    axes[2].set_title("Multiply then gamma correction")
    axes[2].imshow(postgamma)
    pyplot.show()


def part_2():
    """
    Can you recover what the color balance ratios are between the different
    settings?
    """

    # I got these images from wikipedia on color balancing
    prefix = "ex_3_1/part2_im"
    names = ["{}{}.jpg".format(prefix, i+1) for i in range(5)]
    images = [cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
              for name in names]

    # Extract the comparison image, and cast as a float so it can be negative
    # and eventually fractional
    baseline = images[0].astype(float)
    images = images[1:]

    # As a hack, to avoid divide by zero stuff, make 0 values up to 1
    baseline[baseline == 0] = 1e-6

    # Get the pixel by pixel delta or scale values for each pair of image vs.
    # baseline. When you look at the output images, what you should see is
    # 1) baseline
    # 2) comparison image
    # 3) processed baseline which should look like the comparison image
    # METANOTE: This was an attempt to see if there was an average additive or
    # multiplicative change to the color channels. Long story short, straight
    # average addition or multiplication does not seem to adjust the colors
    # very well to match the original. I'll keep this around for comparison.
    deltas = [image - baseline for image in images]
    scales = [image / baseline for image in images]
    for i, name in enumerate(names[1:]):
        print(f"Image {i+1}", "_deltas.")
        evaluate_image(baseline.copy(), deltas[i], name, "_deltas.", add=True)
        evaluate_image(baseline.copy(), scales[i], name, "_scales.", add=False)

    # Next I tried looking up the mathematics of color balancing:
    # https://en.wikipedia.org/wiki/Color_balance#Mathematics_of_color_balance

    # The first thing it said was that a straight multiplication operation,
    # the same scaling value applied per channel to each pixel, has issues: "It
    # has been demonstrated that performing the white balancing in the phosphor
    # set assumed by sRGB tends to produce large errors in chromatic colors,
    # even though it can render the neutral surfaces perfectly neutral"

    # "If the image may be transformed into XYZ values, the color balancing may
    # be performed there. This is called a “wrong von Kries” transformation."
    # It's less good than balancing in RGB, apparently, and is mentioned just
    # as a bridge to other things.

    # "von Kries suggested converting color to the LMS color space, a.k.a. the
    # long/medium/short wavelength cone types. A 3x3 matrix converts RGB/XYZ to
    # LMS, then the three LMS values are scaled to balance the neutral; the
    # color can then be converted back to the desired final color space"

    # WTF no idea
    # "The best color matrix for adapting to a change in illuminant is not
    # necessarily a diagonal matrix in a fixed color space. If the space of
    # illuminants can be described as a linear model with N basis terms,
    # the proper color transformation will be the weighted sum of N fixed
    # linear transformations."

    # Okay, so multiplication by channel is likely the right answer here, but
    # the issue with the given images is that it's very unclear what color
    # space the given images were corrected in. Almost certainly not the
    # RGB that the image data is in. So there's likely a hidden 3x3
    # transformation muddling up the scalars I've been trying to get.

    # Conclusion: Try a few common color spaces and see if you can spot the
    # chosen one (tight variances on the scale), then move on
    scaled_baseline = baseline / 255
    avg_deviations = {
        "hsv": roundtrip(scaled_baseline, images, color.rgb2hsv, color.hsv2rgb,
                         names[1:], suffix="_hsv_scales."),
        "lab": roundtrip(scaled_baseline, images, color.rgb2lab, color.lab2rgb,
                         names[1:], suffix="_lab_scales."),
        "xyz": roundtrip(scaled_baseline, images, color.rgb2xyz, color.xyz2rgb,
                         names[1:], suffix="_xyz_scales."),
    }
    for key, value in avg_deviations.items():
        print(f"{key}: {value[0]}")

    # At the end of this, I don't love any of the transforms. None of them have
    # tiny deviations, leading me to believe that none of them are an exact
    # match for what the original modifier did. In the end I think the
    # (RGB > XYZ > scale > RGB) is slightly better than just (RGB > scale), and
    # they are both actually somewhat reasonable.


def evaluate_image(baseline, processed, name, suffix, add):
    """
    Print some by-channel stats about the given processed image, supporting
    both addition (add=True) and multiplication (add=False). Also write an
    image where the average transform has been applied to the baseline again.
    """
    if add:
        vectors = [processed[:, :, i].flatten() for i in range(3)]
    else:
        # When dealing with division, cut out places where the baseline value
        # is low, which would add more noise to the divided value
        vectors = [processed[:, :, i][baseline[:, :, i] > 10].flatten()
                   for i in range(3)]

    # Calculate stats for the chosen process, then print
    averages = list(map(numpy.average, vectors))
    std_dev = list(map(numpy.std, vectors))
    if add:
        print("Deltas")
    else:
        print("Scales")
    for i in range(3):
        print(f"avg: {averages[i]:.3f}, std: {std_dev[i]:.4f}")

    # Then write an example image where we try to apply the average transform
    # to the whole baseline image
    for i in range(3):
        if add:
            baseline[:, :, i] += averages[i]
        else:
            baseline[:, :, i] *= averages[i]
    # Clamp to 0-255
    baseline[baseline < 0] = 0
    baseline[baseline > 255] = 255
    # Write
    cv2.imwrite(suffix.join(name.split(".")),
                cv2.cvtColor(baseline.astype(numpy.uint8), cv2.COLOR_RGB2BGR))


# This is bad practice, stuffing it all into one function, but convenience!
def roundtrip(scaled_base, images, to_func, from_func, names, suffix):

    processed_base = to_func(scaled_base)
    # Hack to avoid divide by zero
    processed_base[processed_base == 0] = 1e-6

    deviations = []
    roundtrip_images = []
    for image in images:
        processed_image = to_func(image.astype(float) / 255)
        scalars = processed_image / processed_base
        limited_values = [scalars[:, :, i][processed_base[:, :, i] > 0.05]
                          for i in range(3)]
        averages = [numpy.average(limited_values[i]) for i in range(3)]
        deviations.append([numpy.std(limited_values[i]) for i in range(3)])

        # Process the baseline using the average scale
        new_image = processed_base.copy()
        for i in range(3):
            new_image[:, :, i] *= averages[i]
        roundtrip_images.append(from_func(new_image) * 255)

    # Write the image out as debug
    for image, name in zip(roundtrip_images, names):
        image[image < 0] = 0
        image[image > 255] = 255
        cv2.imwrite(suffix.join(name.split(".")),
                    cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_RGB2BGR))

    return (numpy.average(deviations, axis=0), roundtrip_images)


def main():
    # part_0()
    # part_1()
    part_2()


if __name__ == "__main__":
    main()
