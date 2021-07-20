import cv2
from matplotlib import pyplot
import numpy


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
                            cv2.COLOR_RGB2BGR)
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


def main():
    # part_0()
    part_1()


if __name__ == "__main__":
    main()
