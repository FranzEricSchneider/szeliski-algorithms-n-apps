import cv2
from matplotlib import pyplot
import numpy
import rawpy
from scipy.ndimage import convolve


# NOTE TO SELF - eog seems to be able to view raw (dng) files, but poorly.
# Shotwell (installed by default?) seemed to do a good job.


def main():
    """
    Perform the demosaicing yourself on a RAW image. Instead of just bilinear
    interpolation, try one of the more advanced techniques described in Section
    10.3.1. Compare your result to the one produced by the camera. Does your
    camera perform a simple linear mapping between RAW values and the
    color-balanced values in a JPEG?

    NOTE: I will not be sharpening the raw image, or color balancing it, or
    doing gamma correction, which apparently often happens to raw in practice.
    """

    # Get the raw image
    raw = rawpy.imread("ex_3_2/raw.dng")
    # For now, assert a consistent color order, re-evaluate things if and image
    # is used where this is false
    assert raw.color_desc == b"RGBG"

    # Get the camera-created jpg
    comparison = cv2.cvtColor(cv2.imread("ex_3_2/comparison.jpg"),
                              cv2.COLOR_BGR2RGB)

    # First off, double check which pixels correspond to which pixels. I'm
    # assuming R starts off at (0, 0), with B at (1, 1) and G at (0, 1)/(1, 0),
    # but let's check that
    # CONCLUSION 1) This is really hard to read and tell which color stream is
    #   which. I'm pretty positive (0, 1)/(1, 0) are green, since they seem to
    #   match. I guess beyond that it doesn't really matter. It's hard because
    #   white shows up strong in all versions.
    # CONCLUSION 2) I'm pretty sure (0, 0) is blue and (1, 1) is red, because
    #   there's a red pot that shows up a bit lighter on (1, 1) when you zoom
    #   in. In addition, the orange/yellow ribbons seem much lighter in (1, 1).
    plot_decimated_view(raw)

    # Create a bilinear image (simple average of neighbors)
    bilinear = scale_to_uint8(bilinear_interpolate(raw.raw_image), fraction=0.4)
    pyplot.imshow(bilinear)
    pyplot.show()

    # Okay, as far as I can tell I've done bilinear interpolation to do
    # demosaicing correctly, but It looks super weird and green. As far as I
    # can tell that's an accurate representation of the underlying pixel values
    # in the Bayer pattern, on surfaces that I would call white or light grey
    # the raw green values are significantly stronger. I've done some more
    # research about the raw/jpg conversion process and I think that the
    # demosaicing has been done alright and the de-greening would happen
    # naturally during the white balance step.
    # I considered doing more than just bilinear interpolation but I'm going
    # to leave it there since this has been a saga.

    # https://helpx.adobe.com/lightroom-cc/how-to/raw-vs-jpeg.html
    # A JPEG, even one that is straight out of the camera, has already been
    # “developed” by the camera’s image processor. Settings such as brightness,
    # contrast, color saturation, and even sharpening may have already been
    # applied. The look of a JPEG image can be changed in an image editing
    # application, but since it is a compressed format designed to yield
    # smaller file sizes, a lot of tonal and color data has been permanently
    # discarded during the compression process.
    # With a JPEG, white balance is applied by the camera, and there are fewer
    # options to modify it in post-processing. With a raw file, you have
    # complete control over white balance when editing the image.

    # https://en.wikipedia.org/wiki/Raw_image_format#Standardization
    # To be viewed or printed, the output from a camera's image sensor has to
    # be processed, that is, converted to a photographic rendering of the
    # scene, and then stored in a standard raster graphics format such as JPEG.
    # This processing, whether done in-camera or later in a raw-file converter,
    # involves a number of operations, typically including:
    # decoding – image data of raw files are typically encoded for compression
    #   purpose, but also often for obfuscation.
    # demosaicing – interpolating the partial raw data received from the
    #   color-filtered image sensor into a matrix of colored pixels.
    # defective pixel removal – interpolating over data in known bad locations.
    # white balancing – accounting for color temperature of the light.
    # noise reduction – trading off detail for smoothness.
    # color translation – converting from the camera native color space defined
    #   by the spectral sensitivities of the image sensor to an output color
    #   space (typically sRGB for JPEG)
    # tone reproduction – the scene luminance captured by the camera sensors
    #   and stored in the raw file (with a dynamic range of typically 10 or
    #   more bits) needs to be rendered for pleasing effect and correct
    #   viewing on low-dynamic-range monitors or prints; the tone-reproduction
    #   rendering often includes separate tone mapping and gamma compression
    #   steps.
    # compression – for example JPEG compression

    # Cameras and image processing software may also perform additional
    # processing to improve image quality, for example:
    # removal of systematic noise – bias frame subtraction and flat-field
    #   correction
    # dark frame subtraction
    # optical correction – lens distortion, vignetting, chromatic aberration
    #   and color fringing correction
    # contrast manipulation
    # increasing visual acuity by unsharp masking
    # dynamic range compression – lighten shadow regions without blowing out
    #   highlight regions


def plot_decimated_view(raw):
    """
    Plot 2x2 view of the decimated raw image to try and track down which pixels
    are R, G, and B.
    """

    # Some initial questions.
    # * Why is the raw image uint16, but only has a max value of 16,460? The
    #   max would be 65,536
    # * Why is the minimum value 1015? Is there just nothing that black in the
    #   image?

    # Make a 2x2 plot and stick the images on it
    _, axes = pyplot.subplots(2, 2)
    def imshow(i, j):
        # Scale to 255 and cast as int. Do some overscaling here so that
        # features really pop to the human eye, that aren't just the white
        # sunlight
        decimated = raw.raw_image[i::2, j::2]
        scaled = 255 * decimated.astype(float) / (0.15 * numpy.max(decimated))
        scaled[scaled > 255] = 255
        scaled = scaled.astype(numpy.uint8)
        axes[i, j].imshow(scaled)
    imshow(0, 0)
    imshow(1, 0)
    imshow(0, 1)
    imshow(1, 1)
    pyplot.show()


# This function is stupid and inefficient, made when I was poking around and
# figuring out how to make the right thing happen, but I think it's not worth
# bothering to tune up. It served its purpose.
def bilinear_interpolate(patterned_image):
    """Interpolate RGB patterned pixels into a full RGB image.

    NOTE: This assumes [BG, BR] because that's what the pattern on my example
    images is. This function isn't made for a general pattern.

    Arguments:
        patterned_image: (n, m) shaped image in a Bayer pattern
    """

    r = numpy.zeros(patterned_image.shape, dtype=float)
    g = numpy.zeros(patterned_image.shape, dtype=float)
    b = numpy.zeros(patterned_image.shape, dtype=float)

    # I hate this, this seems really inefficient, but three kernels for
    # different pixel weightings
    r[1::2, 1::2] = patterned_image[1::2, 1::2]
    r_kernel_x = numpy.array([[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]])
    r_convolved_x = convolve(r, r_kernel_x, mode="mirror")
    r_kernel_h = numpy.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
    r_convolved_h = convolve(r, r_kernel_h, mode="mirror")
    r_kernel_v = numpy.array([[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    r_convolved_v = convolve(r, r_kernel_v, mode="mirror")
    r[0::2, 0::2] = r_convolved_x[0::2, 0::2]
    r[1::2, 0::2] = r_convolved_h[1::2, 0::2]
    r[0::2, 1::2] = r_convolved_v[0::2, 1::2]

    g[0::2, 1::2] = patterned_image[0::2, 1::2]
    g[1::2, 0::2] = patterned_image[1::2, 0::2]
    g_kernel = numpy.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    g_convolved = convolve(g, g_kernel, mode="mirror")
    g[0::2, 0::2] = g_convolved[0::2, 0::2]
    g[1::2, 1::2] = g_convolved[1::2, 1::2]

    b[0::2, 0::2] = patterned_image[0::2, 0::2]
    b_kernel_x = numpy.array([[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]])
    b_convolved_x = convolve(b, b_kernel_x, mode="mirror")
    b_kernel_h = numpy.array([[0, 0, 0], [0.5, 0, 0.5], [0, 0, 0]])
    b_convolved_h = convolve(b, b_kernel_h, mode="mirror")
    b_kernel_v = numpy.array([[0, 0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    b_convolved_v = convolve(b, b_kernel_v, mode="mirror")
    b[1::2, 1::2] = b_convolved_x[1::2, 1::2]
    b[1::2, 0::2] = b_convolved_v[1::2, 0::2]
    b[0::2, 1::2] = b_convolved_h[0::2, 1::2]

    return numpy.dstack((r, g, b))


def scale_to_uint8(image, fraction):
    """Scale the max value to 255/fraction and cast to uint8.

    Arguments:
        image: (N, M, 3) image of some dtype. Probably float?
        fraction: The fraction of the max value that we want to treat as 255.
            For example, if the fraction is 0.5 and the max value is 1500, then
            everything over 750 will be treated as 255 in uint8 land. I'm still
            working through why the raw images are so dark unless this is done.
    """
    scaled = image * 255 / (fraction * numpy.max(image))
    scaled[scaled > 255] = 255
    return scaled.astype(numpy.uint8)


if __name__ == "__main__":
    main()
