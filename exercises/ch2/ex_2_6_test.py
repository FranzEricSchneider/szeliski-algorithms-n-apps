import numpy


def test_mean_shape():
    """
    Sort of a weird test since I'm testing numpy stuff, but I want to confirm
    that the matrix shapes work the way I expect.
    """

    # Make some dirt-simple RGB images
    image1 = numpy.array([
        [[100, 100, 100], [100, 100, 100]],
        [[100, 100, 100], [100, 100, 100]],
        [[100, 100, 100], [100, 100, 100]],
    ], dtype=numpy.uint8)
    image2 = numpy.array([
        [[150,  90, 120], [180,   0,  20]],
        [[180,   0,  20], [100, 100, 100]],
        [[100, 100, 100], [150,  90, 120]],
    ], dtype=numpy.uint8)
    assert image1.shape == image2.shape == (3, 2, 3)

    image_stack = numpy.array([image1, image2])
    assert image_stack.shape == (2, 3, 2, 3)

    # Check that we can get the mean on each color channel
    mean = numpy.mean(image_stack, axis=0)
    assert mean.shape == (3, 2, 3)

    # Each pixel should be the mean along that color
    assert numpy.all(mean == numpy.array([
        [[125,  95, 110], [140,  50,  60]],
        [[140,  50,  60], [100, 100, 100]],
        [[100, 100, 100], [125,  95, 110]],
    ], dtype=numpy.uint8))

    # Check that we can get all of a certain color
    assert numpy.all(image_stack[:, :, :, 0].flatten() == numpy.array(
        [100, 100, 100, 100, 100, 100, 150, 180, 180, 100, 100, 150],
        dtype=numpy.uint8
    ))
    assert numpy.all(image_stack[:, :, :, 2].flatten() == numpy.array(
        [100, 100, 100, 100, 100, 100, 120, 20, 20, 100, 100, 120],
        dtype=numpy.uint8
    ))