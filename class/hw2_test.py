import numpy

from hw2 import blur, difference, downsample


def test_downsample():
    image = numpy.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1],
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2],
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3],
        [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4],
        [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5],
        [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
        [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7],
        [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8],
        [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ], dtype=numpy.uint8)

    output = downsample(image, num_octaves=3)
    assert len(output) == 3
    assert output[0] is not image

    assert output[0].shape == (12, 12)
    assert numpy.allclose(output[0], image)
    assert output[1].shape == (6, 6)
    assert numpy.allclose(output[1], image[::2, ::2])
    assert output[2].shape == (3, 3)
    assert numpy.allclose(output[2], image[::4, ::4])


def test_blur():
    image = numpy.array([
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 255, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
        [0, 0, 0, 0, 0,   0,   0, 0, 0, 0],
    ], dtype=numpy.uint8)
    downsampled = downsample(image, 2)

    kernels = [
        numpy.array([0, 0, 1, 0, 0], dtype=float).reshape((5, 1)),
        numpy.array([0, 1, 1, 1, 0], dtype=float).reshape((5, 1)),
        numpy.array([1, 1, 1, 1, 1], dtype=float).reshape((5, 1)),
    ]
    blurred = blur(downsampled, kernels)

    # At the top level, we should have the same number of groups as octaves
    assert len(blurred) == 2
    # At the next level down, we should have one image per kernel
    for octave in blurred:
        assert len(octave) == 3
    # Then the image size, dependent on the downsampling
    for image in blurred[0]:
        assert image.shape == (10, 10)
    for image in blurred[1]:
        assert image.shape == (5, 5)

    # Make some spot checks for values
    assert numpy.allclose(blurred[0][1], numpy.array([
        [[0, 0, 0, 0,   0,   0,   0,   0, 0, 0],
         [0, 0, 0, 0,   0,   0,   0,   0, 0, 0],
         [0, 0, 0, 0,   0,   0,   0,   0, 0, 0],
         [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
         [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
         [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
         [0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
         [0, 0, 0, 0,   0,   0,   0,   0, 0, 0],
         [0, 0, 0, 0,   0,   0,   0,   0, 0, 0],
         [0, 0, 0, 0,   0,   0,   0,   0, 0, 0]]
    ]))
    assert numpy.allclose(blurred[1][0], numpy.array([
        [[0, 0, 0,   0, 0],
         [0, 0, 0,   0, 0],
         [0, 0, 255, 0, 0],
         [0, 0, 0,   0, 0],
         [0, 0, 0,   0, 0]]
    ]))
    assert numpy.allclose(blurred[1][2], numpy.array([
        [[255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255]]
    ]))


def test_difference():
    octaves = [
        [
            numpy.array([[1, 2, 3, 4],
                         [2, 3, 4, 1],
                         [3, 4, 1, 2],
                         [4, 1, 2, 3]], dtype=numpy.uint8),
            numpy.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype=numpy.uint8),
        ],
        [
            numpy.array([[101, 10], [11, 1]], dtype=numpy.uint8),
            numpy.array([[2, 22], [20, 202]], dtype=numpy.uint8),
        ]
    ]
    output = difference(octaves)

    # Check size and type
    assert len(output) == 2
    for octave in output:
        assert len(octave) == 1
    assert output[0][0].shape == (4, 4)
    assert output[0][0].dtype == int
    assert output[1][0].shape == (2, 2)
    assert output[1][0].dtype == int

    # Check values
    assert output[0][0] == -octaves[0][0].astype(int)
    assert output[0][1] == octaves[1][1].astype(int) - octaves[1][0].astype(int)
