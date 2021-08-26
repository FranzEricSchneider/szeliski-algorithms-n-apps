import numpy
import pytest
import types

from hw2 import blur, differences, downsample, extremities


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


def test_differences():
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
    output = differences(octaves)

    # Check size and type
    assert len(output) == 2
    for octave in output:
        assert len(octave) == 1
    assert output[0][0].shape == (4, 4)
    assert output[0][0].dtype == int
    assert output[1][0].shape == (2, 2)
    assert output[1][0].dtype == int

    # Check values
    assert numpy.allclose(output[0][0], -octaves[0][0].astype(int))
    assert numpy.allclose(
        output[1][0], octaves[1][1].astype(int) - octaves[1][0].astype(int)
    )


@pytest.fixture
def empty():
    return [
        [
            numpy.zeros((3, 3), dtype=int),
            numpy.zeros((3, 3), dtype=int),
            numpy.zeros((3, 3), dtype=int),
        ]
    ]


class TestExtremities:

    def test_no_extremes(self, empty):
        assert list(extremities(empty)) == []

    @pytest.mark.parametrize("value", [100, -100])
    def test_basic(self, value, empty):
        # Set the middle value of the middle array in the first octave to value
        empty[0][1][1, 1] = value
        assert isinstance(extremities(empty), types.GeneratorType)
        assert list(extremities(empty)) == [[0, 1, 1, 1]]

    def test_octaves(self):
        diffed = [
            [
                numpy.zeros((6, 6), dtype=int),
                numpy.zeros((6, 6), dtype=int),
                numpy.zeros((6, 6), dtype=int),
            ],
            [
                numpy.zeros((3, 3), dtype=int),
                numpy.zeros((3, 3), dtype=int),
                numpy.zeros((3, 3), dtype=int),
            ],
        ]
        diffed[0][1][1, 1] = -10
        diffed[0][1][2, 4] = 20
        diffed[0][1][4, 3] = -20
        diffed[1][1][1, 1] = 10
        assert list(extremities(diffed)) == [
            [0, 1, 1, 1],
            [0, 1, 2, 4],
            [0, 1, 4, 3],
            [1, 1, 1, 1],
        ]

    def test_realistic(self):
        diffed = [
            [   # Fill with random numbers from 0-10
                (numpy.random.random((6, 6)) * 10).astype(int),
                (numpy.random.random((6, 6)) * 10).astype(int),
                (numpy.random.random((6, 6)) * 10).astype(int),
            ],
        ]
        # Set up a high value (1, 1) where it should be overshadowed by another
        # at (2, 2).
        diffed[0][1][1, 1] = 20
        diffed[0][1][2, 2] = 30
        # Then set up a minimum value, also with a competitor
        diffed[0][1][3, 4] = -20
        diffed[0][1][4, 4] = -15
        assert list(extremities(diffed)) == [[0, 1, 2, 2], [0, 1, 3, 4]]
