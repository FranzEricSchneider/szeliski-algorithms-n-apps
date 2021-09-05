import numpy
import pytest
import types

from hw2 import (blur,
                 derivative,
                 differences,
                 downsample,
                 extremities,
                 hessian,
                 )


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
            numpy.array([[1, 2, 3, 4.1],
                         [2, 3, 4.1, 1],
                         [3, 4.1, 1, 2],
                         [4.1, 1, 2, 3]], dtype=float),
            numpy.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype=float),
        ],
        [
            numpy.array([[101, 10.5], [11, 1]], dtype=float),
            numpy.array([[2, 22], [20.5, 202]], dtype=float),
        ]
    ]
    output = differences(octaves)

    # Check size and type
    assert len(output) == 2
    for octave in output:
        assert len(octave) == 1
    assert output[0][0].shape == (4, 4)
    assert output[0][0].dtype == float
    assert output[1][0].shape == (2, 2)
    assert output[1][0].dtype == float

    # Check values
    assert numpy.allclose(output[0][0], -octaves[0][0])
    assert numpy.allclose(
        output[1][0], octaves[1][1] - octaves[1][0]
    )


@pytest.fixture
def empty():
    """
    Make a set of empty images. Note that these must be bigger than (16, 16)
    because of how much buffer we leave around the image edge for later
    SIFTing. See extremities() for details.
    """
    return [
        [
            numpy.zeros((17, 17), dtype=int),
            numpy.zeros((17, 17), dtype=int),
            numpy.zeros((17, 17), dtype=int),
        ]
    ]


class TestExtremities:

    def test_no_extremes(self, empty):
        assert list(extremities(empty)) == []

    @pytest.mark.parametrize("value", [100, -100])
    def test_basic(self, value, empty):
        # Set the middle value of the middle array in the first octave to value
        empty[0][1][8, 8] = value
        print(empty)
        assert isinstance(extremities(empty), types.GeneratorType)
        assert list(extremities(empty)) == [[0, 1, 8, 8]]

    def test_octaves(self):
        diffed = [
            [
                numpy.zeros((34, 34), dtype=int),
                numpy.zeros((34, 34), dtype=int),
                numpy.zeros((34, 34), dtype=int),
            ],
            [
                numpy.zeros((17, 17), dtype=int),
                numpy.zeros((17, 17), dtype=int),
                numpy.zeros((17, 17), dtype=int),
            ],
        ]
        diffed[0][1][12, 20] = 20
        diffed[0][1][16, 16] = -10
        diffed[0][1][22, 22] = -20
        diffed[1][1][8, 8] = 10
        assert list(extremities(diffed)) == [
            [0, 1, 12, 20],
            [0, 1, 16, 16],
            [0, 1, 22, 22],
            [1, 1, 8, 8],
        ]

    def test_realistic(self):
        diffed = [
            [   # Fill with random numbers from 0-10
                (numpy.random.random((18, 18)) * 10).astype(int),
                (numpy.random.random((18, 18)) * 10).astype(int),
                (numpy.random.random((18, 18)) * 10).astype(int),
            ],
        ]
        # Set up a high value where it should be overshadowed by another
        diffed[0][1][8, 8] = 20
        diffed[0][1][8, 9] = 30
        # Then set up a minimum value, also with a competitor
        diffed[0][1][9, 8] = -20
        diffed[0][1][9, 9] = -15
        assert list(extremities(diffed)) == [[0, 1, 8, 9], [0, 1, 9, 8]]


class TestDerivative:

    def test_derivative(self):
        # Run a couple of simple images through the algorithm
        blank = numpy.zeros((5, 5), dtype=numpy.uint8)
        xline = blank.copy()
        xline[:, 2] = 255
        yline = blank.copy()
        yline[2, :] = 255
        output = [derivative(image) for image in [blank, xline, yline]]

        # Assert types and sizes
        for image in output:
            assert image.dtype == float
            assert image.shape == (5, 5, 2)

        # The first image should be blank (no derivative)
        assert numpy.allclose(output[0], numpy.zeros((5, 5, 2)))

        # Make statements about the empty locations of the simple derivatives
        for full_column in [0, 2, 4]:
            assert (output[1][:, full_column] == 0).all()
        for half_column in [1, 3]:
            assert (output[1][:, half_column, 1] == 0).all()
        for full_row in [0, 2, 4]:
            assert (output[2][full_row, :] == 0).all()
        for half_row in [1, 3]:
            assert (output[2][half_row, :, 0] == 0).all()

        # I don't have a strong reason for why the derivative takes a certain
        # value, so I'll just be reasoning on the maximum value without making
        # statements about its actual value.
        value = numpy.max(output[1])
        assert (output[1][:, 1, 0] ==  value).all()
        assert (output[1][:, 3, 0] == -value).all()
        assert (output[2][1, :, 1] ==  value).all()
        assert (output[2][3, :, 1] == -value).all()

        # Here's an example of the simple image derivative output
        # array([[[0, 0], [ 4080, 0], [0, 0], [-4080, 0], [0, 0]],
        #        [[0, 0], [ 4080, 0], [0, 0], [-4080, 0], [0, 0]],
        #        [[0, 0], [ 4080, 0], [0, 0], [-4080, 0], [0, 0]],
        #        [[0, 0], [ 4080, 0], [0, 0], [-4080, 0], [0, 0]],
        #        [[0, 0], [ 4080, 0], [0, 0], [-4080, 0], [0, 0]]])

    def test_scaling(self):
        """Make sure the derivative returns reasonable scaling."""

        # Make a simple image that should have slope 1
        image = numpy.array([[1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5],
                             [1, 2, 3, 4, 5]])
        output = derivative(image)
        print(output)

        # Check that the x slope is 1 in the center (apparently there's a
        # buffer behavior that makes it 0 at the axis 0 edges, whatever)
        assert numpy.allclose(output[:, 1:4, 0], 1.0)

        # Check that it's 0 for all y values
        assert numpy.allclose(output[:, :, 1], 0.0)

        # It should be zero for all edge x values
        assert numpy.allclose(output[:, 0, 0], 0.0)
        assert numpy.allclose(output[:, 4, 0], 0.0)


class TestHessian:

    def test_basic(self):
        # Run a couple of simple images through the algorithm
        blank = numpy.zeros((5, 5), dtype=numpy.uint8)
        blip = blank.copy()
        blip[2, 2] = 255
        output = [hessian(image) for image in [blank, blip]]

        # Assert types and sizes
        for image in output:
            assert image.dtype == float
            assert image.shape == (5, 5, 2, 2)

        # The first image should be blank (no derivative)
        assert numpy.allclose(output[0], numpy.zeros((5, 5, 2, 2)))

        # Extract into components for easier assertions
        rr = output[1][:, :, 0, 0]
        rc = output[1][:, :, 1, 0]
        cc = output[1][:, :, 1, 1]

        # Make statements about the empty locations of the simple hessian
        for full_row in [0, 1, 3, 4]:
            assert (rr[full_row, :] == 0).all()
        for full_x in [0, 2, 4]:
            assert (rc[full_x, :] == 0).all()
            assert (rc[:, full_x] == 0).all()
        for full_column in [0, 1, 3, 4]:
            assert (cc[:, full_column] == 0).all()

        # I don't have a strong reason for why the hessian takes a certain
        # value, so I'll just be reasoning on the maximum value without making
        # statements about its actual value.
        value = numpy.max(rr)
        assert numpy.allclose(rr[2, :], numpy.array([value, 0, -value, 0, value]))
        assert numpy.allclose(rc[1, :], numpy.array([0, value / 2, 0, -value / 2, 0]))
        assert numpy.allclose(rc[3, :], numpy.array([0, -value / 2, 0, value / 2, 0]))
        assert numpy.allclose(cc[:, 2], numpy.array([value, 0, -value, 0, value]))

        # Here's an example of the simple hessian output
        # rr = [[ 0.   0.   0.   0.   0. ]
        #       [ 0.   0.   0.   0.   0. ]
        #       [ 0.5  0.  -0.5  0.   0.5]
        #       [ 0.   0.   0.   0.   0. ]
        #       [ 0.   0.   0.   0.   0. ]]
        # rc = [[ 0.    0.    0.    0.    0.  ]
        #       [ 0.    0.25  0.   -0.25  0.  ]
        #       [ 0.    0.    0.    0.    0.  ]
        #       [ 0.   -0.25  0.    0.25  0.  ]
        #       [ 0.    0.    0.    0.    0.  ]]
        # cc = [[ 0.   0.   0.5  0.   0. ]
        #       [ 0.   0.   0.   0.   0. ]
        #       [ 0.   0.  -0.5  0.   0. ]
        #       [ 0.   0.   0.   0.   0. ]
        #       [ 0.   0.   0.5  0.   0. ]]

    def test_edge(self):
        """
        When presented with an edge the hessian should have a very high ratio
        of principal curvatures. (Paper page 12)
        """

        # Make three types of lines, horizontal, vertical, and diagonal. In
        # addition, make the diagonal case a ridge, in case that turns out
        # differently.
        line0 = numpy.zeros((51, 51), dtype=float)
        line0[25:, :] = 1.0

        line1 = numpy.zeros((51, 51), dtype=float)
        line1[:, 25:] = 1.0

        line2 = numpy.diag(numpy.ones(51, dtype=float))

        for line in [line0, line1, line2]:
            output = hessian(line)
            # Take the hessian at the point of interest, in the middle
            ddx = output[25, 25]
            # 11^2 / 10 is a somewhat arbitrary number, see the paper for
            # why it is chosen. That is their basic chosen R (ratio) value.
            assert numpy.trace(ddx)**2 / numpy.linalg.det(ddx) > (11*2 / 10)
