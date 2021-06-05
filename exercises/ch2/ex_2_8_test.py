import numpy
import pytest

from ex_2_8 import pixel_chromaticity


@pytest.fixture
def image():
    """Makes a (2, 2, 3) color image, dtype uint8."""
    return numpy.array([
        [
            [1, 2, 3],
            [10, 11, 12]
        ],
        [
            [50, 55, 60],
            [100, 150, 200]
        ],
    ]).astype(numpy.uint8)


class TestPixelChromaticity:

    @pytest.mark.parametrize("image,mask", (
        # Not color images
        (numpy.ones((5, 5)).astype(numpy.uint8), None),
        (numpy.ones((5, 5, 1)).astype(numpy.uint8), None),
        # Wrong dtype (float, not uint8)
        (numpy.ones((5, 5, 3)), None),
        # Mask size mismatches
        (
            numpy.ones((5, 5, 3)).astype(numpy.uint8),
            numpy.ones((6, 6)).astype(bool),
        ),
        (
            numpy.ones((5, 5, 3)).astype(numpy.uint8),
            numpy.ones((5, 5, 1)).astype(bool),
        ),
        # Wrong mask dtype (float, not bool)
        (
            numpy.ones((5, 5, 3)).astype(numpy.uint8),
            numpy.ones((5, 5)),
        ),
    ))
    def test_failures(self, image, mask):
        with pytest.raises(AssertionError):
            pixel_chromaticity(image, mask)

    def test_values(self, image):
        """Check that the value mapping operates as expected."""

        # XYZ values corresponding to the image fixture
        XYZ = numpy.array([
            [
                [1.72, 1.83366, 2.99],
                [10.72, 10.83366, 11.99],
            ],
            [
                [53.6, 54.1683, 59.95],
                [136., 141.683, 199.5],
            ],
        ])

        xy = pixel_chromaticity(image)

        # Check basic type assertions
        assert xy.shape == (2, 2, 2)
        assert xy.dtype == numpy.uint8

        # Check the chromaticity calculation
        for i in range(2):
            for j in range(2):
                assert xy[i, j, 0] == int(XYZ[i, j, 0] / numpy.sum(XYZ[i, j]))
                assert xy[i, j, 1] == int(XYZ[i, j, 1] / numpy.sum(XYZ[i, j]))

    # TODO: Test masks
