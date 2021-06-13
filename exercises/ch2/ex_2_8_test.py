from unittest import mock
import numpy
import pytest

from ex_2_8 import (
    detect_skin,
    flatten_nonzero,
    from_masks,
    pixel_chromaticity,
)


@mock.patch("ex_2_8.pixel_chromaticity")
@mock.patch("cv2.imread")
@mock.patch("ex_2_8.get_unprocessed")
def test_from_masks(get_mock, imread_mock, chromaticity_mock):

    # These are the images we want to operate on (2, 2, 2)
    def images():
        return [
            numpy.array([
                [[0.5, 0.5], [0.25, 0.7]],
                [[0.1, 0.9], [0.9, 0.9]],
            ]),
            numpy.array([
                [[1.5, 1.5], [1.25, 1.7]],
                [[1.1, 1.9], [1.9, 1.9]],
            ]),
        ]
    chromaticity_mock.side_effect = images()

    # These paths will do nothing, just check they get used
    paths = ["A", "B"]
    # Build the masks to return certain image pixels
    masks = [
        numpy.array([
            [True, False],
            [False, True],
        ]),
        numpy.array([
            [True, False],
            [True, False],
        ]),
    ]

    # First check the invert=False case
    samples = from_masks(paths=paths, masks=masks, invert=False)

    # Check that the paths are used
    assert get_mock.call_args_list == [mock.call("A"), mock.call("B")]
    # And that the samples are right
    assert numpy.all(samples == numpy.array([
        [0.5, 0.5],
        [0.9, 0.9],
        [1.5, 1.5],
        [1.1, 1.9],
    ]))

    # Check again for invert=True
    chromaticity_mock.side_effect = images()
    samples = from_masks(paths=paths, masks=masks, invert=True)
    assert numpy.all(samples == numpy.array([
        [0.25, 0.7],
        [0.1, 0.9],
        [1.25, 1.7],
        [1.9, 1.9],
    ]))


@pytest.fixture
def image():
    """Makes a (2, 2, 3) color image, dtype uint8."""
    return numpy.array([
        [
            [0, 0, 0],
            [10, 11, 12]
        ],
        [
            [50, 55, 60],
            [100, 150, 200]
        ],
    ]).astype(numpy.uint8)


class TestPixelChromaticity:

    @pytest.mark.parametrize("image", (
        # Not color images
        numpy.ones((5, 5)).astype(numpy.uint8),
        numpy.ones((5, 5, 1)).astype(numpy.uint8),
        # Wrong dtype (float, not uint8)
        numpy.ones((5, 5, 3)),
    ))
    def test_failures(self, image):
        with pytest.raises(AssertionError):
            pixel_chromaticity(image)

    def test_values(self, image):
        """Check that the value mapping operates as expected."""

        # XYZ values corresponding to the image fixture
        XYZ = numpy.array([
            [
                [0, 0, 0],
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
        assert xy.dtype == float

        # Check the chromaticity calculation
        for i in range(2):
            for j in range(2):
                if numpy.sum(XYZ[i, j]) == 0.0:
                    # Check that the chromaticity of full black pixels is
                    # marked as 0 instead of nan
                    assert xy[i, j, 0] == 0.0
                    assert xy[i, j, 1] == 0.0
                else:
                    assert numpy.isclose(xy[i, j, 0],
                                         XYZ[i, j, 0] / numpy.sum(XYZ[i, j]))
                    assert numpy.isclose(xy[i, j, 1],
                                         XYZ[i, j, 1] / numpy.sum(XYZ[i, j]))


def test_flatten_nonzero():
    # Make some appropriate fake images. Full-zero pixels should be left out
    # in the end, pixels where one or more element is non-zero should be
    # included
    images = [
        numpy.array([
            [[0.0, 0.0], [0.0, 1.0], [0.5, 1.0]],
            [[0.2, 0.8], [0.9, 0.0], [0.0, 0.0]],
        ]),
        numpy.array([
            [[2.0, 0.0], [0.0, 0.0], [0.5, 0.0], [1.5, 0.4]],
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.7], [0.0, 0.0]],
            [[0.0, 0.0], [1.1, 0.3], [0.0, 0.0], [0.0, 0.2]],
        ]),
    ]
    samples, masks = flatten_nonzero(images)
    assert numpy.all(samples == numpy.array([
        [0.0, 1.0],
        [0.5, 1.0],
        [0.2, 0.8],
        [0.9, 0.0],
        [2.0, 0.0],
        [0.5, 0.0],
        [1.5, 0.4],
        [0.1, 0.0],
        [0.0, 0.7],
        [1.1, 0.3],
        [0.0, 0.2],
    ]))
    assert len(masks) == 2
    assert numpy.all(masks[0] == numpy.array([
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ))
    assert numpy.all(masks[1] == numpy.array([
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    ))


@mock.patch("ex_2_8.pixel_chromaticity", lambda x: x)
def test_detect_skin():
    image = numpy.array([
        [  # clearly in    clearly out
            [0.75, 0.75], [0.50, 0.75], [0.75, 0.50],
        ],
        [  # in on one axis, out on the other
            [0.90, 0.90], [0.90, 0.60], [0.60, 0.90],
        ],
        [  # in on the main axes, then out
            [0.95, 0.75], [0.75, 0.55], [0.75, 0.52],
        ],
    ])

    mean = numpy.array([0.75, 0.75])

    # Make up some eigenvector axes that are simple to reason about but off
    # axis. Remember, vectors[:, i] should be an eigenvector. Eigenvectors are
    # normalized to length 1
    axes = numpy.array([
        [1,  1],
        [1, -1],
    ], dtype=float)
    axes[:, 0] /= numpy.linalg.norm(axes[:, 0])
    axes[:, 1] /= numpy.linalg.norm(axes[:, 1])
    # These values are the "bounds" within which points should fall. The sqrt(2)
    # is so we can reason about diagonals, and the 1e-6 is to get edge points in
    values = numpy.array([0.1, 0.2]) * numpy.sqrt(2) + 1e-6

    # Detect and check
    skin_mask = detect_skin(image=image, mean=mean, axes=axes, stddev_values=values)

    # Type checks
    assert skin_mask.shape == (3, 3)
    assert skin_mask.dtype == bool

    assert numpy.all(skin_mask == numpy.array([
        [True, False, False],
        [False, True, True],
        [True, True, False],
    ]))