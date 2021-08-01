import numpy

from ex_3_2 import bilinear_interpolate, scale_to_uint8


def test_bilinear_interpolate():
    raw_image = numpy.array(
        [
            [ 0, 15, 10, 30, 30, 90, 20, 30],
            [20, 10,  5, 20, 22, 30, 70, 40],
            [ 6, 15, 16, 30, 26, 90, 32, 40],
            [30, 22, 15,  4, 33, 50, 60, 44],
            [40, 15, 20, 30, 30, 90, 10, 50],
            [40, 34, 25,  8, 44, 70, 50, 48],
            [90, 15, 40, 30, 60, 90, 80, 60],
            [50, 46, 35, 12, 55, 90, 40, 52],
        ],
        dtype=numpy.uint16
    )
    image = bilinear_interpolate(raw_image)

    # Test size and type
    assert image.shape == (8, 8, 3)
    assert image.dtype == float

    # Test values
    # NOTE: This was a mistake to make by hand :) don't do that again
    expected = numpy.array([
        [[10, 17.5, 0], [10,  15,     5], [15, 13.75, 10], [20,   30,   20], [25, 41,    30], [30,  90,     25], [35, 65,   20], [40, 30,   20]],
        [[10, 20,   3], [10,  13.75,  8], [15,  5,    13], [20,21.75, 20.5], [25, 22,    28], [30,  68,     27], [35, 70,   26], [40, 52.5, 26]],
        [[16, 20,   6], [16,  15,    11], [14, 16.25, 16], [12,  30,    21], [26, 43.75, 26], [40,  90,     29], [41, 65,   32], [42, 40,   32]],
        [[22, 30,  23], [22,18.75, 20.5], [13, 15,    18], [4,   27,    23], [27, 33,    28], [50, 68.25, 24.5], [47, 60,   21], [44, 52.5, 21]],
        [[28, 25,  40], [28,  15,    30], [17, 21.25, 20], [6,   30,    25], [33, 49.25, 30], [60,  90,     20], [53, 62.5, 10], [46, 50,   10]],
        [[34, 40,  65], [34,23.75, 47.5], [21, 25,    30], [8, 32.25, 37.5], [39, 44,    45], [70,  68.5,   45], [59, 50,   45], [48, 52.5, 45]],
        [[40, 30,  90], [40,  15,    65], [25, 26.25, 40], [10,  30,    50], [45, 54.75, 60], [80,  90,     70], [65, 60,   80], [50, 60,   80]],
        [[46, 50,  90], [46,  28.75, 65], [29, 35,    40], [12, 37.5,   50], [51, 55,    60], [90,  68.75,  70], [71, 40,   80], [52, 50,   80]],
    ])

    assert numpy.allclose(image, expected)


def test_scale_to_uint8():
    # Make a stupid simple image where the max scales nicely to 255
    image = numpy.array([
        [1, 10, 25.5],
        [12, 3, 1],
    ], dtype=float)

    # Try with a fraction of 1.0
    scaled = scale_to_uint8(image, fraction=1.0)
    assert scaled.dtype == numpy.uint8
    assert numpy.allclose(scaled, numpy.array([
        [10, 100, 255],
        [120, 30, 10],
    ]))

    # Scale things to twice that level, capped at 255
    scaled = scale_to_uint8(image, fraction=0.5)
    assert numpy.allclose(scaled, numpy.array([
        [20, 200, 255],
        [240, 60, 20],
    ]))
