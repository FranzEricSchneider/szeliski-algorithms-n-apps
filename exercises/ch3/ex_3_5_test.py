import numpy
import pytest

import ex_3_5


def test_instantiate_coefficients():
    """Run a simple history through and check the output."""

    # Make a series of 6 "images", shape (3, 3)
    history = [
        numpy.array([[1, 2, 3], [4, 5, 6], [2, 3, 4]]),
        numpy.array([[2, 3, 4], [5, 6, 7], [3, 2, 1]]),
        numpy.array([[3, 2, 1], [9, 5, 2], [4, 7, 7]]),
        numpy.array([[4, 7, 7], [3, 8, 3], [3, 5, 7]]),
        numpy.array([[3, 5, 7], [0, 1, 1], [6, 2, 4]]),
        numpy.array([[6, 2, 4], [1, 3, 5], [9, 5, 2]]),
    ]

    # Unfortunately I have no intuition for what the actual cooeficient values
    # should be, so instead I'm just testing shape and type...
    coefficients = ex_3_5.instantiate_coefficients(history, p=2, N=3)
    assert isinstance(coefficients, numpy.ndarray)
    assert coefficients.shape == (3, 3, 2)  # (3, 3, matches the incoming image
                                            # 2) matches p=2
    assert coefficients.dtype == float


def test_einsum():
    """Just make sure I'm using it right."""

    # So let's say we have 2 (3, 3) grayscale images
    image_0 = numpy.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    image_1 = numpy.array([
        [7, 1, 4],
        [8, 2, 5],
        [9, 3, 6],
    ])
    images = numpy.array([image_0, image_1])
    assert images.shape == (2, 3, 3)

    # And a set of coefficients (necessarily (3, 3, 2))
    coefficients = numpy.array([
        [[0.5, 1.1], [0.5, 1.1], [0.5, 1.1]],
        [[0.5, 1.1], [0.5, 1.1], [0.5, 1.1]],
        [[0.5, 1.1], [0.5, 1.1], [0.5, 1.1]],
    ])
    assert coefficients.shape == (3, 3, 2)

    output = numpy.einsum("kij,ijk->ij", images, coefficients)
    assert output.shape == (3, 3)
    assert numpy.allclose(output, numpy.array([
        [0.5 + 7.7,   1 + 1.1, 1.5 + 4.4],
        [  2 + 8.8, 2.5 + 2.2,   3 + 5.5],
        [3.5 + 9.9,   4 + 3.3, 4.5 + 6.6],
    ]))



# TODO: Add construct equation test about assertion errors
def test_construct_equations():
    """Run a simple version of eq. (18) by hand and check it."""
    matrix, vector = ex_3_5.construct_equations(
        s=numpy.array([1, 7, 4, 2, 6, 3, 1, 7, 4, 3]),
        p=3,
        N=4,
    )
    assert isinstance(matrix, numpy.ndarray)
    assert isinstance(vector, numpy.ndarray)
    assert matrix.dtype == numpy.int64
    assert vector.dtype == numpy.int64

    assert numpy.allclose(
        matrix,
        numpy.array([
            [16 + 4 + 36 + 9 , 28 + 8 + 12 + 18, 4 + 14 + 24 + 6],
            [28 + 8 + 12 + 18, 49 + 16 + 4 + 36, 7 + 28 + 8 + 12],
            [4 + 14 + 24 + 6 ,  7 + 28 + 8 + 12, 1 + 49 + 16 + 4]
        ])
    )
    assert numpy.allclose(
        vector,
        numpy.array([-(4*2 + 2*6 + 6*3 + 3*1),
                     -(7*2 + 4*6 + 2*3 + 6*1),
                     -(1*2 + 7*6 + 4*3 + 2*1)])
    )


class TestPhi:

    @pytest.mark.parametrize("s, i, k, expected", (
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 0, 54),  # 3*3 + 2*2 + 6*6 + 1 + 2*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 2, 0, 69),  # 9*3 + 5*2 + 3*6 + 2 + 6*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 2, 69),  # 3*9 + 2*5 + 6*3 + 2 + 2*6
        ([1, 9, 5, 3, 2, 6, 1, 2], 3, 1, 62),  # 1*5 + 9*3 + 5*2 + 3*6 + 2
    ))
    def test_simple_situations(self, s, i, k, expected):
        """Test some hand-verifiable situations with fixed p/N."""
        assert ex_3_5.phi(s=numpy.array(s), p=3, N=5, i=i, k=k) == expected

    def test_overlong_s(self):
        """
        When s is longer than necessary we want to use the FIRST p+N values,
        because we have constructed phi such that new values should appear at
        the beginning of the array.
        """

        # Expected = 1*3 + 2*4 + 3*5 + 4*6 = 50, which doesn't use the full s
        assert ex_3_5.phi(s=numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                          p=2,
                          N=4,
                          i=2,
                          k=0) == 50
