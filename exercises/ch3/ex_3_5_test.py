import numpy
import pytest

from ex_3_5 import phi


class TestPhi:
    @pytest.mark.parametrize("len_s, p, N", (
        (10, 2, 9),
        (10, 9, 2),
        (5, 3, 3),
    ))
    def test_length_assertion_error(self, len_s, p, N):
        """When given s of insufficient length, raise."""
        s = [1] * len_s
        with pytest.raises(AssertionError):
            phi(s=s, p=p, N=N, i=0, k=0)

    @pytest.mark.parametrize("i, k", (
        ( 4,  0),  # invalid i
        (-1,  0),  # invalid i
        ( 0,  5),  # invalid k
        ( 1, -1),  # invalid k
        ( 4, -1),  # invalid both
    ))
    def test_value_valid_assertion_error(self, i, k):
        """Both i and k values need to be bounded by [0, p]"""
        s = [1] * 20
        with pytest.raises(AssertionError):
            phi(s=s, p=3, N=4, i=i, k=k)

    @pytest.mark.parametrize("s, i, k, expected", (
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 0, 54),  # 3*3 + 2*2 + 6*6 + 1 + 2*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 2, 0, 69),  # 9*3 + 5*2 + 3*6 + 2 + 6*2
        ([1, 9, 5, 3, 2, 6, 1, 2], 0, 2, 69),  # 3*9 + 2*5 + 6*3 + 2 + 2*6
        ([1, 9, 5, 3, 2, 6, 1, 2], 3, 1, 62),  # 1*5 + 9*3 + 5*2 + 3*6 + 2
    ))
    def test_simple_situations(self, s, i, k, expected):
        """Test some hand-verifiable situations with fixed p/N."""
        assert phi(s=s, p=3, N=5, i=i, k=k) == expected

    def test_correct_type(self):
        """Whether passed int or uint8 values, output should still be int."""

        def check(s):
            output = phi(s=s, p=3, N=5, i=0, k=0)
            assert output == 54
            assert isinstance(output, int)

        # Known working set
        s = [1, 9, 5, 3, 2, 6, 1, 2]
        check(s)

        # Try it again with s as an array of uint8
        check(numpy.array(s, dtype=numpy.uint8))
