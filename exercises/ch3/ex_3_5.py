import numpy


"""
Working from section 2.1 here:
http://users.eecs.northwestern.edu/~yingwu/teaching/EECS432/Reading/Toyama_ICCV99.pdf

Which then directs you here:
https://www.commsp.ee.ic.ac.uk/~xl404/papers/Linear%20prediction%20A%20tutorial%20review.pdf
    Particularly II)C) on page 5, where it references eq. (18) on page 4
"""


def phi(s, p, N, i, k):
    """Calculates phi_{i,k} as from eq. (20) in the linear prediction primer.

    From eq. (20) we know that
    phi_{i,k} = sum {n from 0 to N-1} ( s[n-i] - s[n-k] )

    It's important to note that i and k can go from 0 to a value called p. So
    in the case where (n=0, i=p), then we have the case s[-p]. This does NOT
    mean the python thing where we wrap around to the beginning, instead the
    text says "the values from -p to N-1 must all be known: a total of p+N
    samples". To account for that, I'm going to take p as an input and then
    treat the [p] index as 0, like so:
        p = 5
        -5, -4, -3, -2, -1, 0  # We know s must be defined from -p to N-1
         0,  1,  2,  3,  4, 5  # So when we zero-index the array [p] hits our
                               # desired 0 point

    Arguments:
        s: A list of historical "output signal" values, which I am interpreting
            as the actually-seen pixel values. Which makes this a time series
            of uint8.
        p: (int) Number of steps that we need to know "in the future", see
            docstring equation discussion
        N: (int) Number of steps to look back in the past
        i: (int) Index offset for the first term
        k: (int) Index offset for the second term

    Raises: AssertionError if s is not long enough (fitness should be checked
        before this function) or if (i, k) are outside the valid [0, p] range

    Returns: (Int but NOT uint8) sum of a variety of products, as defined by
        the equation.
    """

    # As discussed in the docstring, this is a requirement
    assert len(s) >= p + N

    # The i, k values must be bounded by [0, p]
    assert 0 <= i <= p
    assert 0 <= k <= p

    # Then perform the sum with the offset to make p the "0" point in the array
    # as discussed. Because summed is an int += converts uint8 to full ints.
    summed = 0
    for n in range(p, N + p):
        summed += s[n-i] * s[n-k]

    # Standardize the return type
    return int(summed)