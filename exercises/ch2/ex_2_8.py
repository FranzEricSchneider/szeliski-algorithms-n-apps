import cv2
from glob import glob

import numpy


# Always
numpy.set_printoptions(suppress=True)


# Eq 2.104 on page 88
RGB_to_XYZ = numpy.array([
    [0.49,    0.3,    0.21   ],
    [0.17697, 0.8124, 0.01063],
    [0,       0.01,   0.99   ],
])

def pixel_chromaticity(image, mask=None):
    """
    Arguments:
        images: (n, m, 3) numpy array, dtype uint8. Color image of the scene.
        mask: (n, m) numpy array, dtype bool, or None. If bool, indicates the
            pixels we want to calculate chromaticity for.

    Returns: (n, m, 2) numpy array, dtype uint8. (x, y) chromaticity values,
        which were first transferred to (X, Y, Z) space, then normalized to get
        chromaticity coordinates.
    """
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    assert image.dtype == numpy.uint8
    if mask is not None:
        assert image.shape[0:2] == mask.shape
        assert mask.dtype == bool

    # Get XYZ coordinates by applying RGB_to_XYZ matrix to each pixel
    # numpy.einsum is quite complicated and seems quite useful, as a reminder:
    # https://ajcr.net/Basic-guide-to-einsum/
    # https://stackoverflow.com/questions/25922212/element-wise-matrix-multiplication-in-numpy
    XYZ = numpy.einsum("ij,nmj->nmi", RGB_to_XYZ, image)
    # Normalize into xyz coordinates
    xyz = numpy.einsum("ijk,ij->ijk", XYZ, 1 / numpy.sum(XYZ, axis=2))

    # It's not clear to me why chromaticity is xy instead of xyz, but hey
    return xyz[:, :, :2].astype(numpy.uint8)


if __name__ == "__main__":

    image_paths = glob("ex_2_8_images/PROCESSED/*")
    images = [cv2.imread(path) for path in image_paths]
