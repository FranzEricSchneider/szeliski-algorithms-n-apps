import argparse
import json
import random
import sys

import numpy
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QLabel, QRubberBand


"""
TODO
"""


# Always
numpy.set_printoptions(suppress=True)
# Window size in pixels (width, height)
WINDOW_SIZE = (1000, 500)
# Defined transformations and related keys
TRANSFORMATIONS = {
    # One step is 0.25 (m)
    Qt.Key_PageUp: lambda x: T([0, 0, 0.25 * x]),
    Qt.Key_PageDown: lambda x: T([0, 0, 0.25 * -x]),
    Qt.Key_Right: lambda x: T([0.25 * x, 0, 0]),
    Qt.Key_Left: lambda x: T([0.25 * -x, 0, 0]),
    Qt.Key_Up: lambda x: T([0, 0.25 * x, 0]),
    Qt.Key_Down: lambda x: T([0, 0.25 * -x, 0]),
    # One step is 0.5 (rad)
    Qt.Key_X: lambda x: Rx(0.1 * x),
    Qt.Key_W: lambda x: Rx(0.1 * -x),
    Qt.Key_D: lambda x: Ry(0.1 * x),
    Qt.Key_A: lambda x: Ry(0.1 * -x),
    Qt.Key_E: lambda x: Rz(0.1 * x),
    Qt.Key_Q: lambda x: Rz(0.1 * -x),
    # Just render
    Qt.Key_R: lambda x: numpy.eye(4),
}
# TODO: Make a more meaningful scene
# Show a fixed scene of lines and points, allow the camera to move
SCENE = [
    # points
    numpy.array([[1, 0, 1, 1.0]]),
    numpy.array([[0.5, 0, 1, 1.0]]),
    numpy.array([[0, 1, 1, 1.0]]),
    numpy.array([[0, 0.5, 1, 1.0]]),
    numpy.array([[0, 0, 1, 1.0]]),
    # lines
    numpy.array([[1, 0, 2, 1.0], [0, 1, 2, 1.0]]),
    numpy.array([[0, 0, 2, 1.0], [2, 2, 2, 1.0]]),
    # polygons
    numpy.array([[1, 0, 3, 1.0], [0, 1, 3, 1.0], [0, 0, 3, 1.0]]),
    numpy.array([[1, 0, 4, 1.0], [1, 1, 4, 1.0], [1, 2, 4, 1.0], [0, 1, 4, 1.0]]),
]
# TODO
# https://stackoverflow.com/questions/724219/how-to-convert-a-3d-point-into-2d-perspective-projection ?
# TODO: Look up relationship between focal length, FOV, and aspect ratio
FX = 500
FY = 500
# Camera matrix
K = numpy.array([
    [FX,  0, WINDOW_SIZE[0] / 2],
    [ 0, FY, WINDOW_SIZE[1] / 2],
    [ 0,  0,                  1],
])
# Radius for rendered points (pixels)
R = 10


def T(vector):
    matrix = numpy.eye(4)
    matrix[0:3, 3] = vector
    return matrix


def Rx(angle):
    matrix = numpy.eye(4)
    matrix[1, 1] = numpy.cos(angle)
    matrix[1, 2] = -numpy.sin(angle)
    matrix[2, 1] = numpy.sin(angle)
    matrix[2, 2] = numpy.cos(angle)
    return matrix


def Ry(angle):
    matrix = numpy.eye(4)
    matrix[0, 0] = numpy.cos(angle)
    matrix[2, 0] = -numpy.sin(angle)
    matrix[0, 2] = numpy.sin(angle)
    matrix[2, 2] = numpy.cos(angle)
    return matrix


def Rz(angle):
    matrix = numpy.eye(4)
    matrix[0, 0] = numpy.cos(angle)
    matrix[0, 1] = -numpy.sin(angle)
    matrix[1, 0] = numpy.sin(angle)
    matrix[1, 1] = numpy.cos(angle)
    return matrix


class Window(QLabel):
    """Window for rendering 3D lines and points in 2D."""
    def __init__(self, rectangles, parent=None):
        QLabel.__init__(self, parent)

        # Origin of the camera, and its inverse
        self.camera = T([0, 0, -2])
        self.camera_inv = numpy.linalg.inv(self.camera)

        # Help text
        print("Hit <Escape> to exit, TODO.")
        print("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("Exiting due to <Escape> key")
            self.close()
        elif event.key() in TRANSFORMATIONS.keys():
            transform = TRANSFORMATIONS[event.key()](1)
            self.camera = self.camera.dot(transform)
            self.camera_inv = numpy.linalg.inv(self.camera)
            self.render()

    def render(self, update=True):
        """Draws the current set of rectangles."""
        self.setPixmap(create_pixmap())
        painter = QPainter()
        painter.begin(self.pixmap())
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        # Go through the scene objects in rough order from the camera. Note
        # that this isn't a true representation of occlusion, where shapes
        # could really weave in and out of each other, but it's an
        # approximation that should look alright.
        for points in sorted(
                SCENE,
                reverse=True,  # largest distances come first
                key=lambda x: numpy.linalg.norm(
                    # camera origin       average position of the points
                    self.camera[0:3, 3] - numpy.mean(x[:, 0:3], axis=0)
                )):
            self.draw(points, painter)
        painter.end()
        if update:
            self.update()

    def draw(self, world_points, painter):
        # Get points in the camera frame in 3D
        cam_points = self.camera_inv.dot(world_points.T).T
        # Scale by the z value (makes w non-1)
        cam_points = numpy.einsum("ij,i->ij", cam_points, 1 / cam_points[:, 2])
        # Multiply by the camera matrix to get pixels, leaving out w
        pixels = K.dot(cam_points[:, 0:3].T).T.astype(int)

        # If it's 1D, it's a point:
        if pixels.shape[0] == 1:
            painter.drawEllipse(pixels[0][0], pixels[0][1], R, R)
        # If it's 2D, it's a line:
        elif pixels.shape[0] == 2:
            painter.drawLine(pixels[0][0], pixels[0][1], pixels[1][0], pixels[1][1])
        else:
            painter.drawPolygon(
                QPolygon([
                    QPoint(*corner.tolist()[:2]) for corner in pixels
                ])
            )


def create_pixmap():
    """Create white pixmap of window size."""
    pixmap = QPixmap(*WINDOW_SIZE)
    pixmap.fill(QColor(255, 255, 255))
    return pixmap


def closest(point, rectangles):
    """
    Arguments:
        point: numpy.array two-element integer point, in the same (x, y)
            coordinates as the window
        rectangles: list of the Rectangle class
    Returns: (None, None, None) if the rectangles are empty, otherwise a
        three-element tuple of
        (
            pixel distance [float],
            Rectangle class [with closest point],
            index of closest point within Rectangle
        )
    """
    if not rectangles:
        return None, None, None
    min_dist = 1e6
    chosen = index = None
    for rectangle in rectangles:
        dist, dist_index = rectangle.min_dist(point)
        if dist < min_dist:
            min_dist = dist
            chosen = rectangle
            index = dist_index
    return (min_dist, chosen, index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Draw rectangles.')
    parser.add_argument(
        'saved_rectangles',
        nargs='*',
        help='Space separated paths to saved rectangles, if any'
    )
    args = parser.parse_args()

    app = QApplication([])
    random.seed()

    window = Window([Rectangle.from_file(path)
                     for path in args.saved_rectangles])
    window.setPixmap(create_pixmap())
    window.resize(*WINDOW_SIZE)
    window.show()

    sys.exit(app.exec_())
