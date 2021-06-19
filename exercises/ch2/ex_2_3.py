import argparse
import json
import random
import sys

import numpy
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QLabel, QRubberBand


"""
Allows the camera to maneuver around a 3D scene
"""


# Always
numpy.set_printoptions(suppress=True)
# Window size in pixels (width, height)
WINDOW_SIZE = (1000, 500)
# Defined transformations and related keys
TRANSFORMATIONS = {
    # One step is 0.25 (m)
    Qt.Key_Right:    lambda x: T([0.25 *  x, 0, 0]),
    Qt.Key_Left:     lambda x: T([0.25 * -x, 0, 0]),
    Qt.Key_Down:     lambda x: T([0, 0.25 *  x, 0]),
    Qt.Key_Up:       lambda x: T([0, 0.25 * -x, 0]),
    Qt.Key_PageUp:   lambda x: T([0, 0, 0.25 *  x]),
    Qt.Key_PageDown: lambda x: T([0, 0, 0.25 * -x]),
    # One step is 0.1 (rad)
    Qt.Key_X: lambda x: Rx(0.1 *  x),
    Qt.Key_W: lambda x: Rx(0.1 * -x),
    Qt.Key_D: lambda x: Ry(0.1 *  x),
    Qt.Key_A: lambda x: Ry(0.1 * -x),
    Qt.Key_E: lambda x: Rz(0.1 *  x),
    Qt.Key_Q: lambda x: Rz(0.1 * -x),
    # Just render
    Qt.Key_R: lambda x: numpy.eye(4),
}
# Show a fixed scene of lines and points, allow the camera to move
SCENE = [
    # pyramid of points
    numpy.array([[-1, 1, 0, 1.0]]),
    numpy.array([[-2, 1, 0, 1.0]]),
    numpy.array([[-2, 2, 0, 1.0]]),
    numpy.array([[-1, 2, 0, 1.0]]),
    numpy.array([[-1.5, 1, 0, 1.0]]),
    numpy.array([[-2, 1.5, 0, 1.0]]),
    numpy.array([[-1.5, 2, 0, 1.0]]),
    numpy.array([[-1, 1.5, 0, 1.0]]),
    numpy.array([[-1.25, 1.25, 0.5, 1.0]]),
    numpy.array([[-1.75, 1.75, 0.5, 1.0]]),
    numpy.array([[-1.25, 1.75, 0.5, 1.0]]),
    numpy.array([[-1.75, 1.25, 0.5, 1.0]]),
    numpy.array([[-1.5, 1.5, 1, 1.0]]),
    numpy.array([[-1, 1, 0, 1.0], [-2, 1, 0, 1.0]]),
    numpy.array([[-2, 1, 0, 1.0], [-2, 2, 0, 1.0]]),
    numpy.array([[-2, 2, 0, 1.0], [-1, 2, 0, 1.0]]),
    numpy.array([[-1, 2, 0, 1.0], [-1, 1, 0, 1.0]]),
    numpy.array([[-1, 1, 0, 1.0], [-1.5, 1.5, 1, 1.0]]),
    numpy.array([[-2, 1, 0, 1.0], [-1.5, 1.5, 1, 1.0]]),
    numpy.array([[-2, 2, 0, 1.0], [-1.5, 1.5, 1, 1.0]]),
    numpy.array([[-1, 2, 0, 1.0], [-1.5, 1.5, 1, 1.0]]),
    # building of polygons
    numpy.array([[1, -1, 0, 1.0], [2, -1, 0, 1.0], [2, -1, 0.5, 1.0], [1, -1, 0.5, 1.0]]),
    numpy.array([[1, -2, 0, 1.0], [2, -2, 0, 1.0], [2, -2, 0.5, 1.0], [1, -2, 0.5, 1.0]]),
    numpy.array([[1, -1, 0, 1.0], [1, -2, 0, 1.0], [1, -2, 0.5, 1.0], [1, -1, 0.5, 1.0]]),
    numpy.array([[2, -1, 0, 1.0], [2, -2, 0, 1.0], [2, -2, 0.5, 1.0], [2, -1, 0.5, 1.0]]),
    numpy.array([[2, -2, 0.5, 1.0], [2, -1, 0.5, 1.0], [1, -1, 0.5, 1.0], [1, -2, 0.5, 1.0]]),
    numpy.array([[1.75, -1.75, 0.5, 1.0], [1.75, -1.25, 0.5, 1.0], [1.5, -1.5, 0.8, 1.0]]),
    numpy.array([[1.75, -1.75, 0.5, 1.0], [1.25, -1.75, 0.5, 1.0], [1.5, -1.5, 0.8, 1.0]]),
    numpy.array([[1.25, -1.25, 0.5, 1.0], [1.25, -1.75, 0.5, 1.0], [1.5, -1.5, 0.8, 1.0]]),
    numpy.array([[1.75, -1.25, 0.5, 1.0], [1.25, -1.25, 0.5, 1.0], [1.5, -1.5, 0.8, 1.0]]),
    # road made of lines
    numpy.array([[-1.5, 1, 0, 1.0], [-1.5, -1, 0, 1.0]]),
    numpy.array([[-1.5, 0, 0, 1.0], [1.5, 0, 0, 1.0]]),
    numpy.array([[1.5, 0, 0, 1.0], [1.5, -1, 0, 1.0]]),
    # disconnected set of triangles
    numpy.array([[-1.5, -1, 0, 1.0], [-2.25, -2, 0, 1.0], [-0.75, -1.75, 0, 1.0]]),
    numpy.array([[-0.75, -1.25, 0.5, 1.0], [-2.25, -1.25, 0.5, 1.0], [-1.5, -2, 0.5, 1.0]]),
]
# Create the camera matrix to translate camera-frame 3D stuff into pixels
# See page 56 for camera matrix stuff
# https://stackoverflow.com/questions/724219/how-to-convert-a-3d-point-into-2d-perspective-projection ?
# There's some relationship between focal length, FOV, and aspect ratio that I
# won't bother digging into
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


# Helper geometric functions
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
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)

        # Origin of the camera, and its inverse
        self.camera = T([0, 0, 10]).dot(Rx(numpy.pi))
        self.camera_inv = numpy.linalg.inv(self.camera)

        # Help text
        print("Hit <Escape> to exit, and R to start the render.")
        print("Page up/down zooms in and out")
        print("Arrow keys move camera")
        print("W/X, Q/E, A/D rotate the camera around its axes")
        print("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            print("Exiting due to <Escape> key")
            self.close()
        elif event.key() in TRANSFORMATIONS.keys():
            transform = TRANSFORMATIONS[event.key()](1)
            # Update the camera, the inverse, and render
            self.camera = self.camera.dot(transform)
            self.camera_inv = numpy.linalg.inv(self.camera)
            self.render()

    def render(self, update=True):
        """Draw points in the scene, sorted by distance."""
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
                    # camera origin       average position of the collection
                    self.camera[0:3, 3] - numpy.mean(x[:, 0:3], axis=0)
                )):
            self.draw(points, painter)
        painter.end()
        if update:
            self.update()

    # Amazing. I love it. This does render things in the negative distance,
    # upside down, but I like it too much to get rid of it :)
    def draw(self, world_points, painter):
        # Get points in the camera frame in 3D
        cam_points = self.camera_inv.dot(world_points.T).T
        # Scale by the z value (makes w non-1)
        cam_points = numpy.einsum("ij,i->ij", cam_points, 1 / cam_points[:, 2])
        # Multiply by the camera matrix to get pixels, leaving out w
        pixels = K.dot(cam_points[:, 0:3].T).T.astype(int)

        # If it's 1D, it's a point:
        if pixels.shape[0] == 1:
            painter.drawEllipse(pixels[0][0] - int(R/2.0), pixels[0][1] - int(R/2.0), R, R)
        # If it's 2D, it's a line:
        elif pixels.shape[0] == 2:
            painter.drawLine(pixels[0][0], pixels[0][1], pixels[1][0], pixels[1][1])
        # If there's a larger collection of points, polygon time
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


if __name__ == "__main__":
    app = QApplication([])
    random.seed()

    window = Window()
    window.setPixmap(create_pixmap())
    window.resize(*WINDOW_SIZE)
    window.show()

    sys.exit(app.exec_())
