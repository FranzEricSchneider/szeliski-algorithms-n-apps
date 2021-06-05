import argparse
import json
import random
import sys

import numpy
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QLabel, QRubberBand


"""
Adapted from
https://wiki.python.org/moin/PyQt/Selecting%20a%20region%20of%20a%20widget
"""


# Always
numpy.set_printoptions(suppress=True)
# Defined transformations and related keys
TRANSFORMATIONS = {
    Qt.Key_T: "Translation",
    Qt.Key_R: "Rigid",
    Qt.Key_S: "Similarity",
    Qt.Key_A: "Affine",
    Qt.Key_P: "Perspective",
}
# Window size in pixels (width, height)
WINDOW_SIZE = (1000, 500)
# Pixel distance at which we should "snap" to the nearest corner and assume it
# was clicked
SELECT_THRESHOLD = 10


def T(vector):
    matrix = numpy.eye(3)
    matrix[0:2, 2] = vector
    return matrix


class Rectangle:
    """Class for defining, saving, and warping rectangles."""

    def __init__(self, x, y, width, height):
        # Store points augmented
        self.original_position = numpy.array([
            [x, y, 1],
            [x + width, y, 1],
            [x + width, y + height, 1],
            [x, y + height, 1],
        ])
        # Store and modify the transform
        self.transform = numpy.eye(3)

    @property
    def position(self):
        # You could simplify to original.dot(transform.T), but I think this is
        # more clear
        homogeneous = self.transform.dot(self.original_position.T).T
        normalized = homogeneous / homogeneous[:, -1].reshape((len(homogeneous), 1))
        return normalized.astype(int)

    def __repr__(self):
        return str(self.original_position)

    @classmethod
    def from_file(cls, path):
        # TODO
        pass
        # return cls(**json.load(path))

    def as_qt(self):
        """As a Qt polygon to be drawn by drawPolygon."""
        return QPolygon([
            QPoint(*corner.tolist()[:2]) for corner in self.position
        ])

    def min_dist(self, point):
        """Returns the distance to closest corner, and corner index."""
        return sorted([
            (numpy.linalg.norm(point - corner), index)
            for index, corner in enumerate(self.position[:, 0:2])
        ])[0]

    def adjust(self, start_corner, end_x, end_y, transform_type):
        start = self.position[start_corner]
        end = numpy.array([end_x, end_y, 1])
        vector = end - start
        new_transform = numpy.eye(3)

        if transform_type == "Translation":
            new_transform[0:2, 2] = vector[0:2]
        elif transform_type == "Rigid":
            new_transform = rigid_transform(new_transform, vector, end)
        elif transform_type == "Similarity":
            new_transform = similar_transform(new_transform, vector, end)
        elif transform_type == "Affine":
            new_transform = affine_transform(new_transform, vector)
        elif transform_type == "Perspective":
            new_transform = perspective_transform(new_transform, vector)

        # Finally, apply the new adjustment in a stacked fashion onto the
        # current transform
        self.transform = new_transform.dot(self.transform)


def rigid_transform(transform, vector, end):
    # This is undetermined, 2 DOF have been accounted for (dx, dy) and
    # there is a third left hanging. Instead of figuring out something
    # tricky I'll do something arbitrary to show that rigid transforms
    # work.
    # Calculate a somewhat derived R
    theta = numpy.arctan2(vector[1], vector[0])
    R = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                     [numpy.sin(theta),  numpy.cos(theta), 0],
                     [0, 0, 1]])
    # Start off by moving the rectangle from the starting point to the
    # end point, a.k.a. along the path of vector
    transform = T(vector[0:2]).dot(transform)
    return rotate_around_point(transform, end, R)


def rotate_around_point(transform, end, R):
    # Then we need to do a little dance to rotate around the chosen
    # point. Do that by 1) moving the corner to the origin (-end)
    transform = T(-end[0:2]).dot(transform)
    # Rotate it by R around the origin
    transform = R.dot(transform)
    # Then move it back to where end was
    return T(end[0:2]).dot(transform)


def similar_transform(transform, vector, end):
    # Calculate a somewhat derived R, same as rigid, but scaled
    theta = numpy.arctan2(vector[1], vector[0])
    R = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                     [numpy.sin(theta),  numpy.cos(theta), 0],
                     [0, 0, 1]])
    # Scale the rotation matrix by a scalar where we get bigger to the right,
    # smaller to the left
    scalar = 1 + float(vector[0]) / WINDOW_SIZE[0]
    R[0:2, 0:2] *= scalar
    # Start off by moving the rectangle from the starting point to the
    # end point, a.k.a. along the path of vector
    transform = T(vector[0:2]).dot(transform)
    return rotate_around_point(transform, end, R)


def affine_transform(transform, vector):
    # I am up a creek here. There are 6DOF in an affine transform, what would
    # be a reasonable way to set them? I'll just layer on a couple of things
    # based on a few variables.
    x_scalar = float(vector[0]) / WINDOW_SIZE[0]
    y_scalar = float(vector[1]) / WINDOW_SIZE[1]

    # Move a bit towards the mouse, why not
    transform = T(vector[0:2]).dot(transform)

    # This is totally arbitrary, should flip and skew the axes
    transform[0, 1] += x_scalar
    transform[1, 0] += y_scalar
    transform[0:2, 0] *= (1 + x_scalar)
    transform[0:2, 1] *= (1 + y_scalar)

    return transform


def perspective_transform(transform, vector):
    # Even further up the creek. There are 9DOF now, I'll just layer on a few
    # more things.
    x_scalar = float(vector[0]) / WINDOW_SIZE[0]
    y_scalar = float(vector[1]) / WINDOW_SIZE[1]

    # Move a bit towards the mouse, why not
    transform = T(vector[0:2]).dot(transform)

    # This is totally arbitrary, should flip and skew the axes
    transform[0, 1] += x_scalar
    transform[1, 0] += y_scalar
    transform[0:2, 0] *= (1 + x_scalar)
    transform[0:2, 1] *= (1 + y_scalar)
    # These last-row effects are very strong, so really minimize them (it'll
    # still show up)
    transform[2, :] += numpy.array([x_scalar,
                                    y_scalar,
                                    x_scalar * y_scalar]) * 0.01

    return transform


class Window(QLabel):
    """Window for drawing, saving, and rendering possibly warped rectangles."""
    def __init__(self, rectangles, parent=None):
        QLabel.__init__(self, parent)
        # Populated when a rubber band is active
        self.origin = QPoint()
        # Track whether or not the shift key is pressed
        self.shifted = False
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        # Track the current transformation style and activity
        self.transform_type = "Translation"
        self.transforming = None

        # Track the currently instantiated rectangles
        self.rectangles = rectangles

        # Help text
        print("Hit <Escape> to exit, <Tab> to save, t/r/s/a/p to change deformation mode.")
        print("Shift-click and drag to create a new rectangle")
        print("Click and drag a rectangle point to modify")
        print("")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self.shifted = True
        elif event.key() == Qt.Key_Escape:
            print("Exiting due to <Escape> key")
            self.close()
        elif event.key() == Qt.Key_Tab:
            print("Saving rectangles")
        elif event.key() in TRANSFORMATIONS.keys():
            self.transform_type = TRANSFORMATIONS[event.key()]
            print(f"Changing deformation mode to {self.transform_type}")

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Shift:
            self.shifted = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.shifted:
                self.origin = QPoint(event.pos())
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()
            else:
                point = numpy.array([event.pos().x(), event.pos().y()])
                dist, rectangle, corner = closest(point, self.rectangles)
                if dist is not None and dist < SELECT_THRESHOLD:
                    self.transforming = (rectangle, corner)

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # If a rubber band was created, deal with it now
            if not self.origin.isNull():
                # Wipe the active rubber band
                self.origin = QPoint()
                self.rubber_band.hide()
                # Only save if shift is still being held
                if self.shifted:
                    self.rectangles.append(Rectangle(self.rubber_band.x(),
                                                     self.rubber_band.y(),
                                                     self.rubber_band.width(),
                                                     self.rubber_band.height()))
                    print(f"Captured rectangle: {self.rectangles[-1]}")
                    self.render()
            # If a transformation is active, complete it
            if self.transforming:
                # Give the rectangle the starting corner, final mouse position,
                # and the current transform type, then let it work the
                # transform change out
                rectangle, corner = self.transforming
                rectangle.adjust(corner, event.pos().x(), event.pos().y(),
                                 self.transform_type)
                # And reset it
                self.transforming = None
                # Display any changes
                self.render()

    def render(self, update=True):
        """Draws the current set of rectangles."""
        self.setPixmap(create_pixmap())
        painter = QPainter()
        painter.begin(self.pixmap())
        for rectangle in self.rectangles:
            painter.drawPolygon(rectangle.as_qt())
        painter.end()
        if update:
            self.update()


def create_pixmap():
    """Create white pixmap of window size."""
    pixmap = QPixmap(*WINDOW_SIZE)
    pixmap.fill(QColor(255, 255, 255))
    painter = QPainter()
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
