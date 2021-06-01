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


DEFORMATIONS = {
    Qt.Key_T: "Translation",
    Qt.Key_R: "Rigid",
    Qt.Key_S: "Similarity",
    Qt.Key_A: "Affine",
    Qt.Key_P: "Perspective",
}
WINDOW_SIZE = (1000, 500)


class Rectangle:
    """Class for defining, saving, and warping rectangles."""

    def __init__(self, x, y, width, height):
        self.original_position = numpy.array([
            [x, y],
            [x + width, y],
            [x, y + height],
            [x + width, y + height],
        ])
        self.position = self.original_position.copy()

    def __repr__(self):
        return str(self.original_position)

    @classmethod
    def from_file(cls, path):
        # TODO
        pass
        # return cls(**json.load(path))

    def as_qt(self):
        """As a Qt rectangle to be drawn by drawRect."""
        begin = QPoint(*self.position[0].tolist())
        destination = QPoint(*self.position[-1].tolist())
        return QRect(begin, destination).normalized()

    def min_dist(self, point):
        """Returns the distance to closest corner, and corner index."""
        return sorted([
            (numpy.linalg.norm(point - corner), index)
            for index, corner in enumerate(self.position)
        ])[0]




class Window(QLabel):
    """Window for drawing, saving, and rendering possibly warped rectangles."""
    def __init__(self, rectangles, parent=None):
        QLabel.__init__(self, parent)
        # Populated when a rubber band is active
        self.origin = QPoint()
        # Track whether or not the shift key is pressed
        self.shifted = False
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

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
        elif event.key() in DEFORMATIONS.keys():
            print(f"Changing deformation mode to {DEFORMATIONS[event.key()]}")

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
                print(closest(point, self.rectangles))
                # TODO: Use this to select and move points
                # TODO: Define transformations

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rubber_band.hide()
            if self.shifted:
                self.rectangles.append(Rectangle(self.rubber_band.x(),
                                                 self.rubber_band.y(),
                                                 self.rubber_band.width(),
                                                 self.rubber_band.height()))
                print(f"Captured rectangle: {self.rectangles[-1]}")
                self.render()

    def render(self, update=True):
        """Draws the current set of rectangles."""
        painter = QPainter()
        painter.begin(self.pixmap())
        for rectangle in self.rectangles:
            painter.drawRect(rectangle.as_qt())
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
