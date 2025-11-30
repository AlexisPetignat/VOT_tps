from KalmanFilter import KalmanFilter
import numpy as np


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def copy(self) -> "Point":
        return Point(self.x, self.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"


class BoundingBox:
    def __init__(self, top_left: Point, bot_right: Point) -> None:
        self.top_left = top_left
        self.bot_right = bot_right

    def center(self) -> Point:
        center_x = (self.top_left.x + self.bot_right.x) // 2
        center_y = (self.top_left.y + self.bot_right.y) // 2
        return Point(center_x, center_y)

    def move_center(self, new_center: Point) -> None:
        delta = new_center - self.center()
        self.top_left = self.top_left + delta
        self.bot_right = self.bot_right + delta

    def copy(self) -> "BoundingBox":
        return BoundingBox(self.top_left.copy(), self.bot_right.copy())

    def __str__(self) -> str:
        return f"BoundingBox(Top Left: {self.top_left}, Bottom Right: {self.bot_right})"


class Track:
    def __init__(self, track_id: int, box: BoundingBox) -> None:
        self.track_id = track_id
        self.box = box
        self.missed = 0
        self.total_matches = 0
        self.kf = KalmanFilter(
            dt=0.1,
            x=box.center().x,
            y=box.center().y,
            u_x=1,
            u_y=1,
            std_acc=1.0,
            x_std_meas=0.1,
            y_std_meas=0.1,
        )

    def update_box(self, box: BoundingBox) -> None:
        self.box = box
        self.missed = 0

    def predict_kalman(self) -> Point:
        pred_x, pred_y = self.kf.predict()
        return Point(int(pred_x), int(pred_y))

    def update_kalman(self, detection: Point) -> Point:
        detection_array = np.array([[detection.x], [detection.y]])
        meas_x, meas_y = self.kf.update(detection_array)
        return Point(int(meas_x), int(meas_y))

    def miss(self) -> None:
        self.missed += 1
        if self.missed == 1:
            self.total_matches += 1
        # else:
        #     self.total_matches = 0

    def __str__(self) -> str:
        return f"Track(ID: {self.track_id}, Box: {self.box})"
