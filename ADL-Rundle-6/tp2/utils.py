# Boxes
class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point({self.x}, {self.y})"


class BoundingBox:
    def __init__(self, top_left: Point, bot_right: Point) -> None:
        self.top_left = top_left
        self.bot_right = bot_right

    def __str__(self) -> str:
        return f"BoundingBox(Top Left: {self.top_left}, Bottom Right: {self.bot_right})"


class Track:
    def __init__(self, track_id: int, box: BoundingBox) -> None:
        self.track_id = track_id
        self.box = box
        self.missed = 0

    def update_box(self, box: BoundingBox) -> None:
        self.box = box
        self.missed = 0

    def miss(self) -> None:
        self.missed += 1

    def __str__(self) -> str:
        return f"Track(ID: {self.track_id}, Box: {self.box})"
