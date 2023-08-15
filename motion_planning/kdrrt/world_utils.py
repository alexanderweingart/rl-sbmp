import random
class Point:
    x: float
    y: float

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point({self.x},{self.y})"

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

class Rectangle:
    center_x: float
    center_y: float
    height: float
    width: float

    def __init__(self, center_x, center_y, height, width):
        self.center_x = center_x
        self.center_y = center_y
        self.height = height
        self. width = width

    def __repr__(self) -> str:
        return f"Rectangle(center (x,y): ({self.center_x},{self.center_y}) h: {self.height} w: {self.width})"

class Workspace:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def __repr__(self) -> str:
        return f"Workspace(x_min: {self.x_min}, x_max: {self.x_max}, y_min: {self.y_min}, y_max: {self.y_max})"

    def includes(self, x, y) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


    def sample_pos_uniform(self):
        """ uniformly sample a position in the workspace """
        return Point(
            x=random.uniform(self.x_min, self.x_max),
            y=random.uniform(self.y_min, self.y_max),
        )