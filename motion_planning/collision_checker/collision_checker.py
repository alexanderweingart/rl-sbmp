from collision import Circle, Vector, Poly
from collision import collide, Response
import yaml
import logging
from typing import List, Union


class CarCollisionChecker2D:
    """
    Collision Checker for a rectangular car in a map of rectangle obstacles
    """
    obstacles: List[Poly]

    def __init__(self, car_length: float, car_width: float, inflation: float = 1):
        self.obstacles = []
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.INFO)
        self.inflation = inflation  # factor used to inflate ( > 1) or deflate ( < 1) the size of the obstacles
        self.car_length = car_length
        self.car_width = car_width
        self.car_basis_vertices = self.get_car_basis_vertices()

    @classmethod
    def get_axis_aligned_rectangle_vertices(cls, length, width):
        """
        Returns the vertices of an axis aligned rectangle (theta = 0)
        @param length
        @param width
        @return: list of Vectors representing the vertices
                 of the rectangle (clockwise, from bottom left to bottom right)
        """
        l_half = length / 2
        w_half = width / 2
        return [
            Vector(-w_half, -l_half),
            Vector(-w_half, +l_half),
            Vector(+w_half, +l_half),
            Vector(+w_half, -l_half)
        ]

    @classmethod
    def get_axis_aligned_rectangle(cls, x, y, length, width) -> Poly:
        """
        Returns the Poly object representation of an axis aligned rectangle
        @param x
        @param y
        @param length
        @param width
        @return: Poly object representation of an axis aligned rectangle at pos (x,y)
        """
        vertices = cls.get_axis_aligned_rectangle_vertices(length, width)
        poly = Poly(Vector(x, y), vertices)
        return poly

    def get_car_basis_vertices(self) -> List[Vector]:
        """
        Helper method to construct a list of collision module Vectors that specify the basic car
        polygon at the origin and theta = 0
        @return: List of 2D Vectors representing the cars vertices
        """
        car_centered_mid = Poly.from_box(Vector(0,0), self.car_length, self.car_width).points
        car_centered_rear = [vec + Vector(self.car_length/2, 0) for vec in car_centered_mid]
        return car_centered_rear

    def add_poly_obstacle(self, obstacle: Union[Poly, Circle]) -> None:
        """
        add a new obstacle (Poly object) to the obstacle pool
        @param obstacle: new obstacle to add
        @return: None
        """
        self.obstacles.append(obstacle)

    def add_axis_aligned_rectangle(self, x, y, length, width) -> None:
        """
        Add new axis aligned obstacle to the obstacle pool
        @param x
        @param y
        @param length
        @param width
        @return: None
        """
        poly = self.get_axis_aligned_rectangle(x, y, length, width)
        self.add_poly_obstacle(poly)

    def collides(self, x, y, theta) -> bool:
        """
        Test if the given car pose collides with the obstacles in the pool
        @param x
        @param y
        @param theta
        @return: True if there is a collision, False otherwise
        """
        _rotated_vectors = [vec.rotate(theta) for vec in self.get_car_basis_vertices()]
        # _translated_vectors = [vec + Vector(x, y) for vec in _rotated_vectors]

        obj_poly = Poly(Vector(x, y), _rotated_vectors)  # this also does the translation

        for obs in self.obstacles:
            if collide(obj_poly, obs):
                return True

        return False

    def get_car_poly(self, x, y, theta):
        _rotated_vectors = [vec.rotate(theta) for vec in self.get_car_basis_vertices()]

        obj_poly = Poly(Vector(x, y), _rotated_vectors)  # this also does the translation
        return obj_poly


    def load_obstacles(self, path) -> None:
        """
        Helper method to load obstacles into the pool, base on a .yaml map configuration
        @param path: path to the map
        @return: None
        """
        config = yaml.safe_load(open(path, "r"))
        if 'map' not in config:
            self.logger.warning(f"loaded config <{config}> does not include 'map' keyword")
            raise ValueError

        map_config = config['map']
        if 'obstacles' not in map_config:
            self.logger.warning(f"map loaded from {path} does not define obstacles")
            return

        for obstacle in map_config['obstacles']:
            if obstacle is None:
                continue
            obstacle_type = obstacle['type']
            if obstacle_type == 'box':
                x, y = obstacle['center']
                w, l = obstacle['size']
                w *= self.inflation
                l *= self.inflation
                self.add_axis_aligned_rectangle(x, y, l, w)
            else:
                self.logger.warning(f"type {obstacle_type} is not implemented yet")
                raise NotImplementedError
