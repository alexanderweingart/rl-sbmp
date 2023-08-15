import logging
import datetime
import os
import time
import numpy as np
import pygame
import pygame.math
from numpy import ndarray
from pygame import gfxdraw
from typing import Tuple, Optional
import yaml
from motion_planning.collision_checker import collision_checker
from rich.prompt import FloatPrompt
from motion_planning.kdrrt.visualize import Animation


class AckermannCarArtist:
    def __init__(self, car_length, render_mode, render_fps, trace_mode=False, x_max=1, screen_width=1500,
                 screen_height=1500, map_path: Optional[None] = None):
        self.logger = logging.getLogger(__file__)
        self.trace = []
        self.trace_starts = []
        self.trace_ends = []
        self.trace_mode = trace_mode
        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.car_length = car_length
        self.render_fps = render_fps
        map_config = {}
        self.obstacles = []
        self.x_max = x_max
        self.world_width = self.x_max * 2

        # init from map config (if given)
        if map_path is not None:
            map_config = yaml.safe_load(open(map_path, "r"))
        if "map" in map_config:
            if "dimensions" in map_config["map"]:
                (x_dim, y_dim) = map_config["map"]["dimensions"]
                self.world_width = max(x_dim, y_dim)
            if "obstacles" in map_config["map"]:
                self.obstacles = map_config["map"]["obstacles"]


        self.scale = self.screen_width / self.world_width

        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        else:
            raise NotImplementedError

        self.clock = pygame.time.Clock()

        self.car_coords_base, self.roof_coords_base = self.get_base_coords_car()

    def get_base_coords_car(self, old_version=False) -> Tuple[ndarray, ndarray]:
        """
        constructs the cart and roof-coord arrays based on the car's length and the scale
        @return: cart_coords, roof_coords
        """

        cart_length = self.car_length * self.scale
        cart_width = cart_length / 2
        roof_length = cart_length / 2
        roof_width = cart_width * 0.8
        roof_offset = -cart_length / 5

        roof_bottom_left = (-roof_length / 2 + roof_offset, -roof_width / 2)
        roof_top_left = (-roof_length / 2 + roof_offset, roof_width / 2)
        roof_top_right = (roof_length / 2 + roof_offset, roof_width / 2)
        roof_bottom_right = (roof_length / 2 + roof_offset, -roof_width / 2)

        cart_coords = np.array([(0 - cart_length / 2, -cart_width / 2), (0 - cart_length / 2, cart_width / 2),
                                (cart_length / 2, cart_width / 2), (cart_length / 2, -cart_width / 2)])

        roof_coords = np.array([roof_bottom_left, roof_top_left, roof_top_right, roof_bottom_right])

        if not old_version:
            # this version of the coordinate system is closer to an actual car
            cart_coords = cart_coords + np.array([cart_length/2, 0])
            roof_coords = roof_coords + np.array([cart_length/2, 0])

        return cart_coords, roof_coords

    def draw_car(self, surface, x, y, theta, color_car=(0xf7, 0xb7, 0x31),
                 color_roof=(0, 0, 0)):
        """
        Draws a car to the proviced surface
        @param surface: pygame surface the car should be drawn on
        @param x: the car's x coord
        @param y: the car's y coord
        @param theta: the car's orientation
        @param color_car: the car' color (default: new-york taxi yellow)
        @param color_roof: the color of the car's roof (default: black)
        @return: the car's coordinates (rotated and transposed)
        """
        cartx = x * self.scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = y * self.scale + self.screen_height / 2.0  # MIDDLE OF CART
        cart_coords = [pygame.math.Vector2((c[0], c[1])).rotate_rad(theta) + pygame.math.Vector2((cartx, carty)) for
                       c in self.car_coords_base]
        roof_coords = [pygame.math.Vector2((c[0], c[1])).rotate_rad(theta) + pygame.math.Vector2((cartx, carty)) for
                       c in self.roof_coords_base]

        gfxdraw.aapolygon(surface, cart_coords, color_roof)
        gfxdraw.filled_polygon(surface, cart_coords, color_car)

        gfxdraw.aapolygon(surface, roof_coords, color_roof)
        gfxdraw.filled_polygon(surface, roof_coords, color_roof)
        return cart_coords

    def draw_box_obstacle(self, surface, center_x, center_y, width, height, color=(0xC0, 0xC0, 0xC0)):
        # center_y = -center_y

        box_coords = [
            pygame.math.Vector2(center_x-width/2, center_y + height/2),
            pygame.math.Vector2(center_x+width / 2, center_y + height / 2),
            pygame.math.Vector2(center_x + width / 2, center_y - height / 2),
            pygame.math.Vector2(center_x - width / 2, center_y - height / 2)
        ]
        print(f"obstacle: {box_coords}")
        box_coords = [pygame.math.Vector2(vec[0], self.world_width - vec[1]) * self.scale for vec in box_coords]
        print(f"transformed: {box_coords}")
        gfxdraw.filled_polygon(surface, box_coords, color)

    def draw_all_obstacles(self, surface):
        for obs in self.obstacles:
            (center_x, center_y) = obs["center"]
            (width, height) = obs["size"]
            self.draw_box_obstacle(surface, center_x, center_y, width, height)

    def store_screenshot(self, target_dir: str, ):
        """
        Stores a screenshot of the current pygame window to the given directory. Creates a new directory if
        it does not exist.
        @param target_dir: path where the screenshots should be stored
        """
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        timestamp = time.time()
        pygame.image.save(self.screen, os.path.join(target_dir,
                                                    f"ackermann_environment_screenshot_{timestamp}.jpg"))

    def render_environment_state(self, x: float, y: float, theta: float, goal_tolerance_pos: float,
                                 goal_tolerance_theta: float, perfect_path=None, overlay_traces=False,
                                 screenshot_dir=None, first_state=False, last_state=False,
                                 last_x=None, last_y: Optional[float] = None) -> Optional[ndarray]:
        """
        @param goal_tolerance_theta: tolerance regarding theta (size of the goal region)
        @param goal_tolerance_pos: tolerance regarding pos (size of the goal region)
        @param x: car's x coord
        @param y: car's y coord
        @param theta: car's orientation
        @param perfect_path: list of the states on the perfect path. if provided they will be rendered as a reference overlay
        @param overlay_traces: if true, all runs will be rendered on top of each other
        @param screenshot_dir: if provided, a screenshot will be saved at the end of each rollout
        @param first_state: bool flag to signal that this is the first state of a rollout
        @param last_state: goal flag to signal that this is the last state of a rollout
        previous position (optional. if all are provided, the difference to the current state will be visualized by a line):
        @param last_x: car's previous x coord
        @param last_y: car's previous y coord
        @return: None or rgb array if the artist's render mode is set to rgb_array
        """

        if perfect_path is None:
            perfect_path = []

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))

        # > visualizing the goal region
        # >> orientation
        self.draw_car(surf, 0, 0, -goal_tolerance_theta,
                      color_car=(0xd3, 0xd3, 0xd3), color_roof=(0xa9, 0xa9, 0xa9))
        self.draw_car(surf, 0, 0, goal_tolerance_theta,
                      color_car=(0xd3, 0xd3, 0xd3), color_roof=(0xa9, 0xa9, 0xa9))
        self.draw_car(surf, 0, 0, 0, color_car=(0xd3, 0xd3, 0xd3),
                      color_roof=(0xa9, 0xa9, 0xa9))
        # >> position
        cord_target_x = self.screen_width / 2
        cord_target_y = self.screen_height / 2
        gfxdraw.circle(surf, int(cord_target_x), int(cord_target_y), int(goal_tolerance_pos * self.scale),
                       (10, 0, 0))

        if self.trace_mode:
            color_start = np.asarray([0x9f, 0xc5, 0xe8])
            color_end = np.asarray([0x21, 0x40, 0xff])
            step = (color_end - color_start) / 50
            
            for i, coords in enumerate(self.trace):
                gfxdraw.aapolygon(surf, coords, (0, 0, 0))
                # gfxdraw.filled_polygon(surf, coords, (0, 0, 0))
                gfxdraw.filled_polygon(surf, coords, np.clip(color_start + i*step, 0,0xff))

            # for coords in self.trace_starts:
            #     gfxdraw.aapolygon(surf, coords, (0, 255, 0))
            #     # gfxdraw.filled_polygon(surf, coords, (0, 255, 0))
            #     gfxdraw.filled_polygon(surf, coords, (178,0,0))
            #
            # for coords in self.trace_ends:
            #     gfxdraw.aapolygon(surf, coords, (255, 0, 0))
            #     gfxdraw.filled_polygon(surf, coords, (255, 0, 0))

            # current position
            current_pos_cart_coords \
                = self.draw_car(surf, x, y, theta)

            if first_state:
                if overlay_traces:
                    # safe the current car coordinates as a starting position
                    self.trace_starts.append(current_pos_cart_coords)
                    if len(self.trace) > 0:
                        self.trace_ends.append(self.trace[-1])
                else:
                    # start new trace (losing the last trajectory)
                    self.trace_starts = [current_pos_cart_coords]
                    self.trace = [current_pos_cart_coords]
            else:
                self.trace.append(current_pos_cart_coords)

        self.draw_car(surf, x, y, theta)

        for i in range(1, len(perfect_path)):
            # draw the perfect path, if one is provided
            x0, y0, _ = perfect_path[i - 1]
            x0 = x0 * self.scale + self.screen_width / 2.0
            y0 = y0 * self.scale + self.screen_width / 2.0
            x1, y1, _ = perfect_path[i]
            x1 = x1 * self.scale + self.screen_width / 2.0
            y1 = y1 * self.scale + self.screen_width / 2.0

            gfxdraw.line(surf, int(x0), int(y0), int(x1), int(y1), (192, 192, 192))
            i += 1

        if last_x is not None and last_y is not None:
            # visualize the pos change if the last pos is provided
            last_x_transformed = int(last_x * self.scale + self.screen_width / 2)
            last_y_transformed = int(last_y * self.scale + self.screen_height / 2)
            current_x_transformed = int(x * self.scale + self.screen_width / 2)
            current_y_transformed = int(y * self.scale + self.screen_height / 2)
            gfxdraw.line(surf, last_x_transformed, last_y_transformed,
                         int(current_x_transformed), int(current_y_transformed), (0, 100, 0))

        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        if screenshot_dir is not None:
            os.makedirs(screenshot_dir, exist_ok=True)
            pygame.image.save(surf, os.path.join(screenshot_dir, f"policy_rollout_sample_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"))

    def collision_checker_test(self, map_path: str, s_init=(2.5,2.5,0)):
        cc = collision_checker.CarCollisionChecker2D(self.car_length, self.car_length/2)
        cc.load_obstacles(map_path)
        surf = pygame.Surface((self.screen_width, self.screen_height))

        surf.fill((255, 255, 255))
        self.draw_all_obstacles(surf)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        (x,y,theta) = s_init

        step_size = 0.005
        step_size_theta = np.pi * 0.1
        stop_requested = False

        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_LEFT:
                        x -= step_size
                        if x < 0:
                            x = 0
                    if event.key == pygame.K_RIGHT:
                        x += step_size
                        if x > self.world_width:
                            x = self.world_width

                    if event.key == pygame.K_UP:
                        y += step_size
                        if y > self.world_width:
                            y = self.world_width
                    if event.key == pygame.K_DOWN:
                        y -= step_size
                        if y < 0:
                            y = 0
                    if event.key == pygame.K_SPACE:
                        theta += step_size_theta
                        theta %= np.pi*2

                    if event.key == pygame.K_F1:
                        stop_requested = True

                print(f"CAR: {(x, y, theta)}")


            surf.fill((255, 255, 255))
            self.draw_all_obstacles(surf)

            collides = cc.collides(x, y, theta)
            cc_car_coords = [pygame.math.Vector2(vec[0], self.world_width-vec[1]) * self.scale for vec in cc.get_car_poly(x,y,theta).points]
            gfxdraw.filled_polygon(surf, cc_car_coords, (0x7F, 0xFF, 0xD4))
            if collides:
                self.draw_car(surf, x-self.world_width/2, self.world_width/2-y, -theta, color_car=(0xFF, 0x45, 0x00))
                print(f"COLLISION")
            else:
                self.draw_car(surf, x-self.world_width/2, self.world_width/2-y, -theta)


            self.screen.blit(surf, (0, 0))
            if self.render_mode == "human":
                pygame.event.pump()
                self.clock.tick(self.render_fps)
                pygame.display.flip()

            if stop_requested:
                print("##########")
                print("Collision Checker:")
                print("> obs:")
                print(cc.obstacles)
                print("> car base: ")
                print(cc.get_car_basis_vertices())
                print("> transformed: ")
                print(cc.get_car_poly(x,y,theta))
                print("##########")

                animation = Animation(map_path)
                animation.draw_car(x, y, theta, colors=(animation.car_color_body, animation.car_color_roof))
                animation.show()
                exit()












