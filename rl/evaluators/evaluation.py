import numpy as np
import time
import math
import pygame
from pygame import gfxdraw
metadata = {"render.modes": ["human"], "render_fps":10}

def render_evaluation_results(path: str, screen_width = 500, screen_height = 500, render_mode = "human", max_x = 1, goal_tolerance_pos = 0.1):
    cart_length = 10
    cart_height = math.sqrt(3.0/4.0) * cart_length * 2
    default_cart_coords = np.array([(0-cart_length/2,0), (cart_length/2, 0), (0, cart_height)])
    pygame.init()
    if render_mode == "human":
        pygame.display.init()
        screen = pygame.display.set_mode(
            (screen_width, screen_height)
        )
    else:  # mode == "rgb_array"
        screen = pygame.Surface((screen_width, screen_height))
    clock = pygame.time.Clock()

    world_width = max_x * 2
    scale = screen_width / world_width

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    stats = np.genfromtxt(path, delimiter=",", skip_header=1)
    print(f"read stats: {stats}")

    results = [perfect - result for (_,_,_,perfect, result) in stats]
    worst_result = min(results)

    for (x,y,theta,perfect,result) in stats:
        print(f"({(x,y,theta,perfect,result)})")
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = y * scale + screen_height / 2.0  # MIDDLE OF CART

        cart_coords = [pygame.math.Vector2((c[0], c[1])).rotate_rad(theta - 0.5 * math.pi) + pygame.math.Vector2((cartx, carty))for c in default_cart_coords]

        score = perfect - result
        red = 0
        green = 0
        blue = 0
        print(f"score: {score} [worst: {worst_result}]")
        red = int(255 * abs(score/worst_result))
        green = 255 - red
        color = (red,green,blue)
        print(f"color: {color}")
        gfxdraw.aapolygon(surf, cart_coords, color)
        gfxdraw.filled_polygon(surf, cart_coords, color)

    cord_target_x = screen_width / 2
    cord_target_y = screen_height / 2
    gfxdraw.circle(surf, int(cord_target_x), int(cord_target_y), int(goal_tolerance_pos * scale), (10, 0, 0))

    screen.blit(surf, (0, 0))

    if render_mode == "human":
        pygame.event.pump()
        clock.tick(metadata["render_fps"])
        pygame.display.flip()

    elif render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )
    choice = input("save? [Y|n]")
    if choice != "n":
        pygame.image.save(surf, f"evaluation_{time.time()}.jpg")

if __name__ == "__main__":
    render_evaluation_results("../stats.csv")