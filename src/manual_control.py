# manual_control.py
import pygame
from visualization import Renderer
import numpy as np

def manual_control(env):
    renderer = Renderer(env)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        forward_thrust = 10 if keys[pygame.K_UP] or keys[pygame.K_w] else -1
        rot_thrust = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            rot_thrust -= 0.1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            rot_thrust += 0.1
        if keys[pygame.K_q]:
            env.ship.velocity = 0.0
            env.ship.angular_velocity = 0.0

        action = np.array([forward_thrust, rot_thrust])
        obs, reward, done, truncated, _ = env.step(action)
        # prepare vertical list of values
        lines = [
            f"dx_body: {obs[0]:.2f}",
            f"dy_body: {obs[1]:.2f}",
            f"vx_body: {obs[2]:.2f}",
            f"vy_body: {obs[3]:.2f}",
            f"sin_err: {obs[4]:.2f}",
            f"cos_err: {obs[5]:.2f}",
            f"angular_v: {obs[6]:.2f}",
            f"dist: {obs[7]:.2f}",
            f"REWARD: {reward:.2f}",
        ]
        # move cursor up to overwrite previous block if needed
        if hasattr(manual_control, "_prev_lines") and manual_control._prev_lines:
            print("\033[F" * manual_control._prev_lines, end="")
        for l in lines:
            # clear to end of line after printing
            print(l + "\033[K")
        manual_control._prev_lines = len(lines)

        renderer.render()
        if done or truncated or keys[pygame.K_SPACE]:
            # freeze for a bit
            pygame.time.delay(500)
            env.reset()
    renderer.close()