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

        forward_thrust = 5.0 if keys[pygame.K_UP] else 0.0
        rot_thrust = 0.0
        if keys[pygame.K_LEFT]:
            rot_thrust -= 0.1
        if keys[pygame.K_RIGHT]:
            rot_thrust += 0.1

        action = np.array([forward_thrust, rot_thrust])
        obs, reward, done, truncated, _ = env.step(action)

        renderer.render()
        if done or truncated or keys[pygame.K_SPACE]:
            env.reset()
    renderer.close()