import pygame
import numpy as np
import math
import os

class Renderer:
    def __init__(self, env):
    def __init__(self, env, screen_size=None):
        pygame.init()
        self.screen = pygame.display.set_mode((env.space_size))
        self.env = env

        display_info = pygame.display.Info()
        env_width, env_height = self.env.space_size

        if screen_size is None:
            max_width = max(800, int(display_info.current_w * 0.9))
            max_height = max(600, int(display_info.current_h * 0.9))
            window_width = min(int(env_width), max_width)
            window_height = min(int(env_height), max_height)
            self.screen_size = (window_width, window_height)
        elif isinstance(screen_size, tuple):
            self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        else:
            self.screen_size = (int(screen_size), int(screen_size))

        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Space Odyssey")

        self.scale_x = self.screen_size[0] / env_width
        self.scale_y = self.screen_size[1] / env_height
        self.scale_r = min(self.scale_x, self.scale_y)
        
        base = os.path.join(os.path.dirname(__file__), "../assets")
        self.ship_image = pygame.image.load(os.path.join(base, "ship.png")).convert_alpha()
        self.asteroid_image = pygame.image.load(os.path.join(base, "asteroid.png")).convert_alpha()
        self.clock = pygame.time.Clock()

    def _to_screen(self, pos):
        return np.array([
            pos[0] * self.scale_x,
            pos[1] * self.scale_y
        ])

    def render(self):
        self.screen.fill((0, 0, 0))
        goal_pos = self._to_screen(self.env.goal).astype(int)
        goal_radius = max(3, int(10 * self.scale_r))
        pygame.draw.circle(self.screen, (0, 255, 0), goal_pos, goal_radius)

        # Отрисовка астероидов
        for asteroid in self.env.asteroids:
            pos = self._to_screen(asteroid.position).astype(int)
            diameter = max(1, int(asteroid.radius * 2 * self.scale_r))
            if diameter > 0:
                img = pygame.transform.scale(self.asteroid_image, (diameter, diameter))
                rotated = pygame.transform.rotozoom(img, -(90 + math.degrees(asteroid.angle)), 1.0)
                rect = rotated.get_rect(center=pos)
                self.screen.blit(rotated, rect)

        # Отрисовка корабля
        ship_pos = self._to_screen(self.env.ship.position).astype(int)
        ship_size = max(1, int(self.env.ship.radius * 2 * self.scale_r))
        angle = self.env.ship.angle
        
        # Отрисовка пламени двигателей (ДО отрисовки самого корабля, чтобы огонь был "под" ним)
        forward, rot = self.env.last_action
        ship_x, ship_y = ship_pos
        
        if forward > 0.01:
            # Расчет координат для маршевого двигателя (позади корабля)
            ship_radius = self.env.ship.radius * self.scale_r
            back_x = ship_x - math.cos(angle) * ship_radius
            back_y = ship_y - math.sin(angle) * ship_radius
            
            # Длина пламени зависит от силы тяги
            flame_len = 10 * self.scale_r * forward
            tip_x = back_x - math.cos(angle) * flame_len
            tip_y = back_y - math.sin(angle) * flame_len
            
            # Боковые точки пламени у сопла
            p1 = (back_x - math.sin(angle) * 8 * self.scale_r, back_y + math.cos(angle) * 8 * self.scale_r)
            p2 = (back_x + math.sin(angle) * 8 * self.scale_r, back_y - math.cos(angle) * 8 * self.scale_r)
            
            pygame.draw.polygon(self.screen, (255, 140, 0), [p1, p2, (tip_x, tip_y)])

        if abs(rot) > 0.01:
            # Расчет координат для маневровых двигателей
            side_mult = 1 if rot > 0 else -1
            flame_len = 15 * self.scale_r * abs(rot)
            
            # Точка на носу корабля (сбоку), откуда бьет маневровый огонь
            side_x = ship_x + math.cos(angle) * (self.env.ship.radius * 0.8 * self.scale_r) - math.sin(angle) * (self.env.ship.radius * side_mult * self.scale_r)
            side_y = ship_y + math.sin(angle) * (self.env.ship.radius * 0.8 * self.scale_r) + math.cos(angle) * (self.env.ship.radius * side_mult * self.scale_r)
            
            # Направление выхлопа маневрового двигателя (перпендикулярно кораблю)
            thrust_dir = angle + (math.pi / 2) * side_mult
            
            tip_x = side_x + math.cos(thrust_dir) * flame_len
            tip_y = side_y + math.sin(thrust_dir) * flame_len
            
            p1 = (side_x - math.sin(thrust_dir) * 4 * self.scale_r, side_y + math.cos(thrust_dir) * 4 * self.scale_r)
            p2 = (side_x + math.sin(thrust_dir) * 4 * self.scale_r, side_y - math.cos(thrust_dir) * 4 * self.scale_r)
            
            pygame.draw.polygon(self.screen, (0, 200, 255), [p1, p2, (tip_x, tip_y)]) # Голубое пламя для маневровых

        # Отрисовка самой текстуры корабля
        if ship_size > 0:
            factor = ship_size / self.ship_image.get_width()
            rotated = pygame.transform.rotozoom(self.ship_image, -(90 + math.degrees(angle)), factor)
            rect = rotated.get_rect(center=ship_pos)
            self.screen.blit(rotated, rect)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()