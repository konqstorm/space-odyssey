import pygame
import numpy as np
import math
import os

class Renderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((env.space_size))
        self.env = env
        
        base = os.path.join(os.path.dirname(__file__), "../assets")
        self.ship_image = pygame.image.load(os.path.join(base, "ship.png")).convert_alpha()
        self.asteroid_image = pygame.image.load(os.path.join(base, "asteroid.png")).convert_alpha()
        self.clock = pygame.time.Clock()

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (0, 255, 0), self.env.goal.astype(int), 10) # Сделал цель чуть заметнее

        # Отрисовка астероидов
        for asteroid in self.env.asteroids:
            pos = asteroid.position.astype(int)
            diameter = int(asteroid.radius * 2)
            if diameter > 0:
                img = pygame.transform.scale(self.asteroid_image, (diameter, diameter))
                rotated = pygame.transform.rotozoom(img, -(90 + math.degrees(asteroid.angle)), 1.0)
                rect = rotated.get_rect(center=pos)
                self.screen.blit(rotated, rect)

        # Отрисовка корабля
        ship_pos = self.env.ship.position.astype(int)
        ship_size = int(self.env.ship.radius * 2)
        angle = self.env.ship.angle
        
        # Отрисовка пламени двигателей (ДО отрисовки самого корабля, чтобы огонь был "под" ним)
        forward, rot = self.env.last_action
        ship_x, ship_y = ship_pos
        
        if forward > 0.01:
            # Расчет координат для маршевого двигателя (позади корабля)
            back_x = ship_x - math.cos(angle) * self.env.ship.radius
            back_y = ship_y - math.sin(angle) * self.env.ship.radius
            
            # Длина пламени зависит от силы тяги
            flame_len = 10 * forward
            tip_x = back_x - math.cos(angle) * flame_len
            tip_y = back_y - math.sin(angle) * flame_len
            
            # Боковые точки пламени у сопла
            p1 = (back_x - math.sin(angle) * 8, back_y + math.cos(angle) * 8)
            p2 = (back_x + math.sin(angle) * 8, back_y - math.cos(angle) * 8)
            
            pygame.draw.polygon(self.screen, (255, 140, 0), [p1, p2, (tip_x, tip_y)])

        if abs(rot) > 0.01:
            # Расчет координат для маневровых двигателей
            side_mult = 1 if rot > 0 else -1
            flame_len = 15 * abs(rot)
            
            # Точка на носу корабля (сбоку), откуда бьет маневровый огонь
            side_x = ship_x + math.cos(angle) * (self.env.ship.radius * 0.8) - math.sin(angle) * (self.env.ship.radius * side_mult)
            side_y = ship_y + math.sin(angle) * (self.env.ship.radius * 0.8) + math.cos(angle) * (self.env.ship.radius * side_mult)
            
            # Направление выхлопа маневрового двигателя (перпендикулярно кораблю)
            thrust_dir = angle + (math.pi / 2) * side_mult
            
            tip_x = side_x + math.cos(thrust_dir) * flame_len
            tip_y = side_y + math.sin(thrust_dir) * flame_len
            
            p1 = (side_x - math.sin(thrust_dir) * 4, side_y + math.cos(thrust_dir) * 4)
            p2 = (side_x + math.sin(thrust_dir) * 4, side_y - math.cos(thrust_dir) * 4)
            
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