import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .reward import reward_function
from .observation import get_observation

class Ship:
    def __init__(self, position, angle=0.0):
        self.radius = 20.0
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.angle = angle
        self.angular_velocity = 0.0

    def apply_thrust(self, forward_thrust, rot_thrust, dt):
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.velocity += forward_thrust * direction * dt
        self.angular_velocity += rot_thrust * dt

    def update(self, dt):
        self.angle += self.angular_velocity * dt
        self.position += self.velocity * dt

class Asteroid:
    def __init__(self, position, radius, angle=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.radius = float(radius)
        self.angle = angle

class SpaceEnv(gym.Env):
    def __init__(self, space_size=(1920, 1080), num_asteroids=5, max_steps=1000):
        super().__init__()
        self.space_size = space_size
        self.num_asteroids = num_asteroids
        self.max_steps = max_steps
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Обновлено: теперь мы передаем X, Y и Radius для каждого астероида (num_asteroids * 3)
        obs_dim = 2 + 2 + 1 + 1 + 2 + self.num_asteroids * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        # Корабль и цель лучше не спавнить у самых краев
        self.ship = Ship([np.random.uniform(50, self.space_size[0]-50), np.random.uniform(50, self.space_size[1]-50)])
        self.goal = np.array([np.random.uniform(50, self.space_size[0]-50), np.random.uniform(50, self.space_size[1]-50)], dtype=np.float32)
        
        self.asteroids = []
        safe_margin = 40.0 # Дополнительный зазор вокруг цели
        
        while len(self.asteroids) < self.num_asteroids:
            pos = [np.random.uniform(0, self.space_size[0]), np.random.uniform(0, self.space_size[1])]
            radius = np.random.uniform(20.0, 150.0) # Астероиды разных размеров
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Проверка: астероид не должен перекрывать финиш
            if np.linalg.norm(np.array(pos) - self.goal) > (radius + safe_margin):
                self.asteroids.append(Asteroid(pos, radius, angle))
                
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.ship.position - self.goal)
        self.last_action = np.array([0.0, 0.0]) # Сохраняем для визуализации
        
        return get_observation(self), {}

    def step(self, action):
        # Первый канал действия интерпретируем как "газ" в диапазоне [-1, 1],
        # где <= 0 означает отсутствие тяги вперед.
        forward_thrust = action[0]
        rot_thrust = action[1]

        forward_thrust = np.clip(forward_thrust, 0.0, 1.0)
        rot_thrust = np.clip(rot_thrust, -1.0, 1.0)
        self.last_action = np.array([forward_thrust, rot_thrust])
        
        dt = 0.1
        self.ship.apply_thrust(forward_thrust, rot_thrust, dt)
        self.ship.update(dt)
        distance = np.linalg.norm(self.ship.position - self.goal)

        reward = reward_function(self)
        
        done = False
        truncated = False
        termination_reason = "running"
        
        if distance < self.ship.radius + 5.0:
            done = True
            termination_reason = "goal"
            
        for asteroid in self.asteroids:
            if np.linalg.norm(self.ship.position - asteroid.position) < self.ship.radius + asteroid.radius:
                done = True
                termination_reason = "asteroid"
                break
                
        self.current_step += 1
        if self.current_step >= self.max_steps and not done:
            truncated = True
            termination_reason = "timeout"

        self.prev_distance = distance

        info = {
            "termination_reason": termination_reason,
            "distance_to_goal": float(distance),
        }
        return get_observation(self), reward, done, truncated, info

    
