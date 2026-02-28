import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
        
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
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
        
        return self._get_obs(), {}

    def step(self, action):
        forward_thrust, rot_thrust = action
        
        # Ограничение 3: Только один двигатель одновременно. Выбираем тот, чей импульс больше.
        if abs(forward_thrust) >= abs(rot_thrust):
            rot_thrust = 0.0
        else:
            forward_thrust = 0.0

        forward_thrust = np.clip(forward_thrust, 0.0, 1.0)
        rot_thrust = np.clip(rot_thrust, -1.0, 1.0)
        self.last_action = np.array([forward_thrust, rot_thrust])
        
        dt = 0.1
        self.ship.apply_thrust(forward_thrust, rot_thrust, dt)
        self.ship.update(dt)
        
        current_distance = np.linalg.norm(self.ship.position - self.goal)
        reward = (self.prev_distance - current_distance) * 0.1
        self.prev_distance = current_distance
        
        done = False
        truncated = False
        
        if current_distance < self.ship.radius + 5.0:
            reward += 100.0
            done = True
            
        for asteroid in self.asteroids:
            if np.linalg.norm(self.ship.position - asteroid.position) < self.ship.radius + asteroid.radius:
                reward -= 50.0
                done = True
                break
                
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        # Добавляем радиусы астероидов в наблюдение
        asteroids_data = []
        for a in self.asteroids:
            asteroids_data.extend([a.position[0], a.position[1], a.radius])
            
        obs = np.concatenate([
            self.ship.position,
            self.ship.velocity,
            [self.ship.angle],
            [self.ship.angular_velocity],
            self.goal,
            asteroids_data
        ])
        return obs.astype(np.float32)