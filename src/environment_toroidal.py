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
        
        # without asteroids, see _get_obs
        obs_dim = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    # Вспомогательная функция для расчета дистанции в зацикленном пространстве (Торе)
    def _get_torus_distance(self, pos1, pos2):
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        # Находим кратчайший путь: напрямую или через границу экрана
        dx = min(dx, self.space_size[0] - dx)
        dy = min(dy, self.space_size[1] - dy)
        return np.hypot(dx, dy)

    def reset(self, seed=None, options=None):
        # place ship and goal with a small border from edges
        self.ship = Ship([np.random.uniform(50, self.space_size[0]-50), np.random.uniform(50, self.space_size[1]-50)])
        self.goal = np.array([np.random.uniform(50, self.space_size[0]-50), np.random.uniform(50, self.space_size[1]-50)], dtype=np.float32)
        
        # create asteroids ensuring they don't overlap goal or ship (or each other)
        self.asteroids = []
        safe_margin = 40.0 
        while len(self.asteroids) < self.num_asteroids:
            pos = [np.random.uniform(0, self.space_size[0]), np.random.uniform(0, self.space_size[1])]
            radius = np.random.uniform(20.0, 150.0)
            angle = np.random.uniform(0, 2 * np.pi)

            pos_arr = np.array(pos)
            # check against goal and ship
            if self._get_torus_distance(pos_arr, self.goal) <= (radius + safe_margin):
                continue
            if self._get_torus_distance(pos_arr, self.ship.position) <= (radius + self.ship.radius + safe_margin):
                continue
            # check against existing asteroids
            collision = False
            for a in self.asteroids:
                if self._get_torus_distance(pos_arr, a.position) <= (radius + a.radius + safe_margin):
                    collision = True
                    break
            if collision:
                continue
            self.asteroids.append(Asteroid(pos, radius, angle))
                
        self.current_step = 0
        self.prev_distance = self._get_torus_distance(self.ship.position, self.goal)
        self.last_action = np.array([0.0, 0.0])
        
        return self._get_obs(), {}

    def step(self, action):
        forward_thrust = (action[0] + 1) / 2
        rot_thrust = action[1]

        forward_thrust = np.clip(forward_thrust, 0.0, 1.0)
        rot_thrust = np.clip(rot_thrust, -1.0, 1.0)
        self.last_action = np.array([forward_thrust, rot_thrust])
        
        dt = 0.1
        self.ship.apply_thrust(forward_thrust, rot_thrust, dt)
        self.ship.update(dt)
        
        # 1. ЗАЦИКЛИВАНИЕ ПРОСТРАНСТВА (Pacman)
        self.ship.position[0] %= self.space_size[0]
        self.ship.position[1] %= self.space_size[1]
        
        # 2. НОРМАЛИЗАЦИЯ УГЛА (от 0 до 2*Pi)
        self.ship.angle %= (2 * np.pi)
        
        # Используем torus_distance вместо linalg.norm
        current_distance = self._get_torus_distance(self.ship.position, self.goal)
        # old linear reward based on distance reduction, now replaced with a smoother potential function of distance
        # reward = (self.prev_distance - current_distance) * 0.1
        
        prev_phi = -np.sqrt(self.prev_distance + 1e-6)
        curr_phi = -np.sqrt(current_distance + 1e-6)
        reward = 0.5 * (curr_phi - prev_phi)

        self.prev_distance = current_distance
        
        done = False
        truncated = False
        
        if current_distance < self.ship.radius + 5.0:
            reward += 100.0
            done = True
            
        for asteroid in self.asteroids:
            if self._get_torus_distance(self.ship.position, asteroid.position) < self.ship.radius + asteroid.radius:
                reward -= 50.0
                done = True
                break
                
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, done, truncated, {}

    # for now i removed asteroid data from here, but it shoud be re-added when asteroids will be opt-in
    def _get_obs(self):
        # --- ВЕКТОР ДО ЦЕЛИ С УЧЁТОМ TORUS ---
        dx = self.goal[0] - self.ship.position[0]
        dy = self.goal[1] - self.ship.position[1]

        dx = (dx + self.space_size[0] / 2) % self.space_size[0] - self.space_size[0] / 2
        dy = (dy + self.space_size[1] / 2) % self.space_size[1] - self.space_size[1] / 2

        # --- ПОВОРОТ В BODY FRAME ---
        angle = self.ship.angle
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        dx_body = cos_a * dx - sin_a * dy
        dy_body = sin_a * dx + cos_a * dy

        vx, vy = self.ship.velocity
        vx_body = cos_a * vx - sin_a * vy
        vy_body = sin_a * vx + cos_a * vy

        # --- УГЛОВАЯ ОШИБКА ДО ЦЕЛИ ---
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.ship.angle
        angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi

        sin_err = np.sin(angle_error)
        cos_err = np.cos(angle_error)

        # --- РАССТОЯНИЕ ---
        dist = np.sqrt(dx*dx + dy*dy)

        # len 8 array total
        obs = np.array([
            dx_body,
            dy_body,
            vx_body,
            vy_body,
            sin_err,
            cos_err,
            self.ship.angular_velocity,
            dist
        ], dtype=np.float32)

        return obs