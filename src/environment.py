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
        self.angular_velocity += rot_thrust * dt * 0.35
        self.angular_velocity = np.clip(self.angular_velocity, -1.2, 1.2)

    def update(self, dt):
        self.angle += self.angular_velocity * dt
        self.position += self.velocity * dt
        self.velocity *= 0.995
        self.angular_velocity *= 0.88

class Asteroid:
    def __init__(self, position, radius, angle=0.0):
        self.position = np.array(position, dtype=np.float32)
        self.radius = float(radius)
        self.angle = angle

class SpaceEnv(gym.Env):
    def __init__(self, space_size=(1920, 1080), num_asteroids=5, max_steps=1000):
        super().__init__()
        self.space_size = space_size
        self.max_distance = np.hypot(self.space_size[0], self.space_size[1])
        self.thrust_scale = 0.45
        self.num_asteroids = num_asteroids
        self.max_steps = max_steps
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Обновлено: теперь мы передаем X, Y и Radius для каждого астероида (num_asteroids * 3)
        obs_dim = 2 + 2 + 1 + 1 + 2 + self.num_asteroids * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def _get_distance(self, pos1, pos2):
        return float(np.linalg.norm((pos1 - pos2).astype(np.float64)))

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

            pos_arr = np.array(pos)
            # check against goal and ship
            if self._get_distance(pos_arr, self.goal) <= (radius + safe_margin):
                continue
            if self._get_distance(pos_arr, self.ship.position) <= (radius + self.ship.radius + safe_margin):
                continue
            # check against existing asteroids
            collision = False
            for a in self.asteroids:
                if self._get_distance(pos_arr, a.position) <= (radius + a.radius + safe_margin):
                    collision = True
                    break
            if collision:
                continue
            self.asteroids.append(Asteroid(pos, radius, angle))
                
        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.ship.position - self.goal)
        self.last_action = np.array([0.0, 0.0]) # Сохраняем для визуализации
        
        return self._get_obs(), {}

    def step(self, action):
        forward_thrust = (action[0] + 1) / 2
        rot_thrust = action[1]

        forward_thrust = forward_thrust * self.thrust_scale
        rot_thrust = np.clip(rot_thrust, -1.0, 1.0)
        self.last_action = np.array([forward_thrust, rot_thrust])
        
        dt = 0.1
        self.ship.apply_thrust(forward_thrust, rot_thrust, dt)
        self.ship.update(dt)

        self.ship.angle %= (2 * np.pi)

        out_of_bounds = (
            self.ship.position[0] < 0.0 or self.ship.position[0] > self.space_size[0] or
            self.ship.position[1] < 0.0 or self.ship.position[1] > self.space_size[1]
        )
        if out_of_bounds:
            current_distance = self._get_distance(self.ship.position, self.goal)
            distance_delta = current_distance - self.prev_distance
            self.current_step += 1
            return self._get_obs(), -120.0, True, False, {
                "termination_reason": "out_of_bounds",
                "distance": current_distance,
                "distance_delta": distance_delta
            }
        
        current_distance = self._get_distance(self.ship.position, self.goal)

        dx = self.goal[0] - self.ship.position[0]
        dy = self.goal[1] - self.ship.position[1]

        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.ship.angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        distance_norm = np.clip(current_distance / (self.max_distance + 1e-8), 0.0, 1.0)
        proximity = (1.0 - distance_norm) ** 2
        proximity_quadratic = proximity ** 2

        distance_delta = current_distance - self.prev_distance
        distance_progress = -distance_delta / (self.max_distance + 1e-8)
        distance_reward = distance_progress * (0.1 + 2.6 * proximity_quadratic)

        angle_alignment = np.cos(angle_error)
        angle_reward = 0.45 * angle_alignment * (0.35 + 0.65 * proximity)

        reward = distance_reward + angle_reward

        self.prev_distance = current_distance
        
        done = False
        truncated = False
        
        if current_distance < self.ship.radius + 5.0:
            reward += 100.0
            done = True
            
        for asteroid in self.asteroids:
            if self._get_distance(self.ship.position, asteroid.position) < self.ship.radius + asteroid.radius:
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
        dist = self._get_distance(self.ship.position, self.goal)

        dx_body_norm = np.clip(dx_body / (self.space_size[0] + 1e-8), -1.0, 1.0)
        dy_body_norm = np.clip(dy_body / (self.space_size[1] + 1e-8), -1.0, 1.0)
        vx_body_norm = np.clip(vx_body / 20.0, -1.0, 1.0)
        vy_body_norm = np.clip(vy_body / 20.0, -1.0, 1.0)
        angular_velocity_norm = np.clip(self.ship.angular_velocity / 1.2, -1.0, 1.0)
        dist_norm = np.clip(dist / (self.max_distance + 1e-8), 0.0, 1.0)

        # len 8 array total
        obs = np.array([
            dx_body_norm,
            dy_body_norm,
            vx_body_norm,
            vy_body_norm,
            sin_err,
            cos_err,
            angular_velocity_norm,
            dist_norm
        ], dtype=np.float32)

        return obs