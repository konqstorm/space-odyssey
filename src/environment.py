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
        
        return self._get_obs(), {}

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

        current_distance = np.linalg.norm(self.ship.position - self.goal)
        ddist = self.prev_distance - current_distance
        ship_velocity = np.linalg.norm(self.ship.velocity)

        # Буст за прогресс вблизи цели:
        # далеко коэффициент ~1, рядом с целью плавно растет, но ограниченно.
        max_dist = np.sqrt(self.space_size[0] ** 2 + self.space_size[1] ** 2)
        dist_norm = current_distance / (max_dist + 1e-6)
        alpha = 1.2   # максимум: коэффициент = 1 + alpha (т.е. 2.2x)
        d0 = 0.12     # ширина "зоны усиления" в долях max_dist
        near_goal_boost = 1.0 + alpha / (1.0 + (dist_norm / d0) ** 2)

        reward = (ddist / (0.01 + ship_velocity)) * 0.1 * near_goal_boost
        self.prev_distance = current_distance

        # Добавляем форму награды за стабилизацию курса на цель
        dx = self.goal[0] - self.ship.position[0]
        dy = self.goal[1] - self.ship.position[1]
        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - self.ship.angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
        reward += 0.1 * np.cos(angle_error)
        reward -= 0.1 * abs(np.sin(angle_error))
        reward -= 0.1 * abs(self.ship.angular_velocity)
        reward -= 0.005 * np.linalg.norm(self.ship.velocity)
        
        done = False
        truncated = False
        termination_reason = "running"
        
        if current_distance < self.ship.radius + 5.0:
            reward += 200.0
            done = True
            termination_reason = "goal"
            
        for asteroid in self.asteroids:
            if np.linalg.norm(self.ship.position - asteroid.position) < self.ship.radius + asteroid.radius:
                reward -= 50.0
                done = True
                termination_reason = "asteroid"
                break
                
        self.current_step += 1
        if self.current_step >= self.max_steps and not done:
            truncated = True
            termination_reason = "timeout"

        info = {
            "termination_reason": termination_reason,
            "distance_to_goal": float(current_distance),
        }
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        # --- ВЕКТОР ДО ЦЕЛИ ---
        dx = self.goal[0] - self.ship.position[0]
        dy = self.goal[1] - self.ship.position[1]

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

        max_dist = np.sqrt(self.space_size[0] ** 2 + self.space_size[1] ** 2)
        vx_scale = 20.0
        vy_scale = 20.0
        ang_vel_scale = 2.0

        base_obs = np.array([
            dx_body / self.space_size[0],
            dy_body / self.space_size[1],
            vx_body / vx_scale,
            vy_body / vy_scale,
            sin_err,
            cos_err,
            self.ship.angular_velocity / ang_vel_scale,
            dist / max_dist
        ], dtype=np.float32)

        if self.num_asteroids == 0:
            return base_obs

        asteroid_features = []
        max_radius = float(max(self.space_size))
        asteroids_sorted = sorted(
            self.asteroids,
            key=lambda asteroid: np.linalg.norm(asteroid.position - self.ship.position)
        )
        for asteroid in asteroids_sorted[:self.num_asteroids]:
            rel = asteroid.position - self.ship.position
            ax_body = cos_a * rel[0] - sin_a * rel[1]
            ay_body = sin_a * rel[0] + cos_a * rel[1]
            asteroid_features.extend([
                ax_body / self.space_size[0],
                ay_body / self.space_size[1],
                asteroid.radius / max_radius
            ])

        expected_len = self.num_asteroids * 3
        if len(asteroid_features) < expected_len:
            asteroid_features.extend([0.0] * (expected_len - len(asteroid_features)))

        return np.concatenate([base_obs, np.array(asteroid_features, dtype=np.float32)], axis=0)
