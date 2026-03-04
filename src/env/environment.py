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
    def __init__(self, space_size=(1920, 1080), num_asteroids=5, max_steps=3000):
        super().__init__()
        self.space_size = space_size
        self.num_asteroids = num_asteroids
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # X, Y and radius for each asteroid (num_asteroids * 3)
        obs_dim = 2 + 2 + 1 + 1 + 2 + self.num_asteroids * 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.throttle_gain = 3.0
        self.throttle_center = 0.2
        self.reset()

    def _is_out_of_bounds(self):
        x, y = self.ship.position
        width, height = self.space_size
        return bool(x < 0.0 or x > width or y < 0.0 or y > height)

    def _map_forward_thrust(self, action_value):
        a = float(np.clip(action_value, -1.0, 1.0))
        k = float(self.throttle_gain)
        c = float(self.throttle_center)

        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        lo = _sigmoid(k * (-1.0 - c))
        hi = _sigmoid(k * (1.0 - c))
        cur = _sigmoid(k * (a - c))
        thrust = (cur - lo) / (hi - lo + 1e-8)
        return float(np.clip(thrust, 0.0, 1.0))

    def _uniform_point_in_box(self, x_min, x_max, y_min, y_max):
        return np.array(
            [
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ],
            dtype=np.float32,
        )

    def _spawn_regions(self):
        width, height = self.space_size
        margin = 50.0
        square_size = max(50.0, min(width, height) * 0.35)
        square_size = min(square_size, width / 2.0 - margin, height - 2.0 * margin)

        y_min = (height - square_size) / 2.0
        y_max = y_min + square_size

        left_square = (margin, margin + square_size, y_min, y_max)
        right_square = (width - margin - square_size, width - margin, y_min, y_max)
        middle_region = (left_square[1], right_square[0], margin, height - margin)
        return left_square, right_square, middle_region

    def reset(self, seed=None, options=None):
        left_square, right_square, middle_region = self._spawn_regions()

        if np.random.rand() < 0.5:
            ship_square, goal_square = left_square, right_square
        else:
            ship_square, goal_square = right_square, left_square

        self.ship = Ship(self._uniform_point_in_box(*ship_square))
        self.goal = self._uniform_point_in_box(*goal_square)

        self.asteroids = []
        safe_margin = 40.0

        while len(self.asteroids) < self.num_asteroids:
            pos = self._uniform_point_in_box(*middle_region)
            radius = np.random.uniform(20.0, 150.0)
            angle = np.random.uniform(0, 2 * np.pi)

            # Keep a safe gap around both start and goal.
            if (
                np.linalg.norm(pos - self.goal) > (radius + safe_margin)
                and np.linalg.norm(pos - self.ship.position) > (radius + safe_margin)
            ):
                self.asteroids.append(Asteroid(pos, radius, angle))

        self.current_step = 0
        self.prev_distance = np.linalg.norm(self.ship.position - self.goal)
        self.last_action = np.array([0.0, 0.0])

        return get_observation(self), {}

    def step(self, action):
        # First action channel is mapped to [0, 1] with smooth sigmoid remap.
        forward_thrust = self._map_forward_thrust(action[0])
        rot_thrust = action[1]

        rot_thrust = np.clip(rot_thrust, -1.0, 1.0)
        self.last_action = np.array([forward_thrust, rot_thrust])

        dt = 0.1
        self.ship.apply_thrust(forward_thrust, rot_thrust, dt)
        self.ship.update(dt)
        distance = np.linalg.norm(self.ship.position - self.goal)
        out_of_bounds = self._is_out_of_bounds()

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
            "out_of_bounds": out_of_bounds,
        }
        reward_components = getattr(self, "_last_reward_components", None)
        if isinstance(reward_components, dict):
            info.update(reward_components)
        return get_observation(self), reward, done, truncated, info
