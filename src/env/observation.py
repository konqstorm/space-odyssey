import numpy as np

def get_observation(env):
    # --- ВЕКТОР ДО ЦЕЛИ ---
    dx = env.goal[0] - env.ship.position[0]
    dy = env.goal[1] - env.ship.position[1]

    # --- ПОВОРОТ В BODY FRAME ---
    angle = env.ship.angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    dx_body = cos_a * dx - sin_a * dy
    dy_body = sin_a * dx + cos_a * dy

    vx, vy = env.ship.velocity
    vx_body = cos_a * vx - sin_a * vy
    vy_body = sin_a * vx + cos_a * vy

    # --- УГЛОВАЯ ОШИБКА ДО ЦЕЛИ ---
    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - env.ship.angle
    angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi

    sin_err = np.sin(angle_error)
    cos_err = np.cos(angle_error)

    # --- РАССТОЯНИЕ ---
    dist = np.sqrt(dx*dx + dy*dy)

    max_dist = np.sqrt(env.space_size[0] ** 2 + env.space_size[1] ** 2)
    vx_scale = 20.0
    vy_scale = 20.0
    ang_vel_scale = 2.0

    base_obs = np.array([
        dx_body / env.space_size[0],
        dy_body / env.space_size[1],
        vx_body / vx_scale,
        vy_body / vy_scale,
        sin_err,
        cos_err,
        env.ship.angular_velocity / ang_vel_scale,
        dist / max_dist
    ], dtype=np.float32)

    if env.num_asteroids == 0:
        return base_obs

    asteroid_features = []
    max_radius = float(max(env.space_size))
    asteroids_sorted = sorted(
        env.asteroids,
        key=lambda asteroid: np.linalg.norm(asteroid.position - env.ship.position)
    )
    for asteroid in asteroids_sorted[:env.num_asteroids]:
        rel = asteroid.position - env.ship.position
        ax_body = cos_a * rel[0] - sin_a * rel[1]
        ay_body = sin_a * rel[0] + cos_a * rel[1]
        asteroid_features.extend([
            ax_body / env.space_size[0],
            ay_body / env.space_size[1],
            asteroid.radius / max_radius
        ])

    expected_len = env.num_asteroids * 3
    if len(asteroid_features) < expected_len:
        asteroid_features.extend([0.0] * (expected_len - len(asteroid_features)))

    return np.concatenate([base_obs, np.array(asteroid_features, dtype=np.float32)], axis=0)