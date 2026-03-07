import numpy as np


def get_observation(env):
    # Goal vector in world frame.
    dx = env.goal[0] - env.ship.position[0]
    dy = env.goal[1] - env.ship.position[1]
    goal_vec = np.array([dx, dy], dtype=np.float32)
    goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

    # Transform to ship body frame.
    angle = env.ship.angle
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    dx_body = cos_a * dx - sin_a * dy
    dy_body = sin_a * dx + cos_a * dy

    vx, vy = env.ship.velocity
    vx_body = cos_a * vx - sin_a * vy
    vy_body = sin_a * vx + cos_a * vy

    # Angle to goal relative to ship heading.
    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - env.ship.angle
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    sin_err = np.sin(angle_error)
    cos_err = np.cos(angle_error)
    angle_error_norm = angle_error / np.pi

    # Ship heading relative to world X axis.
    ship_angle_world = (env.ship.angle + np.pi) % (2 * np.pi) - np.pi
    ship_angle_world_norm = ship_angle_world / np.pi

    prev_angle_error = float(getattr(env, "prev_angle_error", angle_error))
    delta_angle_error = angle_error - prev_angle_error
    delta_angle_error = (delta_angle_error + np.pi) % (2 * np.pi) - np.pi

    speed_to_goal = float(np.dot(env.ship.velocity, goal_dir))
    prev_speed_to_goal = float(getattr(env, "prev_speed_to_goal", speed_to_goal))
    delta_speed_to_goal = speed_to_goal - prev_speed_to_goal

    prev_action = np.array(getattr(env, "last_action", np.array([0.0, 0.0])), dtype=np.float32)
    prev_forward = float(prev_action[0])
    prev_torque = float(prev_action[1])

    dist = np.sqrt(dx * dx + dy * dy)

    max_dist = np.sqrt(env.space_size[0] ** 2 + env.space_size[1] ** 2)
    vx_scale = 20.0
    vy_scale = 20.0
    ang_vel_scale = 2.0
    delta_angle_scale = np.pi
    delta_speed_scale = 10.0

    time_to_goal_like = dist / (max(speed_to_goal, 0.0) + 1.0)
    time_to_goal_norm = np.clip(time_to_goal_like / 200.0, 0.0, 2.0)

    x, y = float(env.ship.position[0]), float(env.ship.position[1])
    width, height = env.space_size
    radius = float(env.ship.radius)
    left_clearance = x - radius
    right_clearance = width - x - radius
    top_clearance = y - radius
    bottom_clearance = height - y - radius
    clearances = [left_clearance, right_clearance, top_clearance, bottom_clearance]
    border_idx = int(np.argmin(clearances))
    nearest_border_dist = float(clearances[border_idx])
    nearest_border_dist_norm = np.clip(
        nearest_border_dist / (0.5 * min(width, height) + 1e-6),
        -1.0,
        1.0,
    )

    vx_w, vy_w = float(env.ship.velocity[0]), float(env.ship.velocity[1])
    if border_idx == 0:
        outward_normal = np.array([-1.0, 0.0], dtype=np.float32)
        border_vec_world = np.array([-left_clearance, 0.0], dtype=np.float32)
    elif border_idx == 1:
        outward_normal = np.array([1.0, 0.0], dtype=np.float32)
        border_vec_world = np.array([right_clearance, 0.0], dtype=np.float32)
    elif border_idx == 2:
        outward_normal = np.array([0.0, -1.0], dtype=np.float32)
        border_vec_world = np.array([0.0, -top_clearance], dtype=np.float32)
    else:
        outward_normal = np.array([0.0, 1.0], dtype=np.float32)
        border_vec_world = np.array([0.0, bottom_clearance], dtype=np.float32)

    boundary_approach_speed = max(0.0, float(vx_w * outward_normal[0] + vy_w * outward_normal[1]))
    boundary_approach_speed_norm = np.clip(boundary_approach_speed / 8.0, 0.0, 2.0)

    bdx_body = cos_a * border_vec_world[0] - sin_a * border_vec_world[1]
    bdy_body = sin_a * border_vec_world[0] + cos_a * border_vec_world[1]
    border_angle_body = np.arctan2(bdy_body, bdx_body)
    border_angle_body_norm = border_angle_body / np.pi

    base_obs = np.array(
        [
            dx_body / env.space_size[0],
            dy_body / env.space_size[1],
            vx_body / vx_scale,
            vy_body / vy_scale,
            sin_err,
            cos_err,
            env.ship.angular_velocity / ang_vel_scale,
            dist / max_dist,
            prev_forward,
            prev_torque,
            delta_angle_error / delta_angle_scale,
            delta_speed_to_goal / delta_speed_scale,
            time_to_goal_norm,
            nearest_border_dist_norm,
            boundary_approach_speed_norm,
            border_angle_body_norm,
            angle_error_norm,
            ship_angle_world_norm,
        ],
        dtype=np.float32,
    )

    if env.num_asteroids == 0:
        nearest_count = int(getattr(env, "nearest_rel_vel_asteroids", 2))
        rel_vel_tail = np.zeros(nearest_count * 2, dtype=np.float32)
        return np.concatenate([base_obs, rel_vel_tail], axis=0)

    asteroid_features = []
    max_radius = float(max(env.space_size))
    max_dist = np.sqrt(env.space_size[0] ** 2 + env.space_size[1] ** 2)
    vel_scale = 20.0
    asteroids_sorted = sorted(
        env.asteroids,
        key=lambda asteroid: np.linalg.norm(asteroid.position - env.ship.position)
        - (env.ship.radius + asteroid.radius),
    )
    for asteroid in asteroids_sorted[: env.num_asteroids]:
        rel = asteroid.position - env.ship.position
        center_distance = np.linalg.norm(rel)
        surface_distance = center_distance - (env.ship.radius + asteroid.radius)

        # Absolute/world bearing from ship to asteroid.
        angle_to_asteroid_world = np.arctan2(rel[1], rel[0])
        dir_to_asteroid = rel / (center_distance + 1e-6)
        closing_speed = float(np.dot(env.ship.velocity, dir_to_asteroid))
        projection_on_goal = float(np.dot(rel, goal_dir))

        asteroid_features.extend(
            [
                angle_to_asteroid_world / np.pi,
                surface_distance / (max_dist + 1e-6),
                asteroid.radius / max_radius,
                closing_speed / vel_scale,
                projection_on_goal / (max_dist + 1e-6),
            ]
        )

    feature_dim = int(getattr(env, "asteroid_feature_dim", 5))
    expected_len = env.num_asteroids * feature_dim
    if len(asteroid_features) < expected_len:
        asteroid_features.extend([0.0] * (expected_len - len(asteroid_features)))

    nearest_count = int(getattr(env, "nearest_rel_vel_asteroids", 2))
    rel_vel_tail = []
    for asteroid in asteroids_sorted[:nearest_count]:
        rel_vel_world = -env.ship.velocity
        rvx_body = cos_a * rel_vel_world[0] - sin_a * rel_vel_world[1]
        rvy_body = sin_a * rel_vel_world[0] + cos_a * rel_vel_world[1]
        rel_vel_tail.extend(
            [
                rvx_body / vel_scale,
                rvy_body / vel_scale,
            ]
        )
    expected_rel_vel_len = nearest_count * 2
    if len(rel_vel_tail) < expected_rel_vel_len:
        rel_vel_tail.extend([0.0] * (expected_rel_vel_len - len(rel_vel_tail)))

    return np.concatenate(
        [
            base_obs,
            np.array(asteroid_features, dtype=np.float32),
            np.array(rel_vel_tail, dtype=np.float32),
        ],
        axis=0,
    )
