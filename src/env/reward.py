import numpy as np

GOAL_TERMINAL_REWARD = 600.0
ASTEROID_TERMINAL_PENALTY = -500.0
BOUNDARY_TERMINAL_PENALTY = -450.0
SPIN_TERMINAL_PENALTY = -1000.0
TIMEOUT_BASE_PENALTY = -300.0


def _termination_reward(env, current_distance):
    # Keep order consistent with env.step: goal -> asteroid -> boundary -> spin -> timeout.
    if current_distance < env.ship.radius + 5.0:
        return GOAL_TERMINAL_REWARD

    for asteroid in env.asteroids:
        if np.linalg.norm(env.ship.position - asteroid.position) < env.ship.radius + asteroid.radius:
            return ASTEROID_TERMINAL_PENALTY

    x, y = float(env.ship.position[0]), float(env.ship.position[1])
    width, height = env.space_size
    if x < 0.0 or x > width or y < 0.0 or y > height:
        return BOUNDARY_TERMINAL_PENALTY

    max_ang_speed = float(getattr(env, "max_angular_speed_kill", np.inf))
    if abs(float(env.ship.angular_velocity)) > max_ang_speed:
        return SPIN_TERMINAL_PENALTY

    if (env.current_step + 1) >= env.max_steps:
        # Strong timeout penalty with distance-dependent component.
        return TIMEOUT_BASE_PENALTY - 0.05 * float(current_distance)

    return 0.0


def reward_function(env):
    ship_pos = env.ship.position
    goal_pos = env.goal
    ship_radius = float(env.ship.radius)

    x, y = float(ship_pos[0]), float(ship_pos[1])
    width, height = env.space_size

    # 1) Progress to goal.
    current_distance = float(np.linalg.norm(ship_pos - goal_pos))
    ddist = float(env.prev_distance - current_distance)
    max_dist = float(np.sqrt(width ** 2 + height ** 2))
    dist_norm = current_distance / (max_dist + 1e-6)
    near_goal_boost = 1.0 + 1.1 / (1.0 + (dist_norm / 0.12) ** 2)
    reward_progress = float(np.clip(ddist, -5.0, 5.0) * 0.26 * near_goal_boost)

    # 2) Heading and speed alignment to goal.
    goal_vec = goal_pos - ship_pos
    goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
    speed_to_goal = float(np.dot(env.ship.velocity, goal_dir))
    reward_goal_speed = 0.12 * np.tanh(speed_to_goal / 4.0)

    target_angle = float(np.arctan2(goal_vec[1], goal_vec[0]))
    angle_error = target_angle - float(env.ship.angle)
    angle_error = (angle_error + np.pi) % (2.0 * np.pi) - np.pi
    reward_alignment = 0.07 * np.cos(angle_error) - 0.035 * abs(np.sin(angle_error))

    # Reward reducing heading error (lets policy maneuver instead of freezing).
    prev_angle_error = float(getattr(env, "prev_angle_error", angle_error))
    turn_improve = np.clip(abs(prev_angle_error) - abs(angle_error), -0.25, 0.25)
    reward_turn_improve = 0.18 * turn_improve

    # 3) Rotation/wobble regularization (softened to avoid over-penalizing turns).
    ang_vel = float(env.ship.angular_velocity)
    prev_ang_vel = float(getattr(env, "prev_angular_velocity", 0.0))
    ang_acc = ang_vel - prev_ang_vel
    rot_input = float(abs(getattr(env, "last_action", np.array([0.0, 0.0]))[1]))
    reward_wobble = (
        -0.010 * abs(ang_vel)
        -0.001 * (ang_vel ** 2)
        -0.006 * abs(ang_acc)
        -0.004 * rot_input
    )

    # 4) Boundary safety.
    left_clearance = x - ship_radius
    right_clearance = width - x - ship_radius
    top_clearance = y - ship_radius
    bottom_clearance = height - y - ship_radius
    nearest_border_dist = float(min(left_clearance, right_clearance, top_clearance, bottom_clearance))
    boundary_danger = 1.0 - np.clip(nearest_border_dist / 130.0, 0.0, 1.0)
    reward_bounds = -0.55 * (boundary_danger ** 2)

    # 5) Asteroid avoidance (nearest K only, to avoid dominating all other terms).
    reward_avoid = 0.0
    if env.asteroids:
        safe_surface_distance = 160.0
        nearest_k = 3
        asteroids_sorted = sorted(
            env.asteroids,
            key=lambda a: np.linalg.norm(a.position - ship_pos) - (ship_radius + a.radius),
        )

        for asteroid in asteroids_sorted[:nearest_k]:
            rel = asteroid.position - ship_pos
            center_distance = float(np.linalg.norm(rel))
            surface_distance = center_distance - (ship_radius + float(asteroid.radius))
            danger = 1.0 - np.clip(surface_distance / safe_surface_distance, 0.0, 1.0)

            proximity_penalty = -0.35 * (danger ** 2)

            dir_to_asteroid = rel / (center_distance + 1e-6)
            approach_speed = float(np.dot(env.ship.velocity, dir_to_asteroid))
            approach_penalty = 0.0
            if approach_speed > 0.0:
                approach_penalty = -0.07 * (approach_speed / 6.0) * (danger ** 2)

            reward_avoid += proximity_penalty + approach_penalty

    # 6) Terminal shaping.
    reward_terminal = float(_termination_reward(env, current_distance))

    non_terminal_reward = (
        reward_progress
        + reward_goal_speed
        + reward_alignment
        + reward_turn_improve
        + reward_wobble
        + reward_bounds
        + reward_avoid
    )
    reward_total = non_terminal_reward + reward_terminal

    env._last_reward_components = {
        "reward_progress": float(reward_progress),
        "reward_goal_speed": float(reward_goal_speed),
        "reward_alignment": float(reward_alignment + reward_turn_improve),
        "reward_wobble": float(reward_wobble),
        "reward_bounds": float(reward_bounds),
        "reward_avoid": float(reward_avoid),
        "reward_pass": 0.0,
        "reward_terminal": float(reward_terminal),
        "reward_total": float(reward_total),
    }

    return float(reward_total)



