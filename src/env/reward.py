import numpy as np

def reward_function(env):
    current_distance = np.linalg.norm(env.ship.position - env.goal)
    ddist = env.prev_distance - current_distance
    ship_velocity = np.linalg.norm(env.ship.velocity)

    # Буст за прогресс вблизи цели:
    # далеко коэффициент ~1, рядом с целью плавно растет, но ограниченно.
    max_dist = np.sqrt(env.space_size[0] ** 2 + env.space_size[1] ** 2)
    dist_norm = current_distance / (max_dist + 1e-6)
    alpha = 1.2   # максимум: коэффициент = 1 + alpha (т.е. 2.2x)
    d0 = 0.12     # ширина "зоны усиления" в долях max_dist
    near_goal_boost = 1.0 + alpha / (1.0 + (dist_norm / d0) ** 2)

    progress_reward = ddist * 0.3 * near_goal_boost

    # Добавляем форму награды за стабилизацию курса на цель
    dx = env.goal[0] - env.ship.position[0]
    dy = env.goal[1] - env.ship.position[1]
    goal_vec = np.array([dx, dy], dtype=np.float32)
    goal_dist = np.linalg.norm(goal_vec) + 1e-6
    goal_dir = goal_vec / goal_dist
    speed_to_goal = float(np.dot(env.ship.velocity, goal_dir))
    goal_speed_reward = 0.18 * np.tanh(speed_to_goal / 4.0)

    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - env.ship.angle
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    align_cos_reward = 0.16 * np.cos(angle_error)
    align_sin_penalty = -0.08 * abs(np.sin(angle_error))
    ang_vel_abs_penalty = -0.06 * abs(env.ship.angular_velocity)
    ang_vel_sq_penalty = -0.006 * (env.ship.angular_velocity ** 2) # штраф за "дрожание" при попытке стабилизации
    #reward += 0.01 * np.linalg.norm(env.ship.velocity)

    # --- Obstacle shaping: штрафуем опасную близость и "влет" в астероид ---
    obstacle_proximity_penalty = 0.0
    obstacle_approach_penalty = 0.0
    if env.asteroids:
        safe_surface_distance = 140.0
        proximity_asteroid_count = 3
        asteroids_sorted = sorted(
            env.asteroids,
            key=lambda asteroid: np.linalg.norm(asteroid.position - env.ship.position) - (env.ship.radius + asteroid.radius),
        )

        for asteroid in asteroids_sorted[:proximity_asteroid_count]:
            rel = asteroid.position - env.ship.position
            center_distance = np.linalg.norm(rel)
            surface_distance = center_distance - (env.ship.radius + asteroid.radius)

            if surface_distance >= safe_surface_distance:
                continue

            danger = 1.0 - np.clip(surface_distance / safe_surface_distance, 0.0, 1.0)
            obstacle_proximity_penalty += -0.8 * (danger ** 2)

            denom = np.linalg.norm(rel) + 1e-6
            dir_to_asteroid = rel / denom
            approach_speed = float(np.dot(env.ship.velocity, dir_to_asteroid))
            if approach_speed > 0.0:
                obstacle_approach_penalty += -0.12 * (approach_speed / 8.0) * (danger ** 2)

    goal_bonus = 0.0
    collision_penalty = 0.0
    timeout_penalty = 0.0

    reward = (
        progress_reward
        + goal_speed_reward
        + align_cos_reward
        + align_sin_penalty
        + ang_vel_abs_penalty
        + ang_vel_sq_penalty
        + obstacle_proximity_penalty
        + obstacle_approach_penalty
    )

    
    if current_distance < env.ship.radius + 5.0:
        goal_bonus = 500.0
        reward += goal_bonus
        
    for asteroid in env.asteroids:
        if np.linalg.norm(env.ship.position - asteroid.position) < env.ship.radius + asteroid.radius:
            collision_penalty = -200.0
            reward += collision_penalty
            break
    
    if (env.current_step + 1) >= env.max_steps:
        timeout_penalty = -current_distance * 0.01
        reward += timeout_penalty

    env._last_reward_components = {
        "reward_progress": float(progress_reward),
        "reward_goal_speed": float(goal_speed_reward),
        "reward_alignment": float(align_cos_reward + align_sin_penalty),
        "reward_wobble": float(ang_vel_abs_penalty + ang_vel_sq_penalty),
        "reward_avoid": float(obstacle_proximity_penalty + obstacle_approach_penalty),
        "reward_terminal": float(goal_bonus + collision_penalty + timeout_penalty),
        "reward_total": float(reward),
    }
    
    return reward