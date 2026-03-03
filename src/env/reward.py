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

    reward = (ddist / (0.01 + ship_velocity)) * 0.1 * near_goal_boost

    # Добавляем форму награды за стабилизацию курса на цель
    dx = env.goal[0] - env.ship.position[0]
    dy = env.goal[1] - env.ship.position[1]
    target_angle = np.arctan2(dy, dx)
    angle_error = target_angle - env.ship.angle
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    reward += 0.1 * np.cos(angle_error)
    reward -= 0.1 * abs(np.sin(angle_error))
    reward -= 0.1 * abs(env.ship.angular_velocity)
    reward += 0.01 * np.linalg.norm(env.ship.velocity)

    
    if current_distance < env.ship.radius + 5.0:
        reward += 500.0
        
    for asteroid in env.asteroids:
        if np.linalg.norm(env.ship.position - asteroid.position) < env.ship.radius + asteroid.radius:
            reward -= 50.0
            break
    
    if env.current_step >= env.max_steps:
        reward -= current_distance * 0.01
    
    return reward