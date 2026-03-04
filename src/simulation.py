import pygame
from src.visualization import Renderer

def watch_agent(env, agent_class, model_path):
    print(f"Загрузка модели из {model_path}...")
    agent = agent_class(env)
    agent.load(model_path)
    renderer = Renderer(env)
    
    running = True
    while running:
        state, _ = env.reset()
        done, truncated = False, False
        episode_step = 0
        
        while not (done or truncated) and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            # Используем deterministic=True, чтобы агент не "дрожал", а выбирал лучшие действия
            action_out = agent.select_action(state, deterministic=True)
            if isinstance(action_out, tuple):
                action = action_out[0]
            else:
                action = action_out
            state, reward, done, truncated, _ = env.step(action)
            episode_step += 1

            if episode_step % 300 == 0:
                out_of_bounds = bool(
                    state is not None and (
                        env.ship.position[0] < 0 or env.ship.position[0] > env.space_size[0]
                        or env.ship.position[1] < 0 or env.ship.position[1] > env.space_size[1]
                    )
                )
                print(
                    f"watch step={episode_step}/{env.max_steps} "
                    f"out_of_bounds={out_of_bounds} "
                    f"ship=({env.ship.position[0]:.1f},{env.ship.position[1]:.1f})"
                )
            
            renderer.render()
            
            # Небольшая пауза при столкновении/победе
            if done or truncated:
                pygame.time.delay(500) 

    renderer.close()
