import pygame
from visualization import Renderer

def watch_agent(env, agent_class, model_path):
    print(f"Загрузка модели из {model_path}...")
    agent = agent_class(env)
    agent.load(model_path)
    renderer = Renderer(env)
    
    running = True
    while running:
        state, _ = env.reset()
        done, truncated = False, False
        
        while not (done or truncated) and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            # Используем deterministic=True, чтобы агент не "дрожал", а выбирал лучшие действия
            action, _, _ = agent.select_action(state, deterministic=False)
            state, reward, done, truncated, _ = env.step(action)
            
            renderer.render()
            
            # Небольшая пауза при столкновении/победе
            if done or truncated:
                pygame.time.delay(500) 

    renderer.close()