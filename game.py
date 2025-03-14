import multiprocessing as mp
import os
from multiprocessing import Event

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

# from multiprocessing.synchronize import Event
SCREEN_WIDTH = 1920  # Larghezza schermo totale
SCREEN_HEIGHT = 1080  # Altezza schermo totale
WINDOW_WIDTH = 960  # Meta larghezza schermo

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_window_position(x, y):
    """Imposta la posizione della finestra SDL"""
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"


KEY_ACTIONS = {
    pygame.K_UP: np.array([0, 1, 0]),
    pygame.K_DOWN: np.array([0, 0, 0.8]),
    pygame.K_LEFT: np.array([-1, 0, 0]),
    pygame.K_RIGHT: np.array([1, 0, 0]),
}


def get_action(keys):
    action = np.array([0.0, 0.0, 0.0])
    for key, act in KEY_ACTIONS.items():
        if keys[key]:
            action += act
    return np.clip(action, -1, 1)


def run_ai(user_started, user_end, user_reset, seed, name="AI"):
    np.random.seed(seed)
    set_window_position(WINDOW_WIDTH - 150, 0)
    # Configurazione ambiente AI
    model_path = os.path.join(
        "checkpoints",
        "PPOClippedFalseStackedFrames4DiscreteFalseAccBrakeFalse_1000000_steps.zip",
    )
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    pygame.init()
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        max_episode_steps=10000,
        lap_complete_percent=0.95,
    )
    _, _ = env.reset(seed=seed)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env, log_dir)
    env.unwrapped.screen = pygame.display.set_mode(
        (WINDOW_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    model = PPO.load(model_path, env)

    obs = env.reset()

    clock = pygame.time.Clock()
    maxreward_ia = 0
    reward = 0
    endgame = False
    counter = 0
    time_set = False
    starting_time = 0
    try:
        running = True
        while running:
            if user_started.is_set() and not endgame:
                if not time_set:
                    starting_time = pygame.time.get_ticks()
                    time_set = True
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated = env.step(action)

                if done and reward > 0:
                    if counter < 1:
                        counter = counter + 1
                        continue
                    # Display endgame screen using Pygame
                    end_time = pygame.time.get_ticks()
                    font = pygame.font.SysFont("Arial", 48)
                    final_time = (end_time - starting_time) / 1000
                    text_surface = font.render(
                        "{} finished in {}s!".format(name, final_time),
                        True,
                        (255, 255, 255),
                    )
                    # Clear the screen (fill with black, for example)
                    env.unwrapped.screen.fill((0, 0, 0))
                    # Center the text
                    text_rect = text_surface.get_rect(
                        center=(WINDOW_WIDTH // 2, SCREEN_HEIGHT // 2)
                    )
                    env.unwrapped.screen.blit(text_surface, text_rect)
                    pygame.display.flip()
                    endgame = True

            if maxreward_ia < reward:
                maxreward_ia = reward
                # Gestione eventi
            if user_end.is_set():
                running = False
            if user_reset.is_set():
                obs = env.reset()
                user_reset.clear()
                user_started.clear()
                endgame = False
                counter = 0
                time_set = False
            clock.tick(50)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


def run_human(user_started, user_end, user_reset, seed, name="Human"):
    np.random.seed(seed)
    set_window_position(0, 0)
    # Configurazione ambiente umano
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        max_episode_steps=10000,
        lap_complete_percent=0.95,
    )
    obs, info = env.reset(seed=seed)
    # obs, info = env.reset()
    env.unwrapped.screen = pygame.display.set_mode(
        (WINDOW_WIDTH - 150, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.init()
    clock = pygame.time.Clock()
    first_action = True
    endgame = False
    counter = 0
    starting_time = 0
    try:
        running = True
        while running:
            keys = pygame.key.get_pressed()
            action = get_action(keys)
            if not endgame:
                obs, reward, done, truncated, info = env.step(action)

            if done and reward > 0 and not endgame:
                if counter < 1:
                    counter = counter + 1
                    obs, info = env.reset(seed=seed)
                    continue
                # Display endgame screen using Pygame
                end_time = pygame.time.get_ticks()
                final_time = (end_time - starting_time) / 1000
                font = pygame.font.SysFont("Arial", 48)
                text_surface = font.render(
                    "{} finished in {}s!".format(name, final_time),
                    True,
                    (255, 255, 255),
                )
                # Clear the screen (fill with black, for example)
                env.unwrapped.screen.fill((0, 0, 0))
                # Center the text
                text_rect = text_surface.get_rect(
                    center=(WINDOW_WIDTH // 2, SCREEN_HEIGHT // 2)
                )
                env.unwrapped.screen.blit(text_surface, text_rect)
                pygame.display.flip()
                endgame = True

            if np.any(action != [0, 0, 0]) and first_action:
                # Prima azione utente rilevata
                user_started.set()
                first_action = False
                starting_time = pygame.time.get_ticks()
            # Gestione eventi
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                    user_end.set()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    user_reset.set()
                    first_action = True
                    endgame = False
                    counter = 0

            clock.tick(50)
            if (done or truncated) and reward < 0:
                obs, info = env.reset(seed=seed)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    seed = 0
    user_started = Event()
    user_end = Event()
    user_reset = Event()
    human_process = mp.Process(
        target=run_human,
        args=(
            user_started,
            user_end,
            user_reset,
            seed,
        ),
    )
    ai_process = mp.Process(
        target=run_ai,
        args=(
            user_started,
            user_end,
            user_reset,
            seed,
        ),
    )

    human_process.start()
    ai_process.start()

    human_process.join()
    ai_process.join()
