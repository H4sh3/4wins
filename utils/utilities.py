import numpy as np
from numba import jit
import pygame
from PIL import Image
import torch
import torchvision.transforms as T
from gym_colonizer.envs.colors import *



@jit(parallel=True, nopython=True)
def random_position(x, y, width, height):
    x = np.random.choice(np.arange(x, width - x, 10))
    y = np.random.choice(np.arange(y, height - y, 10))
    return x, y

def get_game_screen(screen, device):
    resize = T.Compose(
        [T.ToPILImage(), T.Resize(80, interpolation=Image.CUBIC), T.ToTensor()]
    )

    screen = np.rot90(pygame.surfarray.array3d(screen))[::-1].transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


def save_game_screen(fname, img):
    im = Image.fromarray(img)
    im.save(fname)

def start_game(width, height):
    score = 0
    # Create the player
    x, y = random_position(80, 80, width, height)
    snake = Snake(x, y, GREEN, WHITE)
    # Start food?
    apples = get_apples(width, height)
    return score, snake, apples
