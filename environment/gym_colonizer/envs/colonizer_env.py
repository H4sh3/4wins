import gym
import numpy as np
import copy
import random
import math
from math import sqrt, pow
from gym import spaces, error
import uuid
from gym_colonizer.envs.colors import Colors
import time

BUBBLE_RADIUS = 20


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Spot():
    def __init__(self, x, y):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 5
        self.color = Colors.gray()
        self.id = uuid.uuid1()
        self.close_resource = []
        self.close_road = []

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)


class Resource():
    def __init__(self, x, y,number):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 5
        self.color = Colors.yellow()
        self.id = uuid.uuid1()
        self.close_spot = []
        self.number = number

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)
    
        font = pygame.font.SysFont("arial", 25)
        text = font.render(str(self.number), 3, (0, 0, 255))
        screen.blit(text, (self.pos.x, self.pos.y))


class Road():
    def __init__(self, x, y):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 3
        self.color = (100, 200, 100)
        self.id = uuid.uuid1()
        self.close_spot = []

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)


class ColonizerEnv(gym.Env):
    # TODO: Add seeding function and fps lock
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second': 350}

    gray = (100, 100, 100)
    white = (255, 255, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 0)
    orange = (255, 128, 0)
    purple = (255, 0, 255)
    cyan = (0, 255, 255)
    black = (0, 0, 0)

    colors = [red, green, blue, yellow, orange, purple, cyan]

    def __init__(self):
        self.window_height = 1000
        self.window_width = 1000
        self.reset()
        self.spots = {}
        self.roads = {}
        self.resources = {}
        self.lines = []
        self.initMap()

    def initMap(self):

        distance = 20

        amount = 4
        mid_y = self.window_height/1.5
        size = 80
        mid_x = self.window_width/2.5
        switch = False
        height = 13

        for y in range(1, height):
            if(y % 2 == 0):
                if(y < height/2):
                    amount += 1
                    if(y % 2 == 0):
                        mid_x -= size/2
                else:
                    amount -= 1
                    if(y % 2 == 0):
                        mid_x += size/2
            for x in range(1, amount):
                s = Spot(mid_x+x*size, mid_y-y*(size-10)/2)
                self.spots[s.id] = s

        used_positions = []
        numbers = [2,3,3,4,4,5,5,6,6,7,8,8,9,9,10,10,11,11,12]
        random.shuffle(numbers)
        for ks1 in self.spots:
            s1 = self.spots[ks1]
            for ks2 in self.spots:
                s2 = self.spots[ks2]
                if s1 == s2: continue
                else:
                    dist = round(sqrt((s2.pos.x - s1.pos.x) **
                                      2 + (s2.pos.y - s1.pos.y)**2))
                    x = (s1.pos.x+s2.pos.x)/2
                    y = (s1.pos.y+s2.pos.y)/2
                    if [x,y] in used_positions: continue

                    if dist == 53 or dist == 35:
                        road = Road(x, y)
                        self.roads[road.id] = road
                        used_positions.append([x,y])
                    if dist == 105:
                        resource = Resource(x,y,numbers.pop())
                        self.resources[resource.id]=resource
                        used_positions.append([x,y])

        # relation between spots and resources
        for s in self.spots:
            spot = self.spots[s]
            for r in self.resources:
                res = self.resources[r]
                dist = round(sqrt((spot.pos.x - res.pos.x) ** 2 + (spot.pos.y - res.pos.y)**2))
                if dist == 52 or dist == 44:
                    self.spots[s].close_resource.append(res.id)
                    self.resources[r].close_spot.append(spot.id)
                    self.lines.append([int(spot.pos.x),int(spot.pos.y),int(res.pos.x),int(res.pos.y)])

        # relation between spots and roads
        for s in self.spots:
            spot = self.spots[s]
            for r in self.roads:
                road = self.roads[r]
                dist = round(sqrt((spot.pos.x - road.pos.x) ** 2 + (spot.pos.y - road.pos.y)**2))
                if dist == 27 or dist == 18:
                    self.spots[s].close_road.append(road.id)
                    self.roads[r].close_spot.append(spot.id)
                    self.lines.append([int(spot.pos.x),int(spot.pos.y),int(road.pos.x),int(road.pos.y)])
    
        print('resources',len(self.resources))
        print('roads',len(self.roads))
        print('used_positions',len(used_positions))        

    def reset(self):
        self.screen = None

    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """

        if mode == 'console':
            print(self._get_game_state)
        elif mode == "human":
            try:
                import pygame
                from pygame import gfxdraw
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode(
                        (round(self.window_width), round(self.window_height)))
                clock = pygame.time.Clock()

                self.screen.fill((255, 255, 255))
                for line in self.lines:
                    pygame.gfxdraw.line(self.screen, *line,Colors.black())
                for key in self.resources:
                    self.resources[key].draw(pygame, self.screen)

                for key in self.spots:
                    self.spots[key].draw(pygame, self.screen)

                for key in self.roads:
                    self.roads[key].draw(pygame, self.screen)


                pygame.display.update()

        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)
