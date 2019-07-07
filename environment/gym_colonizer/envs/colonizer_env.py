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
    def __init__(self, x=0, y=0):
        self.pos = dotdict({'x':x,'y':y})
        self.radius = 5
        self.color = Colors.gray()
        self.id = uuid.uuid1()

    def draw(self,pygame,screen):
        pygame.gfxdraw.filled_circle(screen, round(self.pos.x), round(self.pos.y), self.radius, self.color)

class Resource():
    def __init__(self, x=0, y=0):
        self.pos = dotdict({'x':x,'y':y})
        self.radius = 5
        self.color = Colors.yellow()
        self.id = uuid.uuid1()

    def draw(self,pygame,screen):
        pygame.gfxdraw.filled_circle(screen, round(self.pos.x), round(self.pos.y), self.radius, self.color)

class Road():
    def __init__(self, x=0, y=0):
        self.pos = dotdict({'x':x,'y':y})
        self.radius = 5
        self.color = (100, 200, 100)
        self.id = uuid.uuid1()

    def draw(self,pygame,screen):
        pygame.gfxdraw.filled_circle(screen, round(self.pos.x), round(self.pos.y), self.radius, self.color)


class ColonizerEnv(gym.Env):
    # TODO: Add seeding function and fps lock
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second':350}

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

        self.initMap()

    def initMap(self):

        distance = 20

        amount = 4
        mid_y = self.window_height/1.5
        size = 80
        mid_x = self.window_width/2.5
        switch = False
        for y in range(1,13):
            if(y%2==0):
                if(y < 7):
                    amount+=1
                    if(y %2 == 0):
                        mid_x -= size/2
                else:
                    amount-=1
                    if(y %2 == 0):
                        mid_x += size/2
            for x in range(1,amount):
                s = Spot(mid_x+x*size,mid_y-y*size/2)
                self.spots[s.id] = s
        print(len(self.spots))

        for ks1 in self.spots:
            s1 = self.spots[ks1]
            for ks2 in self.spots:
                s2 = self.spots[ks2]
                if s1 == s2:
                    continue
                else:
                    dist =  sqrt((pow(s1.pos.x-s2.pos.x,2)+pow(s1.pos.y-s1.pos.y,2)))
                    if dist == 0:
                        x = (s1.pos.x+s2.pos.x)/2
                        y = (s1.pos.y+s2.pos.y)/2
                        road = Road(x,y)
                        self.roads[road.id] = road

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
                for key in self.resources:
                    self.resources[key].draw(pygame,self.screen)

                for key in self.spots:
                    self.spots[key].draw(pygame,self.screen)

                for key in self.roads:
                    self.roads[key].draw(pygame,self.screen)

                pygame.display.update()
                time.sleep(100)

        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)