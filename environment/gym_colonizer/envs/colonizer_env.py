import gym
import random
import torch


from gym_colonizer.envs.field_types import CORN, WOOD, SHEEP, CLAY, IRON, DESERT
from math import sqrt
from gym import error
import uuid
from gym_colonizer.envs.colors import *
from gym_colonizer.envs.field_types import WOOD, CLAY, CORN, DESERT, IRON, SHEEP


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_color(res):
    res_color_dict = {
        CORN: YELLOW,
        WOOD: GREEN,
        SHEEP: WHITE,
        CLAY: ORANGE,
        IRON: GRAY,
        DESERT: BLACK
    }
    return res_color_dict.get(res, "Invalid resource!")


def get_resources(shuffle=True):
    resources = []
    for i in range(4):
        resources.append(WOOD)
        resources.append(SHEEP)
        resources.append(CORN)
    for i in range(3):
        resources.append(IRON)
        resources.append(CLAY)

    if shuffle:
        random.shuffle(resources)
    return resources


class Spot():
    def __init__(self, x, y):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 5
        self.color = LIGHT_GRAY
        self.id = uuid.uuid1()
        self.close_resource = []
        self.close_road = []
        self.owner = 0

    def set_owner(self, owner):
        if self.owner is 0:
            self.owner = owner
            if owner == 1:
                self.color = RED
            return True
        else:
            return False

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)


class Resource():
    def __init__(self, x, y, rating, resource):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 25
        self.id = uuid.uuid1()
        self.close_spot = []
        self.rating = rating
        self.resource = resource
        self.color = get_color(self.resource)
        print(self.color)

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)

        font = pygame.font.SysFont("arial", 25)
        text = font.render(str(self.rating), 3, (0, 0, 255))
        screen.blit(text, (self.pos.x-10, self.pos.y-10))


class Road():
    def __init__(self, x1, y1, x2, y2, s1_id, s2_id):
        self.pos = dotdict({'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2)})
        self.radius = 3
        self.color = (100, 200, 100)
        self.id = uuid.uuid1()
        self.close_spot = [s1_id, s2_id]

    def draw(self, pygame, screen):
        pygame.gfxdraw.line(screen, *self.pos.values(), self.color)


class ColonizerEnv(gym.Env):
    # TODO: Add seeding function and fps lock
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second': 350}

    def __init__(self):
        self.window_height = 500
        self.window_width = 500
        self.spots = {}
        self.roads = {}
        self.resources = {}
        self.lines = []
        self.initMap()

    def get_state(self):
        state = []
        for r in self.resources:
            state.append(self.resources[r].rating)

        for s in self.spots:
            state.append(self.spots[s].owner)

        return state

    def initMap(self):
        self.add_spots()
        self.add_roads_and_resources()
        self.add_relations()

    def add_spots(self):
        mid_y = self.window_height/1.1
        mid_x = self.window_width/4.5

        size = 80
        distance = 20
        height = 13
        amount = 4

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

    def add_roads_and_resources(self):
        used_positions = []

        rating = [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        resources = get_resources()

        random.shuffle(rating)
        print('312 ', len(self.spots))
        for key_s1 in self.spots:
            s1 = self.spots[key_s1]
            for key_s2 in self.spots:
                s2 = self.spots[key_s2]
                if s1 == s2:
                    continue
                else:
                    dist = round(sqrt((s2.pos.x - s1.pos.x) **
                                      2 + (s2.pos.y - s1.pos.y)**2))
                    x = (s1.pos.x+s2.pos.x)/2
                    y = (s1.pos.y+s2.pos.y)/2
                    if [x, y] in used_positions:
                        continue

                    # Roads
                    if dist == 53 or dist == 35:
                        road = Road(s1.pos.x, s1.pos.y, s2.pos.x,
                                    s2.pos.y, s1.id, s2.id)
                        self.roads[road.id] = road
                        used_positions.append([x, y])

                    # Resources
                    if dist == 105:
                        print(len(rating))
                        r = rating.pop()
                        if r == 7:
                            resource = Resource(x, y, r, DESERT)
                        else:
                            resource = Resource(x, y, r, resources.pop())
                        self.resources[resource.id] = resource
                        used_positions.append([x, y])

    def add_relations(self):
        # relation between spots and resources
        for s in self.spots:
            spot = self.spots[s]
            for r in self.resources:
                res = self.resources[r]
                dist = round(sqrt((spot.pos.x - res.pos.x) **
                                  2 + (spot.pos.y - res.pos.y)**2))
                if dist == 52 or dist == 44:
                    self.spots[s].close_resource.append(res.id)
                    self.resources[r].close_spot.append(spot.id)
                    self.lines.append([int(spot.pos.x), int(
                        spot.pos.y), int(res.pos.x), int(res.pos.y)])

    def reset(self):
        self.screen = None
        self.spots = {}
        self.roads = {}
        self.resources = {}
        self.initMap()

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
                    pygame.gfxdraw.line(self.screen, *line, BLACK)
                for key in self.resources:
                    self.resources[key].draw(pygame, self.screen)

                for key in self.spots:
                    self.spots[key].draw(pygame, self.screen)

                for key in self.roads:
                    self.roads[key].draw(pygame, self.screen)

                pygame.display.update()

        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def step(self, action):
        cnt = 0
        for key in self.spots.keys():
            if cnt == action:
                if self.spots[key].set_owner(1):
                    # worked
                    #print('legal move')
                    return 10  # return reward based on surrounding resources rating
                else:
                    #print('illegal move')
                    # illegal action
                    return -1
            cnt += 1
#
        #d1 = random.randint(1,6)
        #d2 = random.randint(1,6)
#
        #n = d1+d2
        # print('dice',n)
#
