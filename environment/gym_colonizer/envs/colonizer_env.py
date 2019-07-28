import gym
import random
import torch

from gym_colonizer.envs.field_types import CORN, WOOD, SHEEP, CLAY, IRON, DESERT
from math import sqrt
from gym import error
import uuid
from gym_colonizer.envs.colors import *
from gym_colonizer.envs.field_types import WOOD, CLAY, CORN, DESERT, IRON, SHEEP
import sys
import numpy as np


VILLAGE = 'Village'
TOWN = 'Town'


def itervalues(d):
    return iter(getattr(d, ('itervalues', 'values')[sys.version_info[0] > 2])())


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_color(res):
    res_color_dict = {
        CORN: YELLOW,
        WOOD: GREEN,
        SHEEP: SHEEP_GREY,
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

#    for i in range(9):
#        resources.append(WOOD)
#        resources.append(CLAY)

    if shuffle:
        random.shuffle(resources)
    return resources


class Resource():
    def __init__(self, x, y, rating, resource):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 25
        self.id = uuid.uuid4()
        self.close_spot = []
        self.rating = rating
        self.resource = resource
        self.color = get_color(self.resource)

    def draw(self, pygame, screen):
        pygame.gfxdraw.filled_circle(screen, round(
            self.pos.x), round(self.pos.y), self.radius, self.color)

        font = pygame.font.SysFont("arial", 25)
        text = font.render(str(self.rating), 3, (0, 0, 255))
        screen.blit(text, (self.pos.x-10, self.pos.y-10))


class Spot():
    def __init__(self, x, y):
        self.pos = dotdict({'x': x, 'y': y})
        self.radius = 5
        self.color = LIGHT_GRAY
        self.id = uuid.uuid4()
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
        font = pygame.font.SysFont("arial", 15)
        text = font.render(str(self.rating), 3, (0, 0, 255))
        screen.blit(text, (self.pos.x-10, self.pos.y-10))

    def reset(self):
        self.owner = 0
        self.color = LIGHT_GRAY


class Road():
    def __init__(self, x1, y1, x2, y2, s1, s2):
        self.pos = dotdict({'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2)})
        self.radius = 3
        self.color = WHITE
        self.id = uuid.uuid4()
        self.close_spot = [s1, s2]
        self.owner = 0
        self.rating = 10

    def draw(self, pygame, screen):
        pygame.gfxdraw.line(screen, *self.pos.values(), self.color)

    def set_owner(self, owner):
        if self.owner is 0:
            self.owner = owner
            if owner == 1:
                self.color = BLACK
            return True
        else:
            return False

    def reset(self):
        self.owner = 0
        self.color = WHITE


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
        self.round = 1

        self.sheep = 0
        self.clay = 0
        self.wood = 0
        self.iron = 0
        self.corn = 0
        self.collected_resources = self.init_collected_resources()

        self.used_spots = []
        self.used_roads = []
        self.episode = 0
        self.iteration = 0

    def init_collected_resources(self):
        return {
            CORN: 0,
            WOOD: 0,
            SHEEP: 0,
            CLAY: 0,
            IRON: 0
        }

    def get_state(self):
        state = []
        for s in self.spots:
            state.append(self.spots[s].owner)

        for s in self.roads:
            state.append(self.roads[s].owner)

        state.append(int(self.can_build_road()))
        state.append(int(self.can_build_village()))
        state.append(self.iteration)

        return np.array(state)

    def get_spot_rating_state(self):
        state = []
        for s in self.spots:
            state.append(self.spots[s].rating)
        return state

    def initMap(self):
        self.add_spots()
        self.add_roads_and_resources()
        self.add_relations()
        self.add_spot_rating()

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
                                    s2.pos.y, s1, s2)
                        self.roads[road.id] = road
                        used_positions.append([x, y])
                        self.spots[key_s1].close_road.append(road)
                        self.spots[key_s2].close_road.append(road)

                    # Resources
                    if dist == 105:
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
                    self.spots[s].close_resource.append(res)
                    self.resources[r].close_spot.append(spot.id)

    def add_spot_rating(self):
        for s in self.spots:
            comul_rating = 0
            for r in self.spots[s].close_resource:
                if r.rating == 7:
                    comul_rating += 0
                else:
                    comul_rating += get_rating(r.rating)
            self.spots[s].rating = comul_rating

    def reset(self):
        self.screen = None
        for x in self.spots:
            self.spots[x].reset()
        for x in self.roads:
            self.roads[x].reset()
        self.used_spots = []
        self.used_roads = []
        self.round = 1
        self.episode += 1
        self.iteration = 0
        self.collected_resources = self.init_collected_resources()
        return self.get_state()

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
                # for line in self.lines:
                #    pygame.gfxdraw.line(self.screen, *line, BLACK)
                for key in self.resources:
                    self.resources[key].draw(pygame, self.screen)

                for key in self.spots:
                    self.spots[key].draw(pygame, self.screen)

                for key in self.roads:
                    self.roads[key].draw(pygame, self.screen)

                i = 0
                for key in self.collected_resources:
                    font = pygame.font.SysFont("arial", 25)
                    text = font.render(
                        str(key[:2])+str(self.collected_resources[key]), 3, (0, 0, 255))
                    self.screen.blit(text, (400, 20*i))
                    i += 1

                font = pygame.font.SysFont("arial", 25)
                text = font.render(str(self.episode), 3, (0, 0, 255))
                self.screen.blit(text, (10, 10))

                pygame.display.update()

        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def distribute_resources(self):
        res_sum = 0
        for s in self.used_spots:
            for r in s.close_resource:
                if not r.resource == DESERT:
                    res_sum += r.rating
                    self.collected_resources[r.resource] += r.rating
        return res_sum

    def step(self, action, t):
        self.distribute_resources()
        reward = self.build_resource(action, t)
        if reward > 0 and t > 2:
            self.remove_resources_for(action)
        self.iteration += 1
        return self.get_state(), reward, False, {}

    def build_resource(self, action, t):
        if action == 126:
            return 0
        build = self.get_build_from_action(action)
        if isinstance(build, Spot):
            # need 2 roads between villages
            for r in build.close_road:
                for spot in r.close_spot:
                    if spot.id is not build.id:
                        if spot.owner == 1:
                            return -1
            if not self.spot_connected(build.close_road) and t > 2:
                return -1
            if not self.can_build_village() and t > 2:
                return 0
            valid = build.set_owner(1)
            if valid:
                self.used_spots.append(build)
                return 2
            else:
                return -1

        if isinstance(build, Road):
            if not self.road_connected(build.close_spot):
                return -1
            if not self.can_build_road() or t == 2 or t == 4:
                return 0
            valid = build.set_owner(1)

            if valid:
                self.used_roads.append(build)
                return 1
            else:
                return -1

    def get_build_from_action(self, action):
        if action <= len(self.spots): # villages
            return self.get_spot_by_index(action)
        else:  #roads
            return self.get_road_by_index(action-len(self.spots))

    def get_spot_by_index(self, i):
        if i < len(self.spots):
            cnt = 0
            for spot_id in self.spots.keys():
                if cnt == i:
                    return self.spots[spot_id]
                cnt += 1
        else:
            i -= len(self.spots)
            cnt = 0
            for road_id in self.roads.keys():
                if cnt == i:
                    return self.roads[road_id]
                cnt += 1

    def get_road_by_index(self, i):
        cnt = 0
        for road_id in self.roads.keys():
            if cnt == i:
                return self.roads[road_id]
            cnt += 1

    # not used atm
    def filter_legal_actions(self, actions, step):
        can_build_village = self.can_build_village()
        can_build_road = self.can_build_road()
        actions = actions[0]
        for i, val in enumerate(actions):
            if i < 126:
                s = self.get_build_from_action(i)
                if s.owner == 1:  # build already
                    actions[i] = 0
                else:  # free spot or road
                    if i < len(self.spots):  # Spots
                        if step > 2:  # first two steps can build everywhere and are free
                            if not can_build_village:
                                actions[i] = 0
                            else:
                                connected_to_road = self.spot_connected(
                                    self.spots[s.id].close_road)
                                if not connected_to_road:
                                    actions[i] = 0

                    if i >= len(self.spots):  # Roads
                        # cant build roads in first two steps
                        if not can_build_road or step < 2:
                            actions[i] = 0
                        else:
                            connected = self.road_connected(
                                self.roads[s.id].close_spot)
                            if not connected:
                                actions[i] = 0
        return actions.unsqueeze(0)

    def spot_connected(self, close_road):
        for r in close_road:
            if r.owner == 1:
                return True
        return False

    def road_connected(self, close_spot):
        for spot in close_spot:
            if spot.owner == 1:
                return True
            for r in spot.close_road:
                if r.owner == 1:
                    return True
        return False

    def can_build_village(self):
        cost = 1
        return self.collected_resources[WOOD] > cost and self.collected_resources[CLAY] > cost and self.collected_resources[SHEEP] > cost and self.collected_resources[CORN] > cost

    def can_build_road(self):
        cost = 10
        return self.collected_resources[WOOD] > cost and self.collected_resources[CLAY] > cost

    def remove_resources_for(self, action):
        if action != 128:
            if(action > len(self.spots)):
                self.remove_resources_for_village()
            elif(action < (len(self.spots)+len(self.roads))):
                self.remove_resources_for_road()

    def remove_resources_for_village(self):
        cost = 1
        self.collected_resources[WOOD] -= cost
        self.collected_resources[CLAY] -= cost
        self.collected_resources[SHEEP] -= cost
        self.collected_resources[CORN] -= cost

    def remove_resources_for_road(self):
        cost = 10
        self.collected_resources[WOOD] -= cost
        self.collected_resources[CLAY] -= cost


def get_rating(n):
    n = n-7
    n = n*-1 if n < 0 else n
    n -= 5
    n *= -1
    return n
