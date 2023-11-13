#+AUTHOR: bdunahu
#+TITLE: multiplayer.py
#+DESCRIPTION: goal object for multiagent snake

import pygame as pg
from random import randint
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
GREEN = (0,128,43)

class Goal():
    def __init__(self, display, window_width=640, window_height=480, game_units=40):
        ''' create initial location '''
        self.location = None

        self.display = display
        self.window_width = window_width
        self.window_height = window_height
        self.game_units = game_units

    def reset(self, hazards=[]):
        ''' generate new coordinates for goal '''
        x = randint(0, (self.window_width-self.game_units )//self.game_units )*self.game_units 
        y = randint(0, (self.window_height-self.game_units )//self.game_units )*self.game_units
        self.location = Point(x, y)
        if self.location in hazards:
            self.reset(hazards)

    def draw(self):
        ''' draw rectangle directly on field '''
        pg.draw.rect(self.display, GREEN, pg.Rect(self.location.x, self.location.y,
      					     self.game_units, self.game_units))
