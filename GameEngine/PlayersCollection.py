#+AUTHOR: bdunahu
#+TITLE: PlayersCollection.py
#+DESCRIPTION: methods for handling and querying snake objects

import pygame as pg
from random import randint
from collections import namedtuple
from enum import Enum

class Direction(Enum):
    UP        = 0
    RIGHT     = 1
    DOWN      = 2
    LEFT      = 3

Point = namedtuple('Point', 'x, y')

YELLOW = (255,255,0)
RED =    (255,0,0)
PURPLE = (204,51,255)
WHITE =  (255,255,255)
PLAYER_COLOR = [YELLOW, RED, PURPLE, WHITE]

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
GAME_UNITS = 40
DISPLAY = None

class Players():
    def __init__(self, snake_size, num_players, display, s_starts, window_width=640, window_height=480, game_units=40):
        ''' define array list of new Snake objects '''
        global WINDOW_WIDTH
        global WINDOW_HEIGHT
        global GAME_UNITS
        global DISPLAY

        WINDOW_WIDTH = window_width
        WINDOW_HEIGHT = window_height
        GAME_UNITS = game_units
        DISPLAY = display

        self._index = 0
        self.num_players = num_players

        self.players = [Snake(snake_size, player_id, s_starts[i])
                        for i, player_id in enumerate(range(num_players))]

    def __iter__(self):
        return iter(self.players)

    def __getitem__(self, index):
        return self.players[index]

    def full_reset(self):
        ''' reset every snake position '''
        # map(lambda player:player.reset(), self.players)
        for player in self.players:
            player.reset()

    def move_all(self):
        ''' move all snakes '''
        # map(lambda player:player.move(), self.players)
        for player in self.players:
            player.move()

    def reward_killer(self, player):
        ''' split play length up against killing snakes '''
        killer = self._point_lookup(player.head)
        if not killer == None and not killer == player:
            rewards = player.score.curr_score + player.size
            killer.deficit += rewards
            killer.score.curr_score += rewards
            killer.score.total_score += rewards

    def _point_lookup(self, head):
        for player in self.players:
            if head in player.snake:
                return player
        return None

    def draw(self):
        ''' draw all snakes '''
        # map(lambda player:player.draw(), self.players)
        for player in self.players:
            player.draw()

class Snake():
    def __init__(self, initial_size, player_id, start):
        ''' define initial size (length), direction, and position '''
        self.player_id = player_id
        self.start = start
        self.size = initial_size
        self.direction = None
        self.head = None
        self.snake = []
        self.score = Score(self.player_id)
        self.deficit = 0				# for how many moves does this snake need to grow?

    def reset(self):
        self.score.reset()
        self.deficit = 0
        self.direction = Direction.RIGHT.value
        if self.start == None:
            x = randint(0, (WINDOW_WIDTH-GAME_UNITS )//GAME_UNITS )*GAME_UNITS 
            y = randint(0, (WINDOW_HEIGHT-GAME_UNITS )//GAME_UNITS )*GAME_UNITS
        else:
            x = self.start[0]
            y = self.start[1]
        self.head = Point(x,y)
        self.snake = [self.head]
        for seg in range(self.size-1):
      	    self.snake.append(Point(self.head.x-((seg+1)*GAME_UNITS), self.head.y))

    def move(self):
        ''' update snake coordinates by inserting new head '''
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT.value:
      	    x += GAME_UNITS
        if self.direction == Direction.LEFT.value:
      	    x -= GAME_UNITS                
        if self.direction == Direction.DOWN.value:
            y += GAME_UNITS                
        if self.direction == Direction.UP.value:
      	    y -= GAME_UNITS

        self.head = Point(x, y)
        self.snake.insert(0,self.head)

    def in_wall(self):
        return True if (self.head.x > WINDOW_WIDTH - GAME_UNITS or
            self.head.x < 0 or
            self.head.y > WINDOW_HEIGHT - GAME_UNITS or
            self.head.y < 0) else False
            
    def draw(self):
        ''' draw rectangle(s) directly on field '''
        for seg in self.snake:                      # see explanation in engine.org
      	    pg.draw.rect(DISPLAY, PLAYER_COLOR[self.player_id], pg.Rect(seg.x, seg.y,
      							                GAME_UNITS, GAME_UNITS))
        self.score.draw()

class Score():
    def __init__(self, player_id):
        ''' initialize score counter '''
        self.player_id = player_id
        self.font = pg.font.SysFont("monospace", 16)
        self.curr_score = 0
        self.total_score = 0
        self.deaths = 0
        self.kills = 0

    def scored(self):
        self.curr_score += 1
        self.total_score += 1

    def reset(self):
        self.curr_score = 0

    def draw(self):
        ''' draw score on top left '''
        score_surf = self.font.render(f'Current: {self.curr_score}   Total: {self.total_score}', True, PLAYER_COLOR[self.player_id])
        DISPLAY.blit(score_surf, (0, 0+24*self.player_id))
