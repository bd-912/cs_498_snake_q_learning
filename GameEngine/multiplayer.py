#+AUTHOR: bdunahu
#+TITLE: multiplayer.py
#+DESCRIPTION game engine for multiagent snake

from enum import Enum
import pygame as pg
import numpy as np

from GameEngine import PlayersCollection
from GameEngine import GoalCollection

BLACK = (0, 0, 0)

class CollisionType(Enum):                              # each of these means different outcome for model
    DEATH = 0
    GOAL =  1
    NONE =  2


class Playfield:
    def __init__(self, window_width=640, window_height=480, units=40, g_speed=25, s_size=3):
        ''' initialize pygame modules, snake, goal, and score objects '''
        global DISPLAY
                        
        self._g_speed = g_speed                          # game speed
        self._s_size = s_size                            # initial snake size
        self._units = units
        self._window_width = window_width
        self._window_height = window_height
        self._player_count = -1   			 # number of registered players
        self._player_starts = []                         # starting player positions
        ''' for human feedback '''
        self._game_state = False                         # false is game over
        self._draw_on = True
        self._clock = pg.time.Clock()

        ''' objects '''
        pg.init()
        self.display = pg.display.set_mode(
            [self._window_width, self._window_height],
            pg.HWSURFACE)              			 # display object (see explanation in engine.org)
        self._players = None
        self._goal = None

    def add_player(self, s_start=None):
        '''
        Returns the player's number to the callee.
        If the player count is over four, returns None
        '''
        if self._player_count < 4 or self._game_state == True:
            self._player_starts.append(s_start)
            self._player_count += 1
            return self._player_count
        return None

    def get_heads_tails_and_goal(self):
        '''
        Returns an array of heads, an array of tail positions,
        and the goal position
        '''
        heads = []
        for player in self._players.players:
            heads.append(player.head)
        return heads, self._get_player_bodies(), self._goal.location

    def get_viable_actions(self, player_id):
        '''
        Given a player's id,
        returns a list of actions that does
        not result in immediate death
        '''
        head = self._players.players[player_id].head
        # tail = self._players.players[player_id].snake
        tail = self._get_player_bodies()
        danger_array = np.array([
            head.y-self._units <  0                   or PlayersCollection.Point(head.x, head.y-self._units) in tail, # up
            head.x+self._units >= self._window_width  or PlayersCollection.Point(head.x+self._units, head.y) in tail, # right
            head.y+self._units >= self._window_height or PlayersCollection.Point(head.x, head.y+self._units) in tail, # down
            head.x-self._units < 0                    or PlayersCollection.Point(head.x-self._units, head.y) in tail, # left
        ])

        return np.where(danger_array == False)[0]

    def start_game(self):
        '''
        Initializes player objects, starts the game
        '''
        self._players = PlayersCollection.Players(self._s_size, self._player_count+1, self.display, self._player_starts, window_width=self._window_width, window_height=self._window_height, game_units=self._units)
        self._goal = GoalCollection.Goal(self.display, self._window_width, self._window_height, game_units=self._units)
        self._reset()
        print(f'Game starting with {self._player_count+1} players.')

    def stop_game(self):
        '''
        Restarts the game, allows adding/removing players
        '''
        self._game_state = False
        print(f'Game over!')

    def cleanup(self):
        ''' end game session '''
        pg.quit()

    def player_advance(self, actions, noise=0.0):
        ''' given a list of snake actions '''
        ''' return a list of results, and '''
        ''' update game state             '''

        for player_id, action in enumerate(actions):
            if np.random.uniform() < noise:
                # random action (noise)
                random_choice = [0, 1, 2, 3]
                random_choice.remove(self._players[player_id].direction)
                self._players[player_id].direction = np.random.choice(random_choice)
            else:
                self._players[player_id].direction = action

        self._players.move_all()
        collisions = self._check_player_collisions()
        if self._draw_on:
            self._update_ui()
        return collisions

    def toggle_draw(self):
        ''' turns off and on UI '''
        self._draw_on = not self._draw_on
        print(f'Draw is now {self._draw_on}.')

    def _check_player_collisions(self):
        results = []
        for player in self._players:
            ''' determine what obstacle was hit '''
            hazards = self._get_player_bodies()
            hazards.insert(0, 0)			# FIXME (why is this required?)
            hazards.remove(player.head)
            if (player.head in hazards or player.in_wall()):
                self._players.reward_killer(player)
                player.reset()
                results.append(CollisionType.DEATH)
            elif player.head == self._goal.location:
                player.score.curr_score += 1
                player.score.total_score += 1
                self._goal.reset(hazards)
                results.append(CollisionType.GOAL)
            else:
                if player.deficit <= 0:
                    player.snake.pop()
                else:
                    player.deficit -= 1
                results.append(CollisionType.NONE)
        return results

    def _get_player_bodies(self):
        ''' return an array of all tail coordinates '''
        tails = []
        for player in self._players:
            tails += player.snake
        return tails

    def _update_ui(self):
        ''' flush new positions to screen '''
        self.display.fill(BLACK)
        self._players.draw()
        self._goal.draw()

        pg.display.flip()                               # full screen update
        self._clock.tick(self._g_speed)

    def _reset(self):
        ''' reset game state     '''
        ''' required by training '''
        self._game_state = True                         # false is game over
        self._goal.reset()
        self._players.full_reset()
