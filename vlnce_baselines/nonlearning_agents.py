import json
from collections import defaultdict

import numpy as np
from habitat import Env, logger
from habitat.config.default import Config
from habitat.core.agent import Agent
# from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm, trange

# from vlnce_baselines.common.environments import VLNCEInferenceEnv


class RandomAgent(Agent):
    """Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class HandcraftedAgent(Agent):
    """Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}
