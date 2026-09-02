import random

import numpy as np

from .make_env import make_env
from prettyNEAT import *


class GymTask:
    """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
    """

    def __init__(self, game, paramOnly=False, nReps=1, budget=50_000, gif_file='', time=0):
        """Initializes task environment

        Args:
          game - (string) - dict key of task to be solved (see domain/config.py)

        Optional:
          paramOnly - (bool)  - only load parameters instead of launching task?
          nReps     - (nReps) - number of trials to get average fitness
        """
        if len(game.i_act) != game.input_size:
            raise ValueError(
                f"Invalid activation configuration for {game.env_name}: "
                f"input_size is {game.input_size}, but i_act has "
                f"{len(game.i_act)} entries."
            )
        if len(game.o_act) != game.output_size:
            raise ValueError(
                f"Invalid activation configuration for {game.env_name}: "
                f"output_size is {game.output_size}, but o_act has "
                f"{len(game.o_act)} entries."
            )

        # Network properties
        self.nInput = game.input_size
        self.nOutput = game.output_size
        self.actRange = game.h_act
        self.absWCap = game.weightCap
        self.layers = game.layers
        self.activations = np.r_[np.full(1, 1), game.i_act, game.o_act]

        # Environment
        self.nReps = nReps
        self.maxEpisodeLength = game.max_episode_length
        self.actSelect = game.actionSelect
        if not paramOnly:
            self.env = make_env(game.env_name)

        # == EA-elective-NEAT ==========================================================================================
        self.is_minatar = game.env_name.startswith("minatar:")
        self.images = []

        self.curr_eval = 0
        self.budget = budget
        self.gif_file = gif_file if gif_file.endswith(".gif") else gif_file + ".gif"
        self.gif_file = self.gif_file[:-4] + f"{random.randint(0, 0xffffffff)}.gif"
        self.time=time
        # ==============================================================================================================

        # Special needs...
        self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))

    def getFitness(self, wVec, aVec, hyp=None, view=False, nRep=False, seed=-1):
        """Get fitness of a single individual.

        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          nReps   - (nReps)    - number of trials to get average fitness
          seed    - (int)      - starting random seed for trials

        Returns:
          fitness - (float)    - mean reward over all trials
        """
        if nRep is False:
            nRep = self.nReps
        wVec[np.isnan(wVec)] = 0
        reward = np.empty(nRep)
        for iRep in range(nRep):
            reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed + iRep)
            self.curr_eval += 1
            if self.curr_eval >= self.budget:
                break
        fitness = np.mean(reward)

        # == EA-elective-NEAT ==========================================================================================
        if view:
            directory = os.path.join(*self.gif_file.split('/')[:-1])
            if not os.path.exists(directory):
                os.makedirs(directory)

            print("saving gif file inside", self.gif_file)
            self.images[0].save(self.gif_file, save_all=True, append_images=self.images[1:], optimize=False, duration=1000 // 1000, loop=0)
        # ==============================================================================================================

        return fitness

    def testInd(self, wVec, aVec, view=False, seed=-1):
        """Evaluate individual on task
        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          seed    - (int)      - starting random seed for trials

        Returns:
          fitness - (float)    - reward earned in trial
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
        # == EA-elective-NEAT ==========================================================================================
            if not self.is_minatar:
                self.env.seed(seed)
        state = self.env.reset()
        # ==============================================================================================================
        self.env.t = 0
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)

        # wVec[wVec != 0]
        predName = str(np.mean(wVec[wVec != 0]))
        # == EA-elective-NEAT ==========================================================================================
        state, reward, done, info = self.env.step(action)
        # ==============================================================================================================

        if self.maxEpisodeLength == 0:
            if view:
                self.render(done)
            return reward
        else:
            totalReward = reward

        for tStep in range(self.maxEpisodeLength):
            annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)
            # == EA-elective-NEAT ======================================================================================
            state, reward, done, info = self.env.step(action)
            # ==========================================================================================================
            totalReward += reward
            if view:
                self.render(done)
            if done:
                break
        return totalReward

    def render(self, done):
        if self.needsClosed:
            image = self.env.render(close=done)
        else:
            image = self.env.render(time=self.time)
        self.images.append(image)
