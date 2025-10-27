import math
from probability import distr, draw

class UCBAlgorithm:
    def __init__(self, numActions, rewardMin=0, rewardMax=1):
        """Initializes the UCB algorithm parameters.

        Args:
            numActions (int): The number of possible actions.
            rewardMin (float): The minimum possible reward.
            rewardMax (float): The maximum possible reward.
        """
        self.numActions = numActions
        self.weights = [0] * numActions  # Action selection counts
        self.values = [0.0] * numActions  # Average rewards for each action
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax
        self.totalSteps = 0  # Total time steps
        self.cumulativeReward = 0
        self.bestActionCumulativeReward = 0
        self.weakRegret = 0
        self.regretBound = 0

    def run(self):
        """Selects an action using the UCB strategy.

        Returns:
            int: The selected action.
        """
        self.totalSteps += 1
        for action in range(self.numActions):
            if self.weights[action] == 0:
                return action  # Ensure each action is selected at least once

        ucb_values = [
            self.values[action] + math.sqrt(2 * math.log(self.totalSteps) / self.weights[action])
            for action in range(self.numActions)
        ]
        return ucb_values.index(max(ucb_values))

    def update(self, t, reward, bestActionReward):
        """Updates the UCB parameters after an action is taken.

        Args:
            action (int): The selected action.
            reward (float): The received reward.
            bestActionReward (float): The reward of the best possible action.

        Returns:
            float: The updated average reward for the selected action.
        """
        # Update counts and average reward for the selected action
        action = self.run()
        self.weights[action] += 1
        n = self.weights[action]
        self.values[action] += (reward - self.values[action]) / n

        # Update cumulative rewards
        scaledReward = (reward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        scaledBestActionRewad = (bestActionReward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        self.cumulativeReward += scaledReward
        self.bestActionCumulativeReward += scaledBestActionRewad
        self.calculate_regret(t)

        print(f"Action: {action}, Reward: {reward}, Estimated Reward: {self.values[action]:.3f}")

        return self.values[action], scaledReward

    def calculate_regret(self, t):
        """Calculates the weak regret and regret bound."""
        self.weakRegret = (self.bestActionCumulativeReward - self.cumulativeReward) / (t+1)
        self.regretBound = math.sqrt(2 * self.totalSteps * self.numActions * math.log(self.totalSteps))

        # Log the results
        with open('weights/UCB(3policies).txt', 'a') as f:
            m = f"regret: {self.weakRegret:.3f}\tregret bound: {self.regretBound:.3f}\tcounts: ({', '.join([f'{value:.3f}' for value in distr(self.weights)])})"
            f.write(m + '\n')

    def get_regret(self):
        """Returns the current weak regret and regret bound.

        Returns:
            tuple: The weak regret and regret bound.
        """
        return self.weakRegret, self.regretBound
