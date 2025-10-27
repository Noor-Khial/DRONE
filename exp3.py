import math
from probability import distr, draw


class Exp3Algorithm:
    def __init__(self, numActions, gamma, rewardMin=0, rewardMax=1):
        """Initializes the Exp3 algorithm parameters.

        Args:
            numActions (int): The number of possible actions.
            gamma (float): The exploration parameter.
            rewardMin (float): The minimum possible reward.
            rewardMax (float): The maximum possible reward.
        """
        self.numActions = numActions
        self.gamma = gamma
        self.weights = [1.0] * numActions
        self.cumulativeReward = 0
        self.bestActionCumulativeReward = 0
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax
        self.biases = [0.0] * numActions
        self.weakRegret = 0 
        self.regretBound = 0

    def run(self):
        """Performs one step of the Exp3 algorithm.

        Args:
            t (int): The time step or iteration number.
            agent_location (int): The current location of the agent.

        Returns:
            tuple: The selected action, the received reward, and the estimated reward.
        """
        # Get the probability distribution based on weights
        self.probabilityDistribution = distr(self.weights, self.gamma)
        self.choice = draw(self.probabilityDistribution)

        return self.choice
    
    def update(self, t, reward, bestActionRewad):
        # Get the reward based on the new location
        scaledReward = (reward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        
        # Calculate the estimated reward
        estimatedReward = scaledReward / self.probabilityDistribution[self.choice]

        # Update the weights based on the estimated reward
        self.weights[self.choice] *= math.exp(estimatedReward * self.gamma / self.numActions)  

        # Update regret 
        scaledBestActionRewad = (bestActionRewad - self.rewardMin) / (self.rewardMax - self.rewardMin)
        self.bestActionCumulativeReward +=  scaledBestActionRewad
        self.cumulativeReward += scaledReward
        self.calculate_regret(t)
        return reward, estimatedReward

    
    def calculate_regret(self, t):
        """Calculates the weak regret and regret bound.

        Args:
            t (int): The current time step.

        Returns:
            tuple: The weak regret and regret bound.
        """
        self.weakRegret = (self.bestActionCumulativeReward - self.cumulativeReward) #/ (10* t+1)
        self.regretBound = 2 * math.sqrt(t * self.numActions * math.log(self.numActions))
    
        # log the results of the meta learner 
        with open('weights/EXP3(3policies).txt', 'a') as f:
            m = f"regret: {self.weakRegret:.3f}\tregret bound: {self.regretBound:.3f}\tweights: ({', '.join([f'{weight:.3f}' for weight in distr(self.weights)])})"
            f.write(m + '\n')

        return self.weakRegret/(t+1), self.regretBound/(t+1)

    def get_regret(self):
        """Returns the current weak regret and regret bound.

        Returns:
            tuple: The weak regret and regret bound.
        """
        return self.weakRegret, self.regretBound
