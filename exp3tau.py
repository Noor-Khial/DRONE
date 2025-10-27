import math
from probability import distr, draw


class Exp3TauAlgorithm:
    def __init__(self, numActions, gamma, tau, rewardMin=0.0, rewardMax=1.0, m_memory=0):
        """Mini-batched EXP3 with block size tau.

        Args:
            numActions (int): Number of actions/policies (N).
            gamma (float): Exploration parameter (mixing).
            tau (int): Block size; same action is executed for tau steps.
            rewardMin (float): Min possible reward (for normalization).
            rewardMax (float): Max possible reward (for normalization).
            m_memory (int): (Optional) environment memory parameter m (for policy-regret bound).
        """
        assert numActions >= 2, "numActions must be >= 2"
        assert tau >= 1, "tau must be >= 1"

        self.numActions = numActions
        self.gamma = gamma
        self.tau = tau
        self.m_memory = max(0, int(m_memory))

        # EXP3 state
        self.weights = [1.0] * numActions
        self.probabilityDistribution = distr(self.weights, self.gamma)
        self.choice = None

        # Reward scaling
        self.rewardMin = rewardMin
        self.rewardMax = rewardMax

        # Cumulative (per-step) stats
        self.t = 0  # elapsed steps
        self.cumulativeReward = 0.0
        self.bestActionCumulativeReward = 0.0

        # Cumulative (per-block) stats
        self.j = 0                 # completed blocks
        self.blockRewardSum = 0.0  # sum of unscaled rewards within current block
        self.blockBestRewardSum = 0.0

        # For importance weighting we need the block's action & prob
        self.block_action = None
        self.block_action_prob = None

        # Diagnostics
        self.weakRegret = 0.0
        self.regretBound = 0.0
        self.policyRegretBoundTau = 0.0  # τ-policy regret bound (high-level)

    # --------- Running the algorithm ---------
    def run(self):
        """Selects/returns an action for the current step.

        Returns:
            int: The selected action index.
        """
        # If starting a new block, sample a new action from current distribution
        if (self.t % self.tau) == 0:
            self.probabilityDistribution = distr(self.weights, self.gamma)
            self.choice = draw(self.probabilityDistribution)
            self.block_action = self.choice
            self.block_action_prob = self.probabilityDistribution[self.block_action]
            # reset block accumulators
            self.blockRewardSum = 0.0
            self.blockBestRewardSum = 0.0
        return self.block_action

    def update(self, t, reward, bestActionReward):
        """Consumes one step of feedback (within the current block).

        Args:
            t (int): Global step (1-based or 0-based—used only for logging/rates).
            reward (float): Realized reward for the executed action at this step.
            bestActionReward (float): Oracle best-action reward for this step (for regret tracking).

        Returns:
            tuple: (reward, estimated_block_reward_if_end)
        """
        # Update global step (allow external t to drive, but keep internal counter consistent)
        self.t += 1

        # Normalize step reward
        scaledReward = self._scale_reward(reward)
        scaledBest = self._scale_reward(bestActionReward)

        # Aggregate per-step stats
        self.cumulativeReward += scaledReward
        self.bestActionCumulativeReward += scaledBest

        # Aggregate block stats (unscaled sums, we will normalize consistently via same scaler)
        self.blockRewardSum += reward
        self.blockBestRewardSum += bestActionReward

        estimatedRewardForBlock = None

        # If block finished, do one EXP3 update with the block-average reward
        if (self.t % self.tau) == 0:
            self.j += 1
            # Block-average rewards (then scale)
            avgReward = self.blockRewardSum / self.tau
            avgBest = self.blockBestRewardSum / self.tau

            scaledAvgReward = self._scale_reward(avgReward)
            scaledAvgBest = self._scale_reward(avgBest)

            # Importance-weighted estimate uses the probability at block start
            p = max(self.block_action_prob, 1e-12)
            estimatedRewardForBlock = scaledAvgReward / p

            # EXP3 multiplicative weights update (one update per block)
            # (Note: /N factor mirrors your single-step code style)
            self.weights[self.block_action] *= math.exp((self.gamma / self.numActions) * estimatedRewardForBlock)

            # Refresh distribution for the next block (optional; also done at next run())
            self.probabilityDistribution = distr(self.weights, self.gamma)

            # Compute regrets/bounds (per-step weak regret + τ-policy bound diagnostics)
            self.calculate_regret(self.t)
            self._calculate_tau_policy_regret_bound()

            # Log (block granularity)
            self._log_block_state()

        return reward, estimatedRewardForBlock

    # --------- Regret / Bounds / Utilities ---------
    def calculate_regret(self, t):
        """Per-step weak regret and classic EXP3 bound (for comparison)."""
        # Keep your original normalization for weak regret
        self.weakRegret = (self.bestActionCumulativeReward - self.cumulativeReward) / (10 * t + 1)
        self.regretBound = 2 * math.sqrt(t * self.numActions * math.log(self.numActions))
        return self.weakRegret, self.regretBound

    def _calculate_tau_policy_regret_bound(self):
        """Diagnostics for τ-policy regret bound (high-level form).

        We mirror the reduction: Reg_pi(T) <= τ * R(T/τ) + (T m)/τ + τ
        with R(J) ≈ 2 √(J N log N) in your style.
        """
        if self.j == 0:
            self.policyRegretBoundTau = 0.0
            return self.policyRegretBoundTau

        J = max(1, self.t // self.tau)  # number of block updates so far
        N = self.numActions
        R_J = 2 * math.sqrt(J * N * math.log(N))  # same constant style as your RegretBound

        term1 = self.tau * R_J
        term2 = (self.t * self.m_memory) / max(1, self.tau)
        term3 = self.tau
        self.policyRegretBoundTau = term1 + term2 + term3
        return self.policyRegretBoundTau

    def get_regret(self):
        """Returns the current weak regret and classic regret bound."""
        return self.weakRegret, self.regretBound

    def get_tau_policy_regret_bound(self):
        """Returns the current τ-policy regret bound diagnostic."""
        return self.policyRegretBoundTau

    @staticmethod
    def suggest_tau(T, N, c=7, m_memory=0):
        """Heuristic from the theory: tau* ≈ (c N log N)^(-1/3) T^(1/3), then ensure > m.

        Args:
            T (int): Horizon in steps.
            N (int): Number of actions.
            c (float): Constant from analysis (default 7).
            m_memory (int): Environment memory parameter m.

        Returns:
            int: Suggested block size tau >= m+1.
        """
        base = (T ** (1.0 / 3.0)) / ((c * N * max(1.0, math.log(max(N, 2)))) ** (1.0 / 3.0))
        tau_star = max(1, int(math.ceil(base)))
        return max(tau_star, int(m_memory) + 1)

    # --------- Helpers ---------
    def _scale_reward(self, r):
        # Robust normalization; if degenerate, just return r (assumed already in [0,1])
        denom = (self.rewardMax - self.rewardMin)
        if denom <= 0:
            return r
        return (r - self.rewardMin) / denom

    def _log_block_state(self):
        try:
            with open('weights/EXP3_tau(3policies).txt', 'a') as f:
                w_probs = distr(self.weights, self.gamma)
                m = (
                    f"t: {self.t}\tJ: {self.j}\t"
                    f"weak_regret: {self.weakRegret:.4f}\t"
                    f"std_bound: {self.regretBound:.4f}\t"
                    f"tau_policy_bound: {self.policyRegretBoundTau:.4f}\t"
                    f"weights: ({', '.join([f'{w:.3f}' for w in w_probs])})"
                )
                f.write(m + '\n')
        except Exception:
            # Silent fail if logging path doesn't exist; keeps runtime simple on embedded devices
            pass
