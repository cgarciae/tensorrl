import numpy as np
from scipy.special import softmax


class BoltzmannActorPolicy:
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """

    def __init__(self, from_logits=True):
        self.from_logits = from_logits

    def select_action(self, probs):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            probs (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert probs.ndim == 1
        probs = probs.astype(np.float32)
        nb_actions = probs.shape[0]

        if self.from_logits:
            probs = softmax(probs)

        action = np.random.choice(nb_actions, p=probs)

        return action


class MaxBoltzmannActorPolicy:
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """

    def __init__(self, eps=0.1, tau=1.0):
        super(MaxBoltzmannActorPolicy, self).__init__()
        self.eps = eps

    def select_action(self, probs):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            probs (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert probs.ndim == 1
        probs = probs.astype(np.float32)
        nb_actions = probs.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.choice(range(nb_actions), p=probs)
        else:
            action = np.argmax(probs)

        return action


class EBoltzmannActorPolicy:
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """

    def __init__(self, eps=0.1, tau=1.0):
        super(EBoltzmannActorPolicy, self).__init__()
        self.eps = eps

    def select_action(self, probs):
        """Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)

        # Arguments
            probs (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert probs.ndim == 1
        probs = probs.astype(np.float32)
        nb_actions = probs.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.choice(nb_actions)
        else:
            action = np.random.choice(nb_actions, p=probs)

        return action