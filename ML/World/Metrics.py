import numpy as np


class Accuracy:
    # An experience is a set of batch data that follows an action
    def add(self, exp):
        return exp.label == exp.action  # Gets appended to an epoch list

    # At the end of an epoch, a metric is tabulated
    def tabulate(self, epoch):
        return epoch  # Lists/arrays get concatenated and mean-averaged by default


class MSE:
    def add(self, exp):
        return (exp.label - exp.action) ** 2  # Gets appended to an epoch list

    def tabulate(self, epoch):
        return epoch  # Lists/arrays get concatenated and mean-averaged by default


class Reward:
    def add(self, exp):
        if 'reward' in exp:
            # Note: Taking mean batch-wise assumes each batch same size
            return exp.reward.mean() if hasattr(exp.reward, 'mean') else exp.reward

    def tabulate(self, episode):  # At the end of an episode, a metric is tabulated
        episode = [reward for reward in episode if reward is not None]

        if episode:
            return sum(episode)


class Precision:
    def add(self, exp):
        classes = np.unique(exp.action)

        # TODO use WHERE for the given class. Can't remember if on action or label
        true_positives = {c: (exp.action == exp.label) & (exp.action == c) for c in classes}
        total = {c: sum(exp.action == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average precision
        return sum([true_positives[key] for true_positives, total in epoch for key in true_positives]) \
            / sum([total[key] for true_positives, total in epoch for key in true_positives])


class Recall:
    def add(self, exp):
        classes = np.unique(exp.label)

        # TODO use WHERE for the given class. Can't remember if on action or label
        true_positives = {c: (exp.action == exp.label) & (exp.action == c) for c in classes}
        total = {c: sum(exp.label == c) for c in classes}

        return true_positives, total

    def tabulate(self, epoch):
        # Micro-average precision
        return sum([true_positives[key] for true_positives, total in epoch for key in true_positives]) \
            / sum([total[key] for true_positives, total in epoch for key in true_positives])


# metric.F1=2*precision*recall/(precision+recall)
