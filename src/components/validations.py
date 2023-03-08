import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

class MinLoss:
    def __init__(self):
        self.value = np.nan

    def update(self, min_loss):
        self.value = np.nanmin([self.value, min_loss])

class ValidResult:
    def __init__(self):
        self.values = None

    def append(self, values):
        if self.values is None:
            self.values = values
            return self.values
        self.values = np.concatenate([self.values, values])
        return self.values

class ConfusionMatrix:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.fig = plt.figure(figsize=[4, 4], tight_layout=True)

    def draw(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)

        df_confmat = pd.DataFrame(
            confusion_matrix(idx_probs, idx_labels),
            index=self.config["label"],
            columns=self.config["label"]
        )
        axis = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(df_confmat, ax=axis, cmap="bwr", square=True, annot=True)
        axis.set_xlabel("label")
        axis.set_ylabel("pred")
        return self.fig

class F1Score:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.f1_scores = {
            "macro": None,
            "micro": None
        }

    def calc(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)
        self.f1_scores = {
            "macro": f1_score(idx_probs, idx_labels, average="macro"),
            "micro": f1_score(idx_probs, idx_labels, average="micro")
        }
        return self.f1_scores

class LogLoss:
    def __init__(self, probs, labels, config):
        # const
        self.probs = probs
        self.labels = labels
        self.config = config
        self.prob_min = 10**(-15)
        self.prob_max = 1-10**(-15)

        # variables
        self.logloss = np.nan

    def calc(self):
        norm_probs = self.probs / np.sum(self.probs, axis=1)[:, None]
        log_probs = np.log(np.clip(norm_probs, self.prob_min, self.prob_max))
        self.logloss = -np.mean(np.sum(self.labels * log_probs, axis=1))
        return self.logloss
