from sklearn.inspection import permutation_importance
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mdi(forest):
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=forest.feature_names_in_)
    return forest_importances, std


def plot_mdi(forest_importances, std):
    fig, ax = plt.subplots()
    forest_importances.plot.bar(
        yerr=std, ax=ax, color="#03ef62", ecolor="#05192d")
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return fig, ax


def perm(forest, X_test, y_test, random_state=None, n_repeats=10, n_jobs=2):

    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )
    elapsed_time = time.time() - start_time
    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(
        result.importances_mean, index=forest.feature_names_in_)

    return forest_importances, result.importances_std


def plot_perm(forest_importances, std):
    fig, ax = plt.subplots()
    forest_importances.plot.bar(
        yerr=std, ax=ax, color="#03ef62", ecolor="#05192d")
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    return fig, ax
