#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm

import os

import constants as cs


#
# Creates and plots a standard_t distribution
#
# Args:
#  None
# Returns:
#  None
#
def plot_standard_t(df, size) -> None:
    v = np.random.default_rng().standard_t(df, size)
    plt.plot([i for i in range(size)], v, marker="o")
    plt.show()


def plot_dubuque_temp(data: pd.DataFrame, value: str) -> None:
    plt.plot(data.index, data[value], marker="o")
    plt.ylabel("Temperature (F)")
    plt.xlabel("Time (Months)")
    plt.show()


def main():
    # q1.5
    # plot_standard_t(5, 48)

    # q1.6
    tempdub = os.path.join(cs.DATASETS, "tempdub.dat")
    tempdub = pd.read_csv(tempdub)
    plot_dubuque_temp(tempdub, "tempdub")


if __name__ == "__main__":
    main()
