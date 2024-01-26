#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


#
# Creates and plots a standard_t distribution
#
# Args:
#  None
# Returns:
#  None
#
def plot_standard_t(df, length) -> None:
    df = 5
    length = 48
    v = np.random.default_rng().standard_t(df, length)
    plt.plot([i for i in range(length)], v, marker="o")
    plt.show()


def main():
    q1_5()


if __name__ == "__main__":
    main()
