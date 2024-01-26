#!/usr/bin/env python

# external libraries
import pandas as pd
import matplotlib.pyplot as plt

# standard libraries
import os


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASETS = os.path.join(PROJECT_ROOT, "datasets")

    df = pd.read_csv(
        os.path.join(DATASETS, "GlobTemp.txt"), header=None, dtype=float, names=["temp"]
    )
    plt.plot(df.index, df["temp"], marker="o")
    plt.title("Global temperatures")
    plt.ylabel("temp")
    plt.xlabel("time (months)")
    plt.show()


if __name__ == "__main__":
    main()
