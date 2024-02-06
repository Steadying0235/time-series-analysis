#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import os
import constants as cs


#
# plot a time series data with monthly markers to determine seasonal trends
#
# Args:
#    dates  (type): timestamps of data collection
#    values (type): data values
#    xlabel (str) : plot x axis label
#    ylabel (str) : plot y axis label
#    title  (str) : plot title
#
# Returns:
#    None
#
def plot_monthly(
    dates: pd.DataFrame, values: pd.DataFrame, xlabel: str, ylabel: str, title: str
) -> None:
    # plot time series data
    plt.figure()
    plt.plot(
        dates,
        values,
        marker="o",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # plot monthly markers
    # monthly markers more clearly show the seasonality of the data
    for d, v in zip(dates, values):
        marker = r"$\rm{" + d.strftime("%b") + "}$"
        plt.plot(d, v, marker=marker, markersize=12)
    plt.show()


#
# plot time series data
#
# Args:
#    dates  (type): timestamps of data collection
#    values (type): data values
#    xlabel (str) : plot x axis label
#    ylabel (str) : plot y axis label
#    title  (str) : plot title
#
# Returns:
#    None
#
def plot_ts(
    dates: pd.DataFrame, values: pd.DataFrame, xlabel: str, ylabel: str, title: str
) -> None:

    plt.figure()
    plt.plot(
        dates,
        values,
        marker="o",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def main():

    # problem 3.4
    # read the hours dataset into memory
    hours = pd.read_csv(os.path.join(cs.DATASETS, "hours.dat"))
    # create timestamps for the data
    hours["t"] = pd.date_range("1982-07", periods=len(hours.index), freq="M")
    hours.set_index(hours["t"], inplace=True)

    # plot_ts(
    #     hours.index, hours["hours"], "Time (Months)", "Hours", "Plot of Hours Dataset"
    # )

    # problem 3.6
    beersales = pd.read_csv(os.path.join(cs.DATASETS, "beersales.dat"))
    beersales.set_index(
        pd.date_range("1975-01", periods=len(beersales.index), freq="M"), inplace=True
    )
    # 3.6 a
    # plot_ts(
    #     beersales.index,
    #     beersales["beersales"],
    #     "Time (Months)",
    #     "Beer Sales (Millions of Barrels)",
    #     "Millions of Barrels sold in the U.S. from 1975 - 1990",
    # )
    # 3.6 b
    plot_monthly(
        beersales.index,
        beersales["beersales"],
        "Time (Months)",
        "Beer Sales (Millions of Barrels)",
        "Millions of Barrels sold in the U.S. from 1975 - 1990",
    )
    # 3.6 c
    beersales["month"] = beersales.index.month
    model = smf.ols("beersales ~ month", data=beersales).fit()
    print(model.summary())



if __name__ == "__main__":
    main()
