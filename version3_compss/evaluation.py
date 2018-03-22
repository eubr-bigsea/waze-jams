#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Evaluation - WazeJams.

This program generates some statistics about the
quality of results: ROC graph, Precision-Recall graph,
a histogram to analyze the distribution of prevision
and other statistics.
"""

__author__ = "Lucas Miguel S Ponce"
__email__ = "lucasmsp@gmail.com"

import os
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.nan)
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt


def plot_histogram(pred, name):
    """Plot histogram."""
    plt.clf()
    plt.hist(pred, bins=2)
    plt.title("Predictions Frequency")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim((0, 1))
    plt.xticks([0, 1])
    name = "{}_bars.png".format(name)
    plt.savefig(name)


def plot_ROC(actual, predictions, name):
    """Plot ROC graph."""
    false_positive_rate, true_positive_rate, thresholds = \
        roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    name = "{}_ROC.png".format(name)
    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    plt.savefig(name)
    return roc_auc


def plot_precision_recall(actual, predictions, name):
    """Plot Precision-Recall Graph."""
    precision, recall, threshold = \
        precision_recall_curve(actual, predictions, pos_label=1)

    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    name = "{}_PrecisionRecall.png".format(name)
    plt.savefig(name)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Metrics - WazeJams')
    p.add_argument('-g', '--gridFile', required=True,
                   help='File with Grids List.')
    p.add_argument('-f', '--forecast', required=True,
                   help='File with the predictions')
    p.add_argument('-t', '--table', required=True,
                   help='File with the input data of wazejams (matrix)')
    arg = vars(p.parse_args())

    oficialTable = pd.read_csv(arg['table'], sep=',', header=None)
    Grids = pd.read_csv(arg['gridFile'], sep=',')
    forecasts = pd.read_csv(arg['forecast'], sep=',')

    oficialTable.columns = ['timestamp'] + [str(i) for i in xrange(1, 2501)]
    forecasts.columns = ['IDgrid', 'last_timestamp', 'average',
                         'variance', 'ci_95', 'ci_95.1', 'percentage']
    forecasts["percentage"] = forecasts["percentage"].apply(np.round)

    forecasts = forecasts.merge(Grids, how='left', on='IDgrid')
    forecasts["True"] = np.nan
    instance = forecasts['last_timestamp'].iloc[0]
    trues = oficialTable.loc[oficialTable['timestamp'] == instance].values[0]
    f1 = lambda row: trues[int(row['IDgrid'])]
    forecasts['True'] = forecasts.apply(f1, axis=1)
    f2 = lambda row: row['True'] == int(row['percentage'])
    forecasts['Right'] = forecasts.apply(f2, axis=1)

    N_green = len(forecasts[forecasts.percentage == 0])
    N_total = len(forecasts)
    N_red = N_total - N_green
    acc = 100.0 * len(forecasts[forecasts.Right == True]) / N_total

    true = forecasts['True'].values
    pred = forecasts['percentage'].values
    N_rred = sum(forecasts['True'].values)
    N_rgreen = N_total - N_rred
    base = os.path.basename(arg['forecast']).replace(".txt", "")
    AUC = plot_ROC(true, pred, base) * 100
    plot_precision_recall(true, pred, base)
    # plot_histogram(true, base)
    # plot_histogram(pred,' base)

    print "*" * 20
    print """
    Number of Grids: {}
    Number of predicted grids w/ traffic jams: {}
    Number of predicted grids w/o traffic jams: {}
    Number of real grids w/ traffic jams: {}
    Number of real grids w/o traffic jams: {}
    Accuracy: {:.2f}%
    AUC: {:.2f}%
    """.format(N_total, N_red, N_green, N_rred, N_rgreen, acc, AUC)
    print "*" * 20
