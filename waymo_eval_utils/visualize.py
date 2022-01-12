import matplotlib.pyplot as plt
import os
import numpy as np

def plot_auc(rec, prec, mrec, mprec, class_name, text, save_path=None):
    """
     Draw area under curve
    """
    plt.plot(rec, prec, '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # set window title
    fig = plt.gcf()  # gcf - get current figure
    fig.canvas.set_window_title('AP ' + class_name)
    # set plot title
    plt.title('class: ' + text)
    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    # while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    # plt.show()
    # save the plot
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + "/" + class_name + ".png")
    else:
        plt.show()
    plt.cla()  # clear axes for next plot


def offboard_plot_pr(rec, prec, labels, class_name, text, save_path=None):
    l = len(rec)

    plt_names = []
    for i in range(l):
        n = plt.plot(rec[i], prec[i])
        plt_names.append(n)

    plt.legend(handles=plt_names, labels=labels)

    fig = plt.gcf()

    fig.canvas.set_window_title('AP ' + class_name)
    # set plot title
    plt.title('class: ' + text)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + "/" + class_name + "_full_pr_curve.png")
    else:
        plt.show()
    plt.cla()


def plot_multi_pr(rec, prec, labels, class_name, text, save_path=None, metric='L1'):
    l = len(rec)

    plt_names = []
    for i in range(l):
        n,  = plt.plot(rec[i], prec[i])
        plt_names.append(n)

    plt.legend(handles=plt_names, labels=labels)

    fig = plt.gcf()

    fig.canvas.set_window_title('AP ' + class_name)
    # set plot title
    plt.title(text)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + "/" + class_name + "_" + metric + "_full_pr_curve.png")
    else:
        plt.show()
    plt.cla()


def plot_multi_score_cutoff(score_cut, labels, class_name, text, save_path=None, metric='L1_TP'):

    l = len(score_cut)
    plt_names = []

    for i in range(l):
        n,  = plt.plot(np.arange(0, 1.1, 0.01), score_cut[i])
        plt_names.append(n)

    plt.legend(handles=plt_names, labels=labels)

    fig = plt.gcf()

    fig.canvas.set_window_title(metric + ' Score Cut Off ' + class_name)
    # set plot title
    plt.title(text)

    plt.xlabel('Score Threshold')
    plt.ylabel('Nums')

    # optional - set axes
    axes = plt.gca()  # gca - get current axes
    axes.set_xlim([0.0, 1.0])

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + "/" + class_name + "_" + metric + "_full_score_cutoff.png")
    else:
        plt.show()
    plt.cla()


