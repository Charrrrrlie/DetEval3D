import copy
import pickle

from visual_utils.plot_utils import *


def offboard_visualize(path, labels, class_name, save_path=None):

    tp_scts = {'L1': {'Vehicle': [], 'Pedestrian': [], 'Cyclist': []},
          'L2': {'Vehicle': [], 'Pedestrian': [], 'Cyclist': []}}

    fp_scts = copy.deepcopy(tp_scts)
    precs = copy.deepcopy(tp_scts)
    recs = copy.deepcopy(tp_scts)

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    for p in path:
        tp_sct = pickle.load(open(p + '/tp_sct.pkl', 'rb'))
        fp_sct = pickle.load(open(p + '/fp_sct.pkl', 'rb'))
        prec = pickle.load(open(p + '/precision.pkl', 'rb'))
        rec = pickle.load(open(p + '/recall.pkl', 'rb'))

        for level in ['L1', 'L2']:
            for cls in class_name:
                tp_scts[level][cls].append(tp_sct[level][cls])
                fp_scts[level][cls].append(fp_sct[level][cls])
                precs[level][cls].append(prec[level][cls])
                recs[level][cls].append(rec[level][cls])

    for level in ['L1', 'L2']:
        for cls in class_name:

            pr_text = "PR Curve"
            plot_multi_pr(recs[level][cls], precs[level][cls], labels,
                          class_name=cls, text=pr_text, save_path=save_path, metric=level)

            sct_tp_text = "TP Score Cut Off"
            plot_multi_score_cutoff(tp_scts[level][cls], labels, class_name=cls,
                                    text=sct_tp_text, save_path=save_path, metric=level + '_TP')
            sct_fp_text = "FP Score Cut Off"
            plot_multi_score_cutoff(fp_scts[level][cls], labels, class_name=cls,
                                    text=sct_fp_text, save_path=save_path, metric=level + '_FP')


if __name__ == '__main__':

    # visualize
    path = ['./output/1sweep1stg_class', './output/2sweep2stg_class', './output/3sweep2stg_future_class']
    labels = ['1sweep1stg', '2sweep2stg', '3sweep2stg']
    class_name = ['Vehicle', 'Pedestrian', 'Cyclist']
    save_path = './output/final/'

    offboard_visualize(path, labels, class_name, save_path)
