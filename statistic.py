import os.path
import pdb

from waymo_eval_utils.evals import WaymoDetectionMetricsEstimator
from waymo_eval_utils.loaders import *
from waymo_eval_utils.utils import *
from waymo_eval_utils.visualize import plot_auc
import argparse
import logging
import copy
from petrel_client.client import Client
from tqdm import tqdm


def parse_config():
    parser = argparse.ArgumentParser(description='Offline evaluation tool for Waymo.')
    parser.add_argument('--det_result_path', type=str,
                        default='./val_res/5sweep1stg_detection_pred.bin',
                        help='Path to the prediction result pkl file.')
    parser.add_argument('--gt_info_pkl_path', type=str,
                        default='s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.val.pkl',
                        help='Path to ground-truth info pkl file.')
    parser.add_argument('--output_path', type=str, default='./output/class')

    parser.add_argument('--evaluate_metrics', nargs='+', default=['object'],
                        help='metrics that used in evaluation. support multiple (object, range)')
    parser.add_argument('--petrel_oss_config', type=str, default='~/.petreloss.conf',
                        help='The configuration file of petrel-oss.')
    parser.add_argument('--class_name', nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'],
                        help='The class names of the detection, default=["Vehicle", "Pedestrian", "Cyclist"]')
    # parser.add_argument('--post_processing_eval_metric', type=str, default='waymo')
    # parser.add_argument('--post_processing_recall_thresh_list', nargs='+', default=[0.3, 0.5, 0.7])

    parser.add_argument('--info_with_fakelidar', action='store_true', default=False)
    parser.add_argument('--distance_thresh', type=int, default=1000)

    parser.add_argument('--single_seq_demo', action='store_true', default=False)
    parser.add_argument('--det_input_form', type=str, default='bin',
                        help='The detection input form, optional: [\'bin\',\'ctp_pickle\',\'3dal_pickle\']')

    parser.add_argument('--info_path', type=str, default='./info_map/sequence_info_val.pkl',
                        help='Sequence name to frame id map')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    return args, logger


def find_match(det_boxes3d, gt_boxes3d, device='cuda:0'):
    det_boxes3d = torch.from_numpy(det_boxes3d).cuda(device=device).float().reshape(-1, 7)
    gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(device=device).float().reshape(-1, 7)

    iou3d, iou2d = boxes_iou3d2d_gpu(det_boxes3d, gt_boxes3d)
    iou_val, iou_idx = torch.max(iou3d, dim=1)

    iou_val = iou_val.cpu().numpy()
    iou_idx = iou_idx.cpu().numpy()

    return iou_idx, iou_val


def base_detection_eval(eval_gt_annos, eval_det_annos, evaluator, logger,
                        class_name=['Vehicle', 'Pedestrian', 'Cyclist'], distance_thresh=1000, fake_gt_infos=False):
    assert len(eval_det_annos) == len(eval_gt_annos), 'det and gt must be paired!'

    tp = {'L1': {'Vehicle': [], 'Pedestrian': [], 'Cyclist': []},
          'L2': {'Vehicle': [], 'Pedestrian': [], 'Cyclist': []}}
    fp = copy.deepcopy(tp)

    precision = copy.deepcopy(tp)
    recall = copy.deepcopy(tp)

    mrec = copy.deepcopy(tp)
    mprec = copy.deepcopy(tp)

    tp_score_cutoff = copy.deepcopy(tp)
    fp_score_cutoff = copy.deepcopy(tp)

    logger.info("Converting Data Form")

    det_frameid, det_boxes3d, det_type, det_score, det_overlap_nlz \
        = evaluator.get_waymo_filter(eval_det_annos, class_name, distance_thresh=distance_thresh,
                                     is_gt=False, fake_gt_infos=fake_gt_infos)
    gt_frameid, gt_boxes3d, gt_type, gt_score, gt_difficulty \
        = evaluator.get_waymo_filter(eval_gt_annos, class_name, distance_thresh=distance_thresh,
                                     is_gt=True, fake_gt_infos=fake_gt_infos)

    iou_thred = {'Vehicle': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
    raw_map = {1: 'Vehicle', 2: 'Pedestrian', 4: 'Cyclist'}

    gt_all = {'L1': {'Vehicle': 0.0, 'Pedestrian': 0.0, 'Cyclist': 0.0},
              'L2': {'Vehicle': 0.0, 'Pedestrian': 0.0, 'Cyclist': 0.0}}

    ap = copy.deepcopy(gt_all)

    for i, g_type in enumerate(gt_type):
        if gt_difficulty[i] == 1:
            gt_all['L1'][raw_map[g_type]] += 1
        gt_all['L2'][raw_map[g_type]] += 1

    det = convert_det(det_frameid, det_boxes3d, det_type, det_score, raw_map, class_name)
    gt = convert_gt(gt_frameid, gt_boxes3d, gt_type, gt_difficulty, raw_map, class_name)

    # sum_AP = 0.0
    logger.info("Start to calculate FP and TP!")

    for type in class_name:
        temp_nd = len(det[type]['score'])
        temp_tp = [0] * temp_nd
        temp_fp = [1] * temp_nd

        temp_det = det[type]
        l1_list = []  # l2 will be calculated in full data

        for i in tqdm(range(temp_nd)):
            frame_id = temp_det['frame_id'][i]

            if type in gt[frame_id].keys() and len(gt[frame_id][type]):
                temp_gt = gt[frame_id][type]
                # for one det, we need to match gt in the whole frame
                idx, val = find_match(temp_det['bbox'][i], temp_gt['bbox'])
                idx = np.int(idx)
                # match or not
                if val > iou_thred[type]:
                    if not gt[frame_id][type]['used'][idx]:
                        temp_tp[i] = 1
                        gt[frame_id][type]['used'][idx] = True
                        temp_fp[i] = 0

                    if gt[frame_id][type]['difficulty'][idx] == 1:
                        l1_list.append(i)
                else:
                    l1_list.append(i)
            else:
                l1_list.append(i)

        def accumulate(metric):
            if len(metric) == 0:
                return np.array([0.0])

            cnt = 0
            for idx, val in enumerate(metric):
                metric[idx] += cnt
                cnt += val
            return metric

        temp_tp = np.array(temp_tp)
        temp_fp = np.array(temp_fp)

        tp['L1'][type] = accumulate(temp_tp[l1_list].copy())
        tp['L2'][type] = accumulate(temp_tp.copy())
        fp['L1'][type] = accumulate(temp_fp[l1_list].copy())
        fp['L2'][type] = accumulate(temp_fp.copy())

        l1_scores = det[type]['score'][l1_list]
        l2_scores = det[type]['score']
        l1_score_loc = [max(0, len(l1_scores[l1_scores > i]) - 1) for i in np.arange(0, 1.1, 0.01)]
        l2_score_loc = [max(0, len(l2_scores[l2_scores > i]) - 1) for i in np.arange(0, 1.1, 0.01)]

        tp_score_cutoff['L1'][type] = tp['L1'][type][l1_score_loc]
        tp_score_cutoff['L2'][type] = tp['L2'][type][l2_score_loc]
        fp_score_cutoff['L1'][type] = fp['L1'][type][l1_score_loc]
        fp_score_cutoff['L2'][type] = fp['L2'][type][l2_score_loc]

        logger.info("{}: L1: tp {}, fp {}, L2:tp {}, fp {},".format(type, tp['L1'][type][-1], fp['L1'][type][-1],
                                                                    tp['L2'][type][-1], fp['L2'][type][-1]))
    # calculate!
    logger.info("Start to calculate APs!")
    logger.info("Gt_all: {}".format(gt_all))
    for level in tp.keys():
        for cls in class_name:
            for i in range(len(tp[level][cls])):
                precision[level][cls].append(tp[level][cls][i] / (tp[level][cls][i] + fp[level][cls][i] + 1e-8))
            recall[level][cls] = tp[level][cls] / (gt_all[level][cls] + 1e-8)
            ap[level][cls], mrec[level][cls], mprec[level][cls] = voc_ap(recall[level][cls].tolist(),
                                                                         precision[level][cls].copy())
            # sum_AP += ap[level][cls]

    # sum_AP /= len(class_name)

    # for id in unique_iou_idx:
    #     if gt_difficulty[id]==1:
    #         tp['L1'][raw_map[gt_type[id]]]+=1
    #     elif gt_difficulty[id]==2:
    #         tp['L2'][raw_map[gt_type[id]]]+=1
    #
    # _, iou_in_gt_idx = torch.max(iou3d, dim=1)
    # for i, d_type in enumerate(det_type):
    #     if gt_difficulty[iou_in_gt_idx[i]] == 1:
    #         det_all['L1'][raw_map[d_type]] += 1
    #     elif gt_difficulty[iou_in_gt_idx[i]] == 2:
    #         det_all['L2'][raw_map[d_type]] += 1

    # det_boxes3d:(N,7)  gt_boxes3d:(M,7)
    # iou3d: (N,M)
    # import pdb; pdb.set_trace()
    # for level in precision.keys():
    #     for cls in precision[level].keys():
    #         fp[level][cls] = det_all[level][cls]-tp[level][cls]
    #         fn[level][cls] = gt_all[level][cls]-tp[level][cls]
    #

    return tp, fp, tp_score_cutoff, fp_score_cutoff, ap, precision, recall, mrec, mprec


def preprocessing(args, logger):
    eval_gt_annos = []
    eval_det_annos = []

    # ## loading detection results in different forms ## #
    if args.det_input_form == '3dal_pickle':
        det_result = pickle.load(open(args.det_result_pkl_path, 'rb'))
    else:
        if args.det_input_form == 'bin':
            det_result = bin_loader(args.det_result_path)
            raw_map = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Sign', 4: 'Cyclist'}
        elif args.det_input_form == 'ctp_pickle':
            # list of pickle data
            det_result, _ = pickle_loader(args.det_result_path, from_s3=False)
            raw_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
        else:
            raise NotImplementedError

        with open(args.info_path, 'rb') as f:
            info = pickle.load(f)

        det_result = data_convert(det_result, info, raw_map, [3])  # mask num:[3]
        det_result = seqs2list(det_result)

    # ## loading gt results in different forms ## #
    if 's3' in args.gt_info_pkl_path:
        client = Client(args.petrel_oss_config)
        gt_result_table, gt_result_len = pickle_loader(args.gt_info_pkl_path, client=client, from_s3=True)
    else:
        gt_result_table, gt_result_len = pickle_loader(args.gt_info_pkl_path, from_s3=False)

    missed_gt_table = copy.deepcopy(gt_result_table)

    logger.info("Detection length:{}, Gt length:{}".format(len(det_result), str(gt_result_len)))
    # assert len(det_result)==len(gt_infos),"all frames must be detected!"
    logger.info('Generating evaluation data pair (det, gt)')
    for item in det_result:
        if 'segment' not in item['sequence_name']:
            # fit in ctp pkl sequence_name form
            item['sequence_name'] = 'segment-{}_with_camera_labels'.format(item['sequence_name'])
        eval_det_annos.append(copy.deepcopy(item))
        eval_gt_annos.append(copy.deepcopy(gt_result_table[item['sequence_name']][item['frame_id']]['annos']))
        del missed_gt_table[item['sequence_name']][item['frame_id']]

    # all frames must be detected!
    if not len(det_result) == gt_result_len:
        logger.info("all frames must be detected, empty list will be appended.")

        for k in missed_gt_table.keys():
            for f in missed_gt_table[k].keys():
                logger.info("sequence:{},frame:{} is missed".format(k, f))
                empty = {'sequence_name': k, 'frame_id': f,
                         'boxes_lidar': np.array([]).reshape(0, 7), 'score': np.array([]),
                         'name': np.array([])}
                eval_gt_annos.append(missed_gt_table[k][f]['annos'])
                eval_det_annos.append(empty)

        logger.info("After modification | Prediction Set Length: {}, Ground Truth Set Length: {}."
                    .format(len(eval_gt_annos), len(eval_det_annos)))

    logger.info('Evaluation data pair: ' + str(len(eval_gt_annos)))

    return eval_det_annos, eval_gt_annos


def process_single_seq(args, logger, scene_name, scene_id):
    eval_gt_annos = []
    eval_det_annos = []

    # ## loading detection results in different forms ## #
    if args.det_input_form == '3dal_pickle':
        det_result = pickle.load(open(args.det_result_path, 'rb'))
    else:
        if args.det_input_form == 'bin':
            det_result = bin_loader(args.det_result_path)
            raw_map = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Sign', 4: 'Cyclist'}

            with open(args.info_path, 'rb') as f:
                info = pickle.load(f)

            det_result = data_convert(det_result, info, raw_map, [3])  # mask num:[3]

        elif args.det_input_form == 'ctp_pickle':
            # list of pickle data
            det_result, _ = pickle_loader(args.det_result_path, from_s3=False)
            raw_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}

            with open(args.info_path, 'rb') as f:
                info = pickle.load(f)

            det_result = data_convert(det_result, info, raw_map, [3])  # mask num:[3]
        
        # used for tracking
        elif args.det_input_form == 'temp_pickle': 
            data = pickle.load(open(args.det_result_path, 'rb'))
            det_result = defaultdict(dict)

            for item in data:
                det_result[item['sequence_name']][str(item['frame_id'])] = item

            # det_result, _ = pickle_loader(args.det_result_path, from_s3=False)

        else:
            raise NotImplementedError


    # ## loading gt results in different forms ## #
    if 's3' in args.gt_info_pkl_path:
        client = Client(args.petrel_oss_config)
        gt_result_table, gt_result_len = pickle_loader(args.gt_info_pkl_path, client=client, from_s3=True)
    else:
        gt_result_table, gt_result_len = pickle_loader(args.gt_info_pkl_path, from_s3=False)

    # for sname, sid in zip(scene_name, scene_id):
    eval_det_annos.append(det_result[scene_name.split('segment-')[-1].split('_with_camera_labels')[0]][scene_id])
    eval_gt_annos.append(gt_result_table[scene_name][int(scene_id)]['annos'])

    return eval_det_annos, eval_gt_annos


def main():
    args, logger = parse_config()

    if args.single_seq_demo:
        sname = 'segment-7493781117404461396_2140_000_2160_000_with_camera_labels'
        sid = '2'
        args.output_path = './output/' + args.det_result_path.split('/')[-1]
        eval_det_annos, eval_gt_annos = process_single_seq(args, logger, sname, sid)
    else:
        eval_det_annos, eval_gt_annos = preprocessing(args, logger)

    # What's dumped in demo
    # eval_det_annos = eval_det_annos[:50]
    # eval_gt_annos = eval_gt_annos[:50]

    # with open('./val_res/gt_full.pkl', 'wb') as f:
    #     pickle.dump(eval_gt_annos, f)
    # with open('./val_res/pred_full.pkl', 'wb') as f:
    #     pickle.dump(eval_det_annos, f)

    logger.info('Start Evaluation')
    evaluator = WaymoDetectionMetricsEstimator()
    
    eval_tuple = base_detection_eval(eval_gt_annos, eval_det_annos, evaluator, logger,
                                     class_name=args.class_name,
                                     distance_thresh=args.distance_thresh,
                                     fake_gt_infos=args.info_with_fakelidar)

    logger.info("Start to Save Evaluation Result!")
    save_dict = ['/tp.pkl', '/fp.pkl', '/tp_sct.pkl', '/fp_sct.pkl',
                 '/ap.pkl', '/precision.pkl', '/recall.pkl', '/mrec.pkl', '/mprec']
    for i, item in enumerate(eval_tuple):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(args.output_path + save_dict[i], "wb") as f:
            pickle.dump(item, f)

    tp, fp, tp_sct, fp_sct, ap, precision, recall, mrec, mprec = eval_tuple

    logger.info("Start to Visualize PR curves!")
    for level in ['L1', 'L2']:
        for cls in args.class_name:
            text = "Class {}: AP = {:.2f}%".format(cls, ap[level][cls] * 100)
            cls_name = "{}_{}".format(level, cls)
            output_path = args.output_path
            plot_auc(recall[level][cls], precision[level][cls], mrec[level][cls], mprec[level][cls],
                     cls_name, text=text, save_path=output_path)

    logger.info('================= Evaluation Result =================')


def demo():
    evaluator = WaymoDetectionMetricsEstimator()
    pred = pickle.load(open('./val_res/gt_demo.pkl', 'rb'))
    gt = pickle.load(open('./val_res/gt_demo.pkl', 'rb'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    tp, fp, ap, precision, recall, mrec, mprec = base_detection_eval(gt, pred, evaluator, logger)

    logger.info("Start to Visualize PR curves!")
    for level in tp.keys():
        for cls in tp[level].keys():
            text = "Class {}: AP = {:.2f}%".format(cls, ap[level][cls] * 100)
            cls_name = "{}_{}".format(level, cls)
            output_path = "./output"
            # import pdb;pdb.set_trace()
            plot_auc(recall[level][cls], precision[level][cls], mrec[level][cls], mprec[level][cls],
                     cls_name, text=text, save_path=output_path)


if __name__ == '__main__':
    main()
    # demo()
