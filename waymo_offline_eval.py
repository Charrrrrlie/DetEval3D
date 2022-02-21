import argparse
import logging
import pickle
import io
import copy
import numpy as np

from collections import defaultdict
from petrel_client.client import Client
from waymo_eval_utils.evals import WaymoDetectionMetricsEstimator
from tabulate import tabulate

from waymo_eval_utils.loaders import bin_loader, pickle_loader, data_convert, seqs2list

def parse_config():
    parser = argparse.ArgumentParser(description='Offline evaluation tool for 3dal.')
    parser.add_argument('--det_result_path', type=str,
                        default='./val_res/1sweep1stg_detection_pred.bin',
                        help='Path to the prediction result file.')
    parser.add_argument('--gt_info_pkl_path', type=str,
                        default='s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.all.pkl',
                        help='Path to ground-truth info pkl file.')
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

    parser.add_argument('--info_path', type=str, default='./info_map/sequence_info_val.pkl',
                        help='Sequence name to frame id map')
    parser.add_argument('--det_input_form', type=str, default='bin',
                        help='The detection input form, optional: [\'bin\',\'ctp_pickle\',\'3dal_pickle\']')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    client = Client(args.petrel_oss_config)

    return args, logger, client


def main():
    args, logger, client = parse_config()
    if any([item not in ['object', 'range'] for item in args.evaluate_metrics]):
        raise ValueError('evaluate_metrics error')

    logger.info(args)

    logger.info('Loading ground-truth infos: ' + args.gt_info_pkl_path)
    gt_infos_bytes = client.get(args.gt_info_pkl_path)
    gt_infos = pickle.load(io.BytesIO(gt_infos_bytes))
    gt_infos_table = defaultdict(dict)
    missed_info_table = defaultdict(dict)

    for item in gt_infos:
        gt_infos_table[item['sequence_name']][item['sample_idx']] = item
        missed_info_table[item['sequence_name']][item['sample_idx']] = item

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

    logger.info("Prediction Set Length: {}, Ground Truth Set Length: {}.".format(len(det_result), len(gt_infos)))

    logger.info('Generating evaluation data pair (det, gt)')
    eval_gt_annos = []
    eval_det_annos = []

    logger.info(len(det_result), len(gt_infos))
    # assert len(det_result)==len(gt_infos),"all frames must be detected!"

    for item in det_result:

        if 'segment' not in item['sequence_name']:
            # fit in ctp pkl sequence_name form
            item['sequence_name'] = 'segment-{}_with_camera_labels'.format(item['sequence_name'])

        eval_det_annos.append(copy.deepcopy(item))
        eval_gt_annos.append(copy.deepcopy(gt_infos_table[item['sequence_name']][item['frame_id']]['annos']))
        del missed_info_table[item['sequence_name']][item['frame_id']]

    # all frames must be detected!
    if not len(det_result) == len(gt_infos):
        logger.info("all frames must be detected, empty list will be appended.")

        for k in missed_info_table.keys():
            for f in missed_info_table[k].keys():
                logger.info("sequence:{},frame:{} is missed".format(k, f))
                empty = {'sequence_name': k, 'frame_id': f,
                         'boxes_lidar': np.array([]).reshape(0, 7), 'score': np.array([]),
                         'name': np.array([])}
                eval_gt_annos.append(missed_info_table[k][f]['annos'])
                eval_det_annos.append(empty)

        logger.info("After modification | Prediction Set Length: {}, Ground Truth Set Length: {}."
                    .format(len(eval_gt_annos), len(eval_det_annos)))

    logger.info('\tEvaluation data pair: ' + str(len(eval_gt_annos)))

    logger.info('Start Evaluation')
    evaluator = WaymoDetectionMetricsEstimator()
    ap_dict = evaluator.waymo_evaluation(
        eval_det_annos, eval_gt_annos, config_type=args.evaluate_metrics, class_name=args.class_name,
        distance_thresh=args.distance_thresh, fake_gt_infos=args.info_with_fakelidar)

    logger.info('================= Evaluation Result =================')

    table_header = ['Category', 'L1/AP', 'L1/APH', 'L2/AP', 'L2/APH']
    table_data = []

    if 'object' in args.evaluate_metrics:
        table_data.extend([
            ('Vehicle', ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH'][0], ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH'][0]),
            ('Pedestrain', ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH'][0],
             ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH'][0]),
            ('Cyclist', ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH'][0], ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'][0],
             ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH'][0]),
            ('Sign', ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP'][0], ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH'][0],
             ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP'][0], ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH'][0])
        ])
    if 'range' in args.evaluate_metrics:
        table_data.extend([
            ('Vehicle_[0, 30)', ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/APH'][0], ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/APH'][0]),
            ('Vehicle_[30, 50)', ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/APH'][0]),
            ('Vehicle_[50, +inf)', ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/APH'][0]),
            ('Pedestrian_[0, 30)', ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/APH'][0]),
            ('Pedestrian_[30, 50)', ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/APH'][0]),
            ('Pedestrian_[50, +inf)', ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/APH'][0]),
            ('Cyclist_[0, 30)', ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/APH'][0], ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/APH'][0]),
            ('Cyclist_[30, 50)', ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/APH'][0]),
            ('Cyclist_[50, +inf)', ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/APH'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/APH'][0]),
            ('Sign_[0, 30)', ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_1/APH'][0], ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_2/APH'][0]),
            ('Sign_[30, 50)', ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_1/APH'][0], ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_2/APH'][0]),
            ('Sign_[50, +inf)', ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1/APH'][0], ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2/AP'][0],
             ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2/APH'][0]),
        ])

    _ = [logger.info(line) for line in tabulate(table_data, headers=table_header, tablefmt='grid').splitlines()]

    logger.info('Done.')


if __name__ == '__main__':
    main()
