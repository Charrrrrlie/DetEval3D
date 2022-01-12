import argparse
from waymo_eval_utils.loaders import *

from visual_utils.bbox_points_visualize import *
from statistic import find_match

def parse_config():
    parser = argparse.ArgumentParser(description='Offline evaluation tool for waymo.')
    parser.add_argument('--det_result_path', type=str,
                        default='./val_res/5sweep1stg_detection_pred.bin',
                        help='Path to the prediction result pkl file.')
    parser.add_argument('--gt_info_pkl_path', type=str,
                        default='s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.val.pkl',
                        help='Path to ground-truth info pkl file.')

    parser.add_argument('--det_input_form', type=str, default='bin',
                        help='The detection input form, optional: [\'bin\',\'ctp_pickle\',\'3dal_pickle\']')
    parser.add_argument('--det_score_form', type=str, default='none',
                        help='Visualize scores for detection results, optional:[\'none\',\'conf\',\'iou\']')
    parser.add_argument('--info_path', type=str, default='./info_map/sequence_info_val.pkl',
                        help='Sequence name to frame id map')

    args = parser.parse_args()

    return args

def draw_conf_score(det_result, det_seq_name, frame_num, vis=None, thred=0):
    det_bbox = det_result[det_seq_name][str(frame_num)]['boxes_lidar']
    det_score = det_result[det_seq_name][str(frame_num)]['score']

    mask = det_score > thred
    det_bbox = det_bbox[mask]
    det_score = det_score[mask]

    vis = vis_box(det_bbox.copy(), vis=vis, color=(1, 0, 0), labels=det_score)
    return vis

def draw_iou(det_result, det_seq_name, frame_num, vis=None, thred=0):
    det_bbox = det_result[det_seq_name][str(frame_num)]['boxes_lidar']
    _, det_iou = find_match(det_bbox, bbox)

    mask = det_iou > thred
    det_bbox = det_bbox[mask]
    det_iou = det_iou[mask]

    vis = vis_box(det_bbox.copy(), vis=vis, color=(1, 0, 0), labels=det_iou)
    return vis

# def iou_statistic():
#     # just used for postprocessing check
#     from waymo_eval_utils.utils import boxes_iou3d2d_gpu
#
#
#     def find_match(det_boxes3d, gt_boxes3d, device='cuda:0'):
#         det_boxes3d = torch.from_numpy(det_boxes3d).cuda(device=device).float().reshape(-1, 7)
#         gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(device=device).float().reshape(-1, 7)
#
#         iou3d, iou2d = boxes_iou3d2d_gpu(det_boxes3d, gt_boxes3d)
#
#         iou_val, iou_idx = torch.sort(iou3d, descending=True, dim=1)[:5]
#
#         iou_val = iou_val.cpu().numpy()
#         iou_idx = iou_idx.cpu().numpy()
#
#         return iou_idx, iou_val
#
#     iou_idx, iou_val = find_match(det_result[det_seq_name][str(frame_num)]['boxes_lidar'][:, :7], det_result[det_seq_name][str(frame_num)]['boxes_lidar'][:, :7])
#
#     boxes = torch.from_numpy(det_result[det_seq_name][str(frame_num)]['boxes_lidar'][:, :7]).float().cuda(device='cuda:0')
#     keep = torch.LongTensor(boxes.shape[0])
#     from iou3d_nms import iou3d_nms_cuda
#     num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, 0.7)

# iou_statistic()
# mypath="./validation_check/check.npy" #+str(num).zfill(4)+".npy"
# raw_pt = np.load(mypath)
# print(raw_pt.shape)
# fig=visualize_pts(raw_pt,fig)


if __name__ == '__main__':

    # INPUT DEMO
    # gt_path = './val_res/validation_gt.pkl'
    # det_path = './val_res/3sweep1stg_rotTTA_detection_pred.bin'
    # info_path = './info_map/sequence_info_val.pkl'
    #
    # sequence_name = 'segment-6862795755554967162_2280_000_2300_000_with_camera_labels'  # test set
    # sequence_name = 'segment-7493781117404461396_2140_000_2160_000_with_camera_labels'  # val set
    #
    # frame_num = 2
    # data_form = 'bin'

    args = parse_config()

    # TODO
    sequence_name = 'segment-7493781117404461396_2140_000_2160_000_with_camera_labels'
    frame_num = 2
    det_seq_name = sequence_name.split('_with_camera_labels')[0].split('segment-')[-1]

    det_path = args.det_result_path
    gt_path = args.gt_info_pkl_path
    info_path = args.info_path
    data_form = args.det_input_form

    # plot gt
    gt_data, _ = pickle_loader(gt_path)
    bbox = gt_data[sequence_name][frame_num]['annos']['gt_boxes_lidar'][:, :7]
    # mask = (gt_data[sequence_name][frame_num]['annos']['name'] == 'Vehicle')
    # bbox = bbox[mask]

    print(len(bbox))
    vis = vis_box(bbox.copy(), color=(0, 0, 1))

    # plot det
    if data_form == 'bin':
        det_result = bin_loader(det_path)
        raw_map = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Sign', 4: 'Cyclist'}
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        det_result = data_convert(det_result, info, raw_map, [3])  # mask num:[3]

    elif data_form == 'pkl':
        det_result = defaultdict(dict)
        det_data = pickle.load(open(det_path, 'rb'))

        for item in det_data:
            det_result[item['sequence_name']][str(item['frame_id'])] = item

    elif data_form == 'ctp_pkl':
        det_result = pickle.load(open(det_path, 'rb'))

    if args.det_score_form == 'none':
        vis = vis_box(det_result[det_seq_name][str(frame_num)]['boxes_lidar'].copy(), vis=vis, color=(1, 0, 0))
    elif args.det_score_form == 'conf':
        vis = draw_conf_score(det_result.copy(), det_seq_name, frame_num, vis=vis)
    elif args.det_score_form == 'iou':
        vis = draw_iou(det_result.copy(), det_seq_name, frame_num, vis=vis)

    print(len(det_result[det_seq_name][str(frame_num)]['boxes_lidar']))
    vis.run()



