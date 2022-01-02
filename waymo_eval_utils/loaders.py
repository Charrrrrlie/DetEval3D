import numpy as np
import pickle
import io
from collections import defaultdict
# from petrel_client.client import Client
from waymo_open_dataset.protos import metrics_pb2


def bin_loader(data_path):
    """
    :param data_path: .bin detection data
    :return: seqs: sequence_name dict of timestamp dict
    """
    objects = metrics_pb2.Objects()
    objects.ParseFromString(open(data_path, 'rb').read())

    # each:
    # each.object-> .box / .type
    # each.score /.context_name /.frame_timestamp_micros
    seqs = {}
    for each in objects.objects:
        # print(type(each.frame_timestamp_micros))
        # print(type(each))
        boxes_lidar = np.array(
            [each.object.box.center_x, each.object.box.center_y, each.object.box.center_z, each.object.box.length,
             each.object.box.width, each.object.box.height, each.object.box.heading])
        if each.context_name not in seqs.keys():
            seqs[each.context_name] = dict.fromkeys([str(each.frame_timestamp_micros)],
                                                    {'sequence_name': each.context_name,
                                                     'timestamp': each.frame_timestamp_micros, 'score': [each.score],
                                                     'name': [each.object.type], 'boxes_lidar': [boxes_lidar]})
        else:
            if str(each.frame_timestamp_micros) not in seqs[each.context_name].keys():
                seqs[each.context_name][str(each.frame_timestamp_micros)] = {'sequence_name': each.context_name,
                                                                             'timestamp': each.frame_timestamp_micros,
                                                                             'score': [each.score],
                                                                             'name': [each.object.type],
                                                                             'boxes_lidar': [boxes_lidar]}
            else:
                seqs[each.context_name][str(each.frame_timestamp_micros)]['score'].append(each.score)
                seqs[each.context_name][str(each.frame_timestamp_micros)]['name'].append(each.object.type)
                seqs[each.context_name][str(each.frame_timestamp_micros)]['boxes_lidar'].append(boxes_lidar)

    return seqs


# load pickle and convert into sequence_name dict of frame_id dict
def pickle_loader(data_path, client=None, from_s3=False):
    if from_s3:
        infos_bytes = client.get(data_path)
        data = pickle.load(io.BytesIO(infos_bytes))
    else:
        data = pickle.load(open(data_path, 'rb'))
    data_table = defaultdict(dict)

    for item in data:
        data_table[item['sequence_name']][item['sample_idx']] = item

    return data_table, len(data)


def data_convert(seqs, infos, raw_map, mask_num=[3]):
    '''
    :param seqs: sequence_name dict of timestamp dict
    :param infos: includes pose and frame_id info
    :param raw_map: from class_id to class_name
    :return: seqs: sequence_name dict of frame_id dict
    '''

    for k in seqs.keys():
        # flag=False
        for f in seqs[k].keys():
            if f in infos.keys():
                # print(("Processed:{}[!!!!]{}").format(k,f))
                # cnt+=1
                # flag = True
                seqs[k][f].update({'pose': infos[f]['pose'], 'frame_id': infos[f]['sample_idx']})
                # check_dict[f]-=1
                # change form in every frame for every keys
                idx = []
                for kk in seqs[k][f]:
                    if kk == 'name':
                        # print(seqs[k][f][kk])
                        # print(type(seqs[k][f][kk]))
                        temp = []
                        for i in range(seqs[k][f][kk].__len__()):
                            if seqs[k][f][kk][i] in mask_num:
                                continue
                            # mask with 3:SIGN type
                            idx.append(i)
                            temp.append(raw_map[seqs[k][f][kk][i]])

                        seqs[k][f][kk] = np.array(temp)

                        seqs[k][f][kk] = np.array(seqs[k][f][kk])
                    elif kk == 'boxes_lidar' or kk == 'score':
                        seqs[k][f][kk] = np.array(seqs[k][f][kk])

                seqs[k][f]['boxes_lidar'] = seqs[k][f]['boxes_lidar'][idx]
                seqs[k][f]['score'] = seqs[k][f]['score'][idx]

            #     if not ( len(seqs[k][f]['boxes_lidar'])==len(seqs[k][f]['score']) and len(seqs[k][f]['boxes_lidar'])== len(seqs[k][f]['name'])):
            #         print('ohno')
            #
    #         else:
    #             print('ohno')
    # check whether all info keys are used
    # for k in check_dict.keys():
    #     if check_dict[k]!=0:
    #         print(k)

    for k in seqs.keys():
        temp_key = list(seqs[k].keys()).copy()
        for f in temp_key:
            # alter key_value from timestamp to frame_id
            seqs[k][str(seqs[k][f]['frame_id'])] = seqs[k][f]
            del seqs[k][f]
        # print(seqs[k].keys())

    return seqs


def seqs2list(seqs):
    res_list = []
    for k in seqs.keys():
        for f in seqs[k].keys():
            res_list.append(seqs[k][f])

    return res_list


def convert_gt(gt_frameid, gt_boxes3d, gt_type, gt_difficulty, raw_map, class_name):
    """
        gt in frame_id dict of class_type dict
    """

    gt = defaultdict(dict)
    uni_id = np.unique(gt_frameid)

    for uni in uni_id:
        for k in class_name:
            gt[uni][k] = {}

    for i in range(len(gt_frameid)):
        bbox = gt_boxes3d[i]
        id = gt_frameid[i]
        type = raw_map[gt_type[i]]
        diff = gt_difficulty[i]

        if len(gt[id][type]) == 0:
            gt[id][type] = {'difficulty': [diff], 'bbox': [bbox], 'used': [False]}
        else:
            gt[id][type]['difficulty'].append(diff)
            gt[id][type]['bbox'].append(bbox)
            gt[id][type]['used'].append(False)

    for uni in uni_id:
        for k in class_name:
            if len(gt[uni][k]):
                gt[uni][k]['difficulty'] = np.array(gt[uni][k]['difficulty'])
                gt[uni][k]['bbox'] = np.array(gt[uni][k]['bbox'])

    return gt


def convert_det(det_frameid, det_boxes3d, det_type, det_score, raw_map, class_name):
    '''
       det in class dict and sorted by confidence score reversely
    '''

    det = defaultdict(dict)
    uni_id = np.unique(det_frameid)
    for k in class_name:
        det[k] = {}

    for i in range(len(det_frameid)):
        bbox = det_boxes3d[i]
        id = det_frameid[i]
        type = raw_map[det_type[i]]
        score = det_score[i]

        if len(det[type]) == 0:
            det[type] = {'score': [score], 'bbox': [bbox], 'frame_id': [id]}
        else:
            det[type]['score'].append(score)
            det[type]['bbox'].append(bbox)
            det[type]['frame_id'].append(id)

    for k in det.keys():
        if len(det[k]):
            det[k]['score'] = np.array(det[k]['score'])
            idx = np.argsort(-det[k]['score'])  # reversed
            det[k]['score'] = np.array([det[k]['score'][i] for i in idx])
            det[k]['bbox'] = np.array([det[k]['bbox'][i] for i in idx])
            det[k]['frame_id'] = np.array([det[k]['frame_id'][i] for i in idx])

    return det
