# Offline evaluation tool

## TL;DL

Requirements: identical with 3dal but need to install `tabulate` by `pip install tabulate`

`python -u al3d_eval.py --det_result_path <PATH_TO_DETECTION_RESULT> --evaluate_metrics object` 

or

`srun -p shlab_adg python -u waymo_offline_eval.py --det_result_path <PATH_TO_DETECTION_RESULT_PKL> --gt_info_pkl_path <PATH_TO_GT_PKL>`

or even

`python -u waymo_offline_eval.py --det_result_path <PATH_TO_DETECTION_RESULT> --evaluate_metrics object range`

`--evaluate_metrics` indicates the desired evaluation that you want to conduct. Support `object` (by default), `range`, or, `object range`. More metrics means much elapsed time, `object` is fastest and adequate for most circumstance.

E.g.

```
srun -p shlab_adg python -u al3d_eval.py \
  --det_result_path ./val_res/1sweep1stg_detection_pred.bin \
  --gt_info_pkl_path s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.val.pkl
```

As for the <PATH_TO_XXX>, you can use the path of file that either in local file system or s3.

## Intro

This is a tool for offline evaluating 3dal detection results. You can use the detection results of the 3dal pipeline to conduct the evaluation.

You must have the detection result and the ground-truth annotations as the input and the evaluator will invoke the waymo_open_dataset tool to assess the performance.

You can make the prediction without considering the order, to do so, you must include the sequence name (a.k.a. sequence_name) and the index of the frame in the sequence (a.k.a. frame_id) in the generated detection result.

In order to make the evaluation, you will also need a ground-truth annotations. However, I have generated these files and stored them in `s3://Datasets/Waymo/gt_infos/`, including:

```
s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.train.pkl    
s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.val.pkl
s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.all.pkl
```

We use `s3://Datasets/Waymo/gt_infos/centerpoint_align_onesweep.gt_infos/gt.all.pkl` by default.

## Usage

```
usage: waymo_offline_eval.py [-h] [--det_result_path DET_RESULT_PATH]
                             [--gt_info_pkl_path GT_INFO_PKL_PATH]
                             [--evaluate_metrics EVALUATE_METRICS [EVALUATE_METRICS ...]]
                             [--petrel_oss_config PETREL_OSS_CONFIG]
                             [--class_name CLASS_NAME [CLASS_NAME ...]]
                             [--info_with_fakelidar]
                             [--distance_thresh DISTANCE_THRESH]
                             [--raw_map_path RAW_MAP_PATH]
                             [--det_input_form DET_INPUT_FORM]

Offline evaluation tool for 3dal.

optional arguments:
  -h, --help            show this help message and exit
  --det_result_path DET_RESULT_PATH
                        Path to the prediction result file.
  --gt_info_pkl_path GT_INFO_PKL_PATH
                        Path to ground-truth info pkl file.
  --evaluate_metrics EVALUATE_METRICS [EVALUATE_METRICS ...]
                        metrics that used in evaluation. support multiple
                        (object, range)
  --petrel_oss_config PETREL_OSS_CONFIG
                        The configuration file of petrel-oss.
  --class_name CLASS_NAME [CLASS_NAME ...]
                        The class names of the detection, default=["Vehicle",
                        "Pedestrian", "Cyclist"]
  --info_with_fakelidar
  --distance_thresh DISTANCE_THRESH
  --info_path INFO_PATH
                        Sequence name to frame id map
  --det_input_form DET_INPUT_FORM
                        The detection input form, optional:
                        ['bin','ctp_pickle','3dal_pickle']

```

## Layout of prediction result format

It support different kinds of input. Details in [README.intro](../README.md)

## GT infos generation

You can use this script to generate the ground-truth infos pkl file.

TBA
