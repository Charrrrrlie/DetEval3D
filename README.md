## Installation

We implement the repo under Ubuntu 16.04 with gcc 5.4.0.

(note: open3d 0.9.0 only fits python<=3.6, and Ubuntu 16.04 only fits open3d<0.10.0)

```shell
conda create -n eval_check python=3.6

conda activte eval_check

conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch

pip install -r requirements.txt

pip3 install waymo-open-dataset-tf-2-4-0 --user

cd iou3d_nms && python setup.py build_ext --inplace  # compile necessary iou_nms funcs

```

## Getting Started

It's an unoffical evaluation scripts for Waymo 3D Perception task. It supports basic statistical analysis for 3D perception, 
including plotting `AP curves` in each type and each level of difficulty and plotting `Score Cut off curves` for TP
and FP results. 

### Introduction

Ground truth data .pkl should be in the following form. 

```shell
[
    {
        sequence_name: a string like "segment-10203656353524179475_7625_000_7645_000_with_camera_labels",
        frame_id: an int that indicates the index of the frame in the sequence,
        names: a numpy string array in [N],
        score: a numpy float array in [N],
        boxes_lidar: a numpy float32 array in [N, 7],
        pose: a numpy float array in [4, 4]
    },
    {
    ...
    },
    ...
]
```

**For detection results, we support `.bin`, `centerpoint pickle` and `3dal pickle` forms.**

For .bin file, the Waymo Official submission file, use `--det_input_form bin` to specify it:

```shell
[
    objects
    {
      score: a float number,
      context_name: a string like "10203656353524179475_7625_000_7645_000", without ["segment-"] and ["with_camera_labels"]
      object
        {
            box{
              center_x, center_y, center_z,
              width, length, height, heading
            },
            type: TYPE_CLASS, e.g:[TYPE_VEHICLE]       
        }
    },
        
]
```

For centerpoint pickle, it's a dict of dict, and use `--det_input_form ctp_pickle` to specify:
```shell
sequence_name:
{
    timestamp:
    {
        sequence_name: a string like "10203656353524179475_7625_000_7645_000", without ["segment-"] and ["with_camera_labels"]
        timestamp: a float number of time
        score: [N, 1] confidence score
        name: [N, 1] class in number, for {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
        boxes_lidar: [N, 7] bounding boxes
        
    }
}
```

For 3dal pickle, it's a list for each frame, and use `--det_input_form 3dal_pickle` to specify:
```shell
[   
    {
        sequence_name: a string like "10203656353524179475_7625_000_7645_000", without ["segment-"] and ["with_camera_labels"]
        frame_id: an int number of frame for every sequence
        score: [N, 1] confidence score
        name: [N, 1] class in number, for {0:'UNKNOWN', 1: 'Vehicle', 2: 'Pedestrian', 3: 'Sign', 4: 'Cyclist'}
        boxes_lidar: [N, 7] bounding boxes
        pose: [4,4] a numpy float array
    },
    {
      ...
    }
    ...
]
```

### Statistic calculation
Running `python statistic.py` to calculate basic statistical results 

by specify `--det_result_pkl_path`, `--gt_info_pkl_path`,
`--output_path` and `--det_input_form`. Examples are in `script.py`, or you can try demo function than main to check it.

For details, using -h to help:
```shell
optional arguments:
  -h, --help            show this help message and exit

  --det_result_path DET_RESULT_PKL_PATH
                        Path to the prediction result file.

  --gt_info_pkl_path GT_INFO_PKL_PATH
                        Path to ground-truth info pkl file.

  --output_path OUTPUT_PATH

  --evaluate_metrics EVALUATE_METRICS [EVALUATE_METRICS ...]
                        metrics that used in evaluation. support multiple (object, range)

  --petrel_oss_config PETREL_OSS_CONFIG
                        The configuration file of petrel-oss.

  --class_name CLASS_NAME [CLASS_NAME ...]
                        The class names of the detection, default=["Vehicle", "Pedestrian", "Cyclist"]

  --info_with_fakelidar

  --distance_thresh DISTANCE_THRESH

  --det_input_form DET_INPUT_FORM
                        The detection input form, optional: ['bin','ctp_pickle','3dal_pickle']

  --info_path INFO_PATH
                        Sequence name to frame id map

```

### Curves Visualization

Running `python visualizer.py` 

Before that, remember to modify `path` and `labels` in main function in a list of all results you want to visualize.

## Friend links
[Waymo Protos](https://github.com/waymo-research/waymo-open-dataset)

[AP calculation in 2D detection](https://github.com/Cartucho/mAP)