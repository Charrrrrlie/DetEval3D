# Waymo Submission Format Convertion

## Installation

Refer to 
[Waymo Official Intro](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)

## Personal Info
`location`

{PATH_TO_BAZEL}/bazel_waymo_eval/waymo-od/waymo_open_dataset/metrics/tools/test_submission.txtpb

```
# Modify this file to fill in your information and then
# run create_submission to generate a submission file.

task: DETECTION_3D
account_name: "demo.demo.demo@gmail.com"
# Change this to your unique method name. Max 25 chars.
unique_method_name: "A_UNI_METHOD"

authors: "Charles"

affiliation: "HomeStay University"
description: "Demo. TBD"

method_link: "A link to your method, e.g. paper, code etc."

# This is optional. Set this to evaluate your model latency.
# docker_image_source: "Link to the latency submission Docker image stored in Google Storage bucket or pushed to Google Container/Artifact Registry. See submission.proto for # # details."

# See submission.proto for allowed types.
sensor_type: LIDAR_TOP

number_past_frames_exclude_current: 0
number_future_frames_exclude_current: 0

object_types: TYPE_VEHICLE
object_types: TYPE_PEDESTRIAN
object_types: TYPE_CYCLIST

# Self-reported latency in seconds. This is NOT shown on the leaderboard for
# now. But it is still recommended to set this. That is needed to evaluate
# your model latency on our server.
latency_second: -1
```


**unique_method_name** should be less than 25 chars.

**account_name** should be a `GMAIL` address.


## Generating Model

`location`
```
cd {PATH_TO_BAZEL}/bazel_waymo_eval/waymo-od

bazel-bin/waymo_open_dataset/metrics/tools/create_submission

--input_filenames={.bin file path}
--output_filename={/model_name file path} # will generate 'model_name00', 'model_name01' files in output path.
--submission_filename={txtpb file path} # path of .txtpb 
```

`e.g.`

```
bazel-bin/waymo_open_dataset/metrics/tools/create_submission  
--input_filenames='~/Documents/Prjs/eval_check/val_res/1sweep1stg.bin' 
--output_filename='~/Documents/Prjs/submission/model' 
--submission_filename='~/Documents/Prjs/bazel_waymo_eval/waymo-od/waymo_open_dataset/metrics/tools/test_submission.txtpb'
```

## Getting Compressed

```
cd {output_filename}

tar cvf my_model.tar model*

gzip my_model.tar
```

## Submission 

[Waymo 3D Perception Submission Website](https://waymo.com/open/challenges/2021/real-time-3d-prediction/)

