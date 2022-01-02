import os

os.system(" python statistic.py --gt_info_pkl_path \'./val_res/validation_gt.pkl\' --det_result_pkl_path \'./val_res/1sweep1stg_detection_pred.bin\' --output_path \'./output/1sweep1stg_class\'")
os.system(" python statistic.py --gt_info_pkl_path \'./val_res/validation_gt.pkl\' --det_result_pkl_path \'./val_res/2sweep2stgTTA_detection_pred.bin\' --output_path \'./output/2sweep2stg_class\'")
os.system(" python statistic.py --gt_info_pkl_path \'./val_res/validation_gt.pkl\' --det_result_pkl_path \'./val_res/3sweep2stg_future_flipTTA_detection_pred.bin\' --output_path \'./output/3sweep2stg_future_class\'")
os.system(" python statistic.py --gt_info_pkl_path \'./val_res/validation_gt.pkl\' --det_result_pkl_path \'./val_res/5sweep1stg_detection_pred.bin\' --output_path \'./output/5sweep1stg_class\'")
os.system(" python statistic.py --gt_info_pkl_path \'./val_res/validation_gt.pkl\' --det_result_pkl_path \'./val_res/3sweep1stg_detection_pred.bin\' --output_path \'./output/3sweep1stg_class\'")