import os, sys, json
from os import path

import numpy as np
from PIL import Image
from cntk import load_model
from easydict import EasyDict as edict
    
if __name__ == "__main__":
    import argparse
    import os
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='FRCNN Training')
    
    parser.add_argument('--tagged-images', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing tagged image(s) in jpg format', required=True)
    
    parser.add_argument('--model-path', type=str, metavar='<path>',
                        help='Path to pretrained model file', required=True)

    parser.add_argument('--num-test', type=int, metavar='<integer>',
                        help='Number of testing images. For example: 5',
                        required=True)

    parser.add_argument('--conf-threshold', type=float, metavar='<float>',
                        help='Confidence threshold when drawing bounding boxes. Choose a float in this range: [0, 1).', required=False)

    args = parser.parse_args()

    from utils.config_helpers import merge_configs
    import utils.od_utils as od

    available_detectors = ['FasterRCNN']

    def download_base_model():
        print("\nDownloading AlexNet base model...")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder,"..", "..", "..", "PretrainedModels"))
        from download_model import download_model_by_name
        download_model_by_name("AlexNet_ImageNet_Caffe")

    download_base_model()

    def get_configuration():
        from utils.config_helpers import merge_configs
        from FasterRCNN_config import cfg as detector_cfg
        from utils.configs.AlexNet_config import cfg as network_cfg
        from utils.configs.custom_image_config import cfg as dataset_cfg
        return merge_configs([detector_cfg, network_cfg, dataset_cfg])

        
    def run_faster_rcnn():
        print("Running training")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "FasterRCNN"))
        from cntk import load_model
        from FasterRCNN_train import prepare
        from FasterRCNN_eval import compute_test_set_aps
        import numpy as np

        cfg = get_configuration()
        prepare(cfg, False)

        cfg["DATA"].NUM_TEST_IMAGES = args.num_test
        cfg["CNTK"].MAKE_MODE = True
        if not (args.conf_threshold is None):
            cfg.RESULTS_NMS_CONF_THRESHOLD = args.conf_threshold
        
        trained_model = load_model(args.model_path)
        eval_results = compute_test_set_aps(trained_model, cfg)
       
        for class_name in eval_results: print('Average precision (AP) for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
        print('Mean average precision (AP) = {:.4f}'.format(np.nanmean(list(eval_results.values()))))
           
    run_faster_rcnn() 
