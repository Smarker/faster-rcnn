import os, sys, json
from os import path

import numpy as np
from PIL import Image
from cntk import load_model
from easydict import EasyDict as edict
    
def get_classes_description(model_file_path, classes_count):
    model_dir = path.dirname(model_file_path)
    classes_names = {}
    model_desc_file_path = path.join(model_dir, 'class_map.txt')
    if not path.exists(model_desc_file_path):
        # use default parameter names:
        return [ "class_{}".format(i) for i in range(classes_count)]
    with open(model_desc_file_path) as handle:
        class_map = handle.read().strip().split('\n')
        return [class_name.split('\t')[0] for class_name in class_map]

if __name__ == "__main__":
    import argparse
    import os
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='FRCNN Training')
    
    parser.add_argument('--tagged-images', type=str, metavar='<path>',
                        help='Path to image file or to a directory containing tagged image(s) in jpg format', required=True)
    
    parser.add_argument('--output', type=str, metavar='<directory path>',
                        help='Path to output directory', required=False)

    parser.add_argument('--model-path', type=str, metavar='<path>',
                        help='Path to pretrained model', required=False)

    parser.add_argument('--num-train', type=int, metavar='<integer>',
                        help='Number of training images. For example: 200',
                        required=True)

    parser.add_argument('--num-test', type=int, metavar='<integer>',
                        help='Number of testing images. For example: 5',required=True)

    parser.add_argument('--conf-thresh', type=float, metavar='<float>',
                        help='Enter a confidence threshold to draw bounding boxes on objects. For example 0.82', required=False)

    parser.add_argument('--json-output', type=str, metavar='<file path>',
                        help='Path to output JSON file', required=False)

    args = parser.parse_args()

#    from FasterRCNN.FasterRCNN_train import FasterRCNN_Trainer
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

    def create_custom_image_annotations():
        print("\nCreating custom image annotations...")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "utils", "annotations"))
        from annotations_helper import create_class_dict, create_map_files
        data_set_path = args.tagged_images
        
        class_dict = create_class_dict(data_set_path)
        create_map_files(data_set_path, class_dict, training_set=True)
        create_map_files(data_set_path, class_dict, training_set=False)

    create_custom_image_annotations()

    def create_custom_config():
        print("\nCreating custom config for your image set in /cntk/Examples/Images/Detection/utils/configs...")
        with open("./utils/configs/custom_image_config.py","w+") as config:
            config.write("""from easydict import EasyDict as edict
__C = edict()
__C.DATA = edict()
cfg = __C
__C.DATA.DATASET = \"CustomImages\"
__C.DATA.MAP_FILE_PATH = %s
__C.DATA.CLASS_MAP_FILE = \"class_map.txt\"
__C.DATA.TRAIN_MAP_FILE = \"train_img_file.txt\"
__C.DATA.TRAIN_ROI_FILE = \"train_roi_file.txt\"
__C.DATA.TEST_MAP_FILE = \"test_img_file.txt\"
__C.DATA.TEST_ROI_FILE = \"test_roi_file.txt\"
__C.DATA.NUM_TRAIN_IMAGES = %s
__C.DATA.NUM_TEST_IMAGES = %s
__C.DATA.PROPOSAL_LAYER_SCALES = [4, 8, 12]
__C.roi_min_side_rel = 0.04
__C.roi_max_side_rel = 0.4
__C.roi_min_area_rel = 2 * __C.roi_min_side_rel * __C.roi_min_side_rel
__C.roi_max_area_rel = 0.33 * __C.roi_max_side_rel * __C.roi_max_side_rel
__C.roi_max_aspect_ratio = 4.0
""" % (args.tagged_images, args.num_train, args.num_test))

    create_custom_config()

    def run_faster_rcnn():
        print("Running training")
        base_folder = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(base_folder, "FasterRCNN"))
        from cntk import load_model
        from run_faster_rcnn import get_configuration
        from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
        from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
        cfg = get_configuration()
        prepare(cfg, False)
        
        if not (args.model_path is None):
            trained_model = load_model(args.model_path)
            eval_results = compute_test_set_aps(trained_model, cfg)
        else:
            trained_model = train_faster_rcnn(cfg)
    
    run_faster_rcnn() 
