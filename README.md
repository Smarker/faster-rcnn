# Training and Prediction Scripts for CNTK Faster-RCNN

These script removes a lot of the manual edits needed for the config files, downloading of models, and setup of annotations necessary to run the [CNTK detection example](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FasterRCNN).

You can use the CNTK dockerfile located [here](https://github.com/CatalystCode/CNTK-faster-rcnn) and then place `train.py` and `predict.py` in `/cntk/Examples/Images/Detection`.

## Before Training

Set up your custom image directory in the [format required by CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/object-detection-using-fast-r-cnn#train-on-your-own-data). The directory will be structured as:

```
.
├── negative
│ ├── a0vqvtsowhoubmczrq4q.jpg
│ ├── avhrylgho1dg1ns6q6wb.jpg
│ └── ictav2a3ahahv2pcusck.jpg
├── positive
│ ├── aljrnxc0ttj07dc2riel.bboxes.labels.tsv
│ ├── aljrnxc0ttj07dc2riel.bboxes.tsv
│ ├── aljrnxc0ttj07dc2riel.jpg
│ ├── icde4ql7u7clfv3lmani.bboxes.labels.tsv
│ ├── icde4ql7u7clfv3lmani.bboxes.tsv
│ └── icde4ql7u7clfv3lmani.jpg
└── testImages
 ├── bz7mfvk1etwl0rzofewu.bboxes.labels.tsv
 ├── bz7mfvk1etwl0rzofewu.bboxes.tsv
 └── bz7mfvk1etwl0rzofewu.jpg
```

## Training and Prediction Parameters

| tag                       | value expected      |
| --------------------------| --------------------|
| gpu                       | To use `gpu` in either training or prediction, specify `1`. Otherwise, `cpu` will be used. |
| tagged-images             | Provide a path to the directory containing your custom images prepared for CNTK object detection. The directory should contain information formatted as described [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN#train-on-your-own-data). I recommend using the [VoTT tool](https://github.com/Microsoft/VoTT) to create the formatted directory. |
| num-train                 | The number of training images used to train the model. |
| num-epochs                | The number of `epochs` used to train the model. One `epoch` is one complete training cycle on the training set. |
| num-test                  | The number images to test. |
| model-path                | The path to your trained model. To get a trained model, run `train.py`. The training script will output the directory where your trained model is stored. Also, you can look at the model path below, since that is the expected path where your model will reside when you run the training. |
| conf-threshold            | The `confidence threshold` used to determine when bounding boxes around detected objects are drawn. A confidence threshold of `0` will draw all bounding boxes determined by CNTK. A threshold of `1` will only draw a bounding box around the exact location you had originally drawn a bounding box, i.e. you trained and tested on the same image. Provide a `float` falling between `0` and `1`. The `default` confidence theshold is `0`. |

## Training
```
python train.py
  [--gpu 1] 
  --tagged-images /CustomImages
  --num-train 200 
  [--num-epochs 3]
```

### Training Output

After you run training, `/cntk/Examples/Image/Detection/FasterRCNN/Output/` will contain one new item:
* `faster_rcnn_eval_AlexNet_e2e.model` - trained model used as input for predictions

## Predictions
```
python predict.py
  [--gpu 1] 
  --tagged-images /CustomImages 
  --num-test 5
  --model-path /cntk/Examples/Image/Detection/FasterRCNN/Output/faster_rcnn_eval_AlexNet_e2e.model
  [--conf-threshold 0.82]
```

### Prediction Output
After you run your predictions, `/cntk/Examples/Image/Detection/FasterRCNN/Output/` will contain two new items: 

* `CustomImages directory` - contains custom images with bounding boxes drawn on detected objects
* `custom_images_output.json` - json output of `bounding boxes`, `confidence levels`, and `class names` for each image

#### custom_images_output.json

```
{ "images": 
  {
    "/Reverb/labelled-guitars/testImages/adfvzfswiuv0a1erna5k.jpg": {
      "class": "body",
      "bounding_boxes": [
        {"confidence_level": "0.536132", "bounding_box": {"x1": 317, "x2": 799, "y1": 65, "y2": 493}},
        {"confidence_level": "0.632784", "bounding_box": {"x1": 0, "x2": 389, "y1": 167, "y2": 507}},
        {"confidence_level": "0.767789", "bounding_box": {"x1": 0, "x2": 799, "y1": 102, "y2": 595}},
        {"confidence_level": "0.588904", "bounding_box": {"x1": 527, "x2": 780, "y1": 96, "y2": 579}},
        {"confidence_level": "0.743675", "bounding_box": {"x1": 0, "x2": 512, "y1": 196, "y2": 718}}
      ]
    }, 
    "/Reverb/labelled-guitars/testImages/aayjfcuulg99o3zpctux.jpg": {
      "class": "body", 
      "bounding_boxes": [
        {"confidence_level": "0.872948", "bounding_box": {"x1": 79, "x2": 582, "y1": 136, "y2": 764}},
        {"confidence_level": "0.629768", "bounding_box": {"x1": 158, "x2": 594, "y1": 180, "y2": 552}}
      ]
    },
    "/Reverb/labelled-guitars/testImages/caaqxk85v3izwweqvbsi.jpg": {
      "class": "__background__",
      "bounding_boxes": []
    }
  }
}
```