# Training and Prediction Scripts for CNTK Faster-RCNN

These script removes a lot of the manual edits needed for the config files, downloading of models, and setup of annotations necessary to run the [CNTK detection example](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FasterRCNN).

Once you have [correctly set up CNTK](https://github.com/jcjimenez/CNTK-docker/blob/master/ubuntu-14.04/version_2/cpu/runtime/python-3/Dockerfile), place `train.py` and `predict.py` in `/cntk/Examples/Images/Detection`.

| tag             | value expected      |
| ----------------| --------------------|
| tagged-images   | Provide a path to the directory containing your custom images prepared for CNTK object detection. The directory should contain information formatted as described [here](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN#train-on-your-own-data). I recommend using the [VoTT tool](https://github.com/Microsoft/VoTT) to create the formatted directory. |
| num-train       | The number of training images used to train the model. |
| num-epochs      | The number of `epochs` used to train the model. One `epoch` is one complete training cycle on the training set. |
| num-test        | The number images to test. |
| model-path      | The path to your trained model. To get a trained model, run `train.py`. The training script will output the directory where your trained model is stored. Also, you can look at the model path below, since that is the expected path where your model will reside when you run the training. |
| conf-threshold  | The `confidence threshold` used to determine when bounding boxes around detected objects are drawn. A confidence threshold of `0` will draw all bounding boxes determined by CNTK. A threshold of `1` will only draw a bounding box around the exact location you had originally drawn a bounding box, i.e. you trained and tested on the same image. Provide a `float` falling between 0 and 1. |

## Training
```
python train.py 
  --tagged-images /CustomImages
  --num-train 200 
  [--num-epochs 3]
```

## Predictions
```
python predict.py 
  --tagged-images /CustomImages 
  --num-test 5
  --model-path /cntk/Examples/Image/Detection/FasterRCNN/Output/faster_rcnn_eval_AlexNet_e2e.model [--conf-threshold 0.82]
```