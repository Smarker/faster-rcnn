# Training and Prediction Scripts for CNTK Faster-RCNN

These script removes a lot of the manual edits needed for the config files, downloading of models, and setup of annotations necessary to run the CNTK detection example.
Place this script in `/cntk/Examples/Images/Detection`.

## Training
```
python train.py --tagged-images <image dir> --num-train 200 --num-test 5
```

## Predictions
```
python predict.py --tagged-images <image dir> --num-train 200 --num-test 5 --model-path /cntk/Examples/Image/Detection/FasterRCNN/Output/faster_rcnn_eval_AlexNet_e2e.model
```