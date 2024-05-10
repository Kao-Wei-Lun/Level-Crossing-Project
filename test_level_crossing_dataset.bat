call conda activate yolov7
python test.py --data level_crossing_dataset/data.yaml --img 640 --batch 8 --conf 0.001 --iou 0.5 --device 0 --weights best.pt --name yolov7_640_val_level_crossing_dataset
pause