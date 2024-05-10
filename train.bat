call conda activate yolov7
python train.py --workers 8 --device 0 --batch-size 8 --data dataset/data.yaml --img 640 640 --cfg cfg/training/yolov7-anchor.yaml --weights '' --name yolov7 --hyp data/new_hyp.yaml --epochs 100
pause