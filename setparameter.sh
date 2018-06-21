#!/usr/bin/env bash

TRAIN="$1"
VAL="$2"

cp model/ssd_resnet50_512-0000.params model/ssd_resnet50_512-0001.params

mv dataset/names/pascal_voc.names dataset/names/pascal_voc.names.org

cp data/class.txt dataset/names/pascal_voc.names

python tools/prepare_dataset.py --dataset pascal --year $1 --set trainval --target ./data/train.lst

python tools/prepare_dataset.py --dataset pascal --year $2 --set test --target ./data/val.lst --shuffle False

python3 train.py --batch-size 4 --network resnet50 --lr 0.0001 --pretrained model/ssd_resnet50_512 --data-shape 512 --num-class 20 --class-names data/class.txt --end-epoch 10
 

python3 deploy.py --num-class 20 --epoch 10 --network resnet50 --data-shape 512

echo demo.py --images --class-names
echo python demo.py --prefix model/deploy_ssd_resnet50_512 --epoch 5 --deploy --class-names data/class.txt --images data/VOCdevkit/VOC3003/JPEGImages/frame00067.jpg

