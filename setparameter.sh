#!/usr/bin/env bash

TRAIN="$1"
VAL="$2"

#cp model/ssd_resnet50_512-0000.params model/ssd_resnet50_512-0001.params

mv dataset/names/pascal_voc.names dataset/names/pascal_voc.names.org

cp data/class.txt dataset/names/pascal_voc.names

python tools/prepare_dataset.py --dataset pascal --year $1 --set trainval --target ./data/train.lst

python tools/prepare_dataset.py --dataset pascal --year $2 --set test --target ./data/val.lst --shuffle False

#python3 train.py --batch-size 4 --network resnet50 --lr 0.0001 --pretrained model/ssd_resnet50_512 --data-shape 512 --num-class 20 --class-names data/class.txt --end-epoch 10
 
/mxnet/example/ssd# python3 train.py --batch-size 4 --network resnet50 --lr 0.0001 --pretrained model/ssd_resnet50_512 --data-shape 512 --num-class 20 --class-names data/class.txt --end-epoch 100 --begin-epoch 1 --epoch 0

python3 deploy.py --num-class 20 --epoch 10 --network resnet50 --data-shape 512

#echo python demo.py --prefix model/deploy_ssd_resnet50_512 --epoch 5 --deploy --class-names data/class.txt --images data/VOCdevkit/VOC3003/JPEGImages/frame00067.jpg

echo python demo.py --prefix model/deploy_ssd_resnet50_512 --epoch 12 --deploy --class-names data/class.txt --images data/tukuba00006 --nomake 1

