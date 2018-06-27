#!/usr/bin/env bash

#START=START EPOCH :END=END EPOCH
#MARGEN=EPOCH margen
#DIR=SAVE Json file 

START="$1"
END="$2"
MARGEN="$3"
DIR="$4"
echo mkdir $4
for((i=$START;i<$END;i=i+$MARGEN));do
  echo python3 deploy.py --num-class 20  --network resnet50 --data-shape 512 --epoch $i
  echo mkdir $4/$i
  echo python demo.py --prefix model/deploy_ssd_resnet50_512 --deploy --class-names data/class.txt --images data/tukuba00006 --thresh 0.1 --nomake 1 --epoch $i --outputdir $4/$i
done



