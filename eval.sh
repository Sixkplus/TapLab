#!/bin/bash
rm /4T/junyi/cityscapes/cityscapesScripts-master/results/*
cp ./Test_id/*id* /4T/junyi/cityscapes/cityscapesScripts-master/results/
python /4T/junyi/cityscapes/cityscapesScripts-master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
