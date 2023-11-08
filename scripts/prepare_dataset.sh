#!/usr/bin/env bash

help() {
    echo "------------------------------------------------------------------------------------"
    echo "$0 - prepare datasets for training and inference of SSSegmentation."
    echo "------------------------------------------------------------------------------------"
    echo "Usage:"
    echo "    bash $0 <dataset name>"
    echo "Options:"
    echo "    <dataset name>: The dataset name you want to download and prepare."
    echo "                    The keyword should be in ['ade20k', 'pascalvoc', 'cityscapes']"
    echo "    <-h> or <--help>: Show this message."
    echo "Examples:"
    echo "    If you want to fetch ADE20k dataset, you can run 'bash $0 ade20k'."
    echo "    If you want to fetch Cityscapes dataset, you can run 'bash $0 cityscapes'."
    echo "------------------------------------------------------------------------------------"
    exit 0
}

DATASET=$1
OPT="$(echo $DATASET | tr '[:upper:]' '[:lower:]')"
if [ "$OPT" == "-h" ] || [ "$OPT" == "--help" ] || [ "$OPT" == "" ]; then
    help
elif [[ "$OPT" == "ade20k" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/ADE20k.tar.gz
        tar zxvf ADE20k.tar.gz
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf ADE20k.tar.gz
elif [[ "$OPT" == "pascalvoc" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.001
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.002
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/VOCdevkit.zip.003
        7z x VOCdevkit.zip.001
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf VOCdevkit.zip.001 VOCdevkit.zip.002 VOCdevkit.zip.003
elif [[ "$OPT" == "cityscapes" ]]; then
    {
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.zip
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z01
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z02
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z03
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z04
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z05
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z06
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z07
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z08
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z09
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z10
        wget https://github.com/SegmentationBLWX/modelstore/releases/download/ssseg_datasets/CityScapes.z11
        7z x CityScapes.zip
    } || {
        echo "Fail to download ${DATASET} dataset."
        exit 0
    }
    rm -rf CityScapes.zip CityScapes.z01 CityScapes.z02 CityScapes.z03 CityScapes.z04 \
           CityScapes.z04 CityScapes.z05 CityScapes.z06 CityScapes.z07 CityScapes.z08 \
           CityScapes.z09 CityScapes.z10 CityScapes.z11 
else
    echo "Preparing dataset ${DATASET} is not supported in this script now."
    exit 0
fi
echo "Download ${DATASET} done."