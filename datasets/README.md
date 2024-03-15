## Preparing Few-Shot Segmentation Datasets
Download following datasets:


> #### 1. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from this Google Drive: [train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing), [val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing). (and locate both train2014/ and val2014/ under annotations/ directory).

> #### 2. FSS-1000
> Download FSS-1000 images and annotations from this [Google Drive](https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3/view?usp=sharing).

> #### 3. LVIS-92<sup>i</sup>
> Download COCO2017 train/val images: 
> ```bash
> wget http://images.cocodataset.org/zips/train2017.zip
> wget http://images.cocodataset.org/zips/val2017.zip
> ```
> Download LVIS-92<sup>i</sup> extended mask annotations from our Google Drive: [lvis.zip](https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view?usp=sharing).


> #### 4. PACO-Part
> Download COCO2017 train/val images: 
> ```bash
> wget http://images.cocodataset.org/zips/train2017.zip
> wget http://images.cocodataset.org/zips/val2017.zip
> ```
> Download PACO-Part extended mask annotations from our Google Drive: [paco.zip](https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view?usp=sharing).

> #### 5. Pascal-Part
> Download VOC2010 train/val images: 
> ```bash
> wget http://roozbehm.info/pascal-parts/trainval.tar.gz
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
> ```
> Download Pascal-Part extended mask annotations from our Google Drive: [pascal.zip](https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view?usp=sharing).

Create a directory 'datasets' for the above datasets and appropriately place each dataset to have following directory structure:

    datasets/
    ├── COCO2014/           
    │   ├── annotations/
    │   │   ├── train2014/
    │   │   └── val2014/
    │   ├── train2014/
    │   ├── val2014/
    │   └── splits
    │   │   ├── trn/
    │   │   └── val/
    ├── FSS-1000/
    │   ├── data/
    │   │   ├── ab_wheel/
    │   │   ├── ...
    │   │   └── zucchini/
    │   └── splits/   
    │   │   ├── test.text
    │   │   ├── trn.txt
    │   │   └── val.txt
    ├── LVIS/
    │   ├── coco/
    │   │   ├── train2017/
    │   │   └── val2017/
    │   ├── lvis_train.pkl
    │   └── lvis_val.pkl
    ├── PACO-Part/
    │   ├── coco/
    │   │   ├── train2017/
    │   │   └── val2017/
    │   ├── paco/
    │   │   ├── paco_part_train.pkl
    │   │   └── paco_part_val.pkl
    ├── Pascal-Part/  
    │   ├── VOCdevkit/
    │   │   ├── VOC2010/
    │   │   │   ├── Annotations_Part_json_merged_part_classes/
    │   │   │   ├── JPEGImages/
    │   │   │   └── all_obj_part_to_image.json

