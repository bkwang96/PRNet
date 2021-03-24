# Polynomial Regression Network for Variable-Number Lane Detection
This repository is the official implementation of "Polynomial Regression Network for Variable-Number Lane Detection" (accepted by ECCV 2020). It is designed for lane detection task.
[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf) 

## Overall Network Architecture
Lane detection is a fundamental yet challenging task in autonomous driving and intelligent traffic systems due to perspective projection and occlusion. Most of previous methods utilize semantic segmentation to identify the regions of traffic lanes in an image, and then adopt some curve-fitting method to reconstruct the lanes. 
In this work, we propose to use polynomial curves to represent traffic lanes and then propose a novel polynomial regression network (PRNet) to directly predict them, where semantic segmentation is not involved. Specifically, PRNet consists of one major branch and two auxiliary branches: 
(1) polynomial regression to estimate the polynomial coefficients of lanes, (2) initialization classification to detect the initial retrieval point of each lane, and (3) height regression to determine the ending point of each lane. Through the cooperation of three branches, PRNet can detect variable-number of lanes and is highly effective and efficient.The overall architecture of PRNet is shown as below.
<img src="./img/network_structure.png" width="1000"/>

## Lane Detection Performance
We experimentally evaluate the proposed PRNet on two popular benchmark datasets: TuSimple and CULane. The results show that our method significantly outperforms the previous state-of-the-art methods in terms of both accuracy and speed. 
|             | Plateform                | TuSimple (Acc)           | CULane (F1-score)      | Time (FPS)|
|:-----------:|:------------------------:|:------------------------:|:----------:|:---------:|
|   LaneNet  |           GTX1080Ti       |          96.38%          |   -        |  52  |
|    SCNN    |           GTX1080Ti       |          96.53%          |    71.6    |   20   | 
|    SAD     |          GTX1080Ti        |           96.64%         |    71.8    |    79   |   
|    FastDraw  |          GTX1080        |           95.2%          |    -       |    90   |       
| PRNet (BiSeNet) |          GTX1080Ti   |           97.18%         |    74.8    |   110   |  
| PRNet (ERFNet) |           GTX1080Ti   |          97.00%          |    76.4    |    81   |


Some visual examples of our PRNet on several images are presented as follows.
<img src="./img/lane_detection.gif" width="1000"/>
<img src="./img/show.png" width="1000"/>


## Usage

The code has been tested on pytorch=1.0.1 and python3.6. Please refer to `requirements.txt` for detailed information. You should install the packages with command bellow

**To Install python packages**
````
conda install --yes --file requirements.txt
````

We provide a model trained on CULane dataset and some demo images.
The trained model can be found in [prnet.pth](https://pan.baidu.com/s/1fiADGSgiS1zbQCa-jYaUwg).
For example, visualize our proposed method on the demo images, you can try:
````
CUDA_VISIBLE_DEVICES=0 python demo.py
````


## License and Citation
This software and associated documentation files (the "Software"), and the research paper (Polynomial Regression Network for Variable-Number Lane Detection) including but not limited to the figures, and tables (the "Paper") are provided for academic research purposes only and without any warranty. Any commercial use requires my consent. When using any parts of the Software or the Paper in your work, please cite the following paper:
```
@inproceedings{wang2020polynomial,
  title={Polynomial Regression Network for Variable-Number Lane Detection},
  author={Wang, Bingke and Wang, Zilei and Zhang, Yixin},
  booktitle={European Conference on Computer Vision},
  pages={719--734},
  year={2020},
  organization={Springer}
}
```
