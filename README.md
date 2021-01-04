# Polynomial Regression Network for Variable-Number Lane Detection
This repository is the official implementation of "Polynomial Regression Network for Variable-Number Lane Detection" (accepted by ECCV 2020). It is designed for lane detection task.

[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630698.pdf) 


## Install & Requirements
The code has been tested on pytorch=1.0.1 and python3.6. Please refer to `requirements.txt` for detailed information.

**To Install python packages**
````
conda install --yes --file requirements.txt
````


## Demo
We provide a trained model and some demo images.

For example, visualize our proposed method on the demo images, you can try:
````
CUDA_VISIBLE_DEVICES=0 python demo.py
````




## Citation
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
