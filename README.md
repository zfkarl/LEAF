# LEAF: Unveiling Two Sides of the Same Coin in Semi-supervised Facial Expression Recognition  

## Introduction
LEAF is a semi-supervised framework for facial expression recognition, which simultaneously enhances representations and pseudo-labels.

![image](https://github.com/zfkarl/LEAF/blob/master/Code_LEAF/LEAF-Framework.png)

## Getting Started
This is an example of how to run LEAF on the RAFDB dataset. 

#### Requirements
- Python==3.6.9
- PyTorch==1.3.0
- Torchvision==0.4.1
  
##### RAFDB 
<pre>python main.py --dataset 'rafdb' --n_labeled 98 --epochs 20 </pre> 


## How To Cite LEAF
If you use this code in your research, please kindly cite the following paper:
```
@article{zhang2024leaf,
  title={LEAF: Unveiling Two Sides of the Same Coin in Semi-supervised Facial Expression Recognition},
  author={Zhang, Fan and Cheng, Zhi-Qi and Zhao, Jian and Peng, Xiaojiang and Li, Xuelong},
  journal={arXiv preprint arXiv:2404.15041},
  year={2024}
}  
```

## Contact
If you have any questions, please contact me through email (zfkarl1998@gmail.com).

## Acknowledgement
Our codebase is built based on Ada-CM and Unified SSL Benchmark (USB). We thank the authors for the nicely organized code!
