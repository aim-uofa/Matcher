<div align="center">

<h1>Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching </h1>

[Yang Liu](https://scholar.google.com/citations?user=9JcQ2hwAAAAJ&hl=en)<sup>1*</sup>, &nbsp; 
Muzhi Zhu<sup>1*</sup>, &nbsp; 
Hengtao Li<sup>1*</sup>, &nbsp;
[Hao Chen](https://stan-haochen.github.io/)<sup>1</sup>, &nbsp;
[Xinlong Wang](https://www.xloong.wang/)<sup>2</sup>, &nbsp;
[Chunhua Shen](https://cshen.github.io/)<sup>1</sup>

<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/), &nbsp;
<sup>2</sup>[Beijing Academy of Artificial Intelligence](https://www.baai.ac.cn/english.html)


</div>

## 🚀 Overview
<div align="center">
<img width="800" alt="image" src="figs/framework.png">
</div>

## 📖 Description

Powered by large-scale pre-training, vision foundation models exhibit significant potential in open-world image understanding. Even though individual models have limited capabilities, 
combining multiple such models properly can lead to positive synergies and unleash their full potential. In this work, we present **Matcher**, which segments anything with one shot 
by integrating an all-purpose feature extraction model and a class-agnostic segmentation model. Naively connecting the models results in unsatisfying performance, e.g., the models tend 
to generate matching outliers and false-positive mask fragments. To address these issues, we design a bidirectional matching strategy for accurate cross-image semantic dense matching 
and a robust prompt sampler for mask proposal generation. In addition, we propose a novel instance-level matching strategy for controllable mask merging. The proposed Matcher method 
delivers impressive generalization performance across various segmentation tasks, all without training. For example, it achieves 52.7% mIoU on COCO-20<sup>i</sup> for one-shot semantic 
segmentation, surpassing the state-of-the-art specialist model by 1.6%. In addition, our visualization results show open-world generality and flexibility on images in the wild.


## 🗓️ TODO
- [ ] Online Demo 
- [ ] Release code and models


## 🖼️ Demo

### One-Shot Semantic Segmantation

<div align="center">
<img width="800" alt="image" src="figs/oss.png">
</div>

### One-Shot Object Part Segmantation

<div align="center">
<img width="800" alt="image" src="figs/part.png">
</div>

### Cross-Style Object and Object Part Segmentation

<div align="center">
<img width="800" alt="image" src="figs/cross_style.png">
</div>

### Controllable Mask Output

<div align="center">
<img width="800" alt="image" src="figs/control.png">
</div>


### Video Object Segmentation

<div align="center">
<img width="800" alt="image" src="figs/vos.png">
  
https://github.com/aim-uofa/Matcher/assets/119775808/49c118d6-d01a-4782-a197-57ef97daa960

</div>


## 🎫 License

The content of this project itself is licensed under [LICENSE](LICENSE).

## 🖊️ Citation


If you find this project useful in your research, please consider cite:


```BibTeX
@article{liu2023matcher,
  title={Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching},
  author={Liu, Yang and Zhu, Muzhi and Li, Hengtao and Chen, Hao and Wang, Xinlong and Shen, Chunhua},
  journal={arXiv preprint arXiv:2305.13310},
  year={2023}
}
```
