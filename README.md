# Adversarial Robustness through the Lens of Convolutional Filters
Paul Gavrikov, Janis Keuper

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

Presented at: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) - The Art of Robustness: Devil and Angel in Adversarial Machine Learning Workshop

[Paper](https://openaccess.thecvf.com/content/CVPR2022W/ArtOfRobust/html/Gavrikov_Adversarial_Robustness_Through_the_Lens_of_Convolutional_Filters_CVPRW_2022_paper.html) | [ArXiv](https://arxiv.org/abs/2204.02481) | [HQ Poster](https://zenodo.org/record/6760593)

**This is a specialized article on Robustness, derived from our main paper: https://github.com/paulgavrikov/CNN-Filter-DB/**

Abstract: *Deep learning models are intrinsically sensitive to distribution shifts in the input data. In particular, small, barely perceivable perturbations to the input data can force models to make wrong predictions with high confidence. An common defense mechanism is regularization through adversarial training which injects worst-case perturbations back into training to strengthen the decision boundaries, and to reduce overfitting. In this context, we perform an investigation of 3x3 convolution filters that form in adversarially-trained models. Filters are extracted from 71 public models of the linf-RobustBench CIFAR-10/100 and ImageNet1k leaderboard and compared to filters extracted from models built on the same architectures but trained without robust regularization. We observe that adversarially-robust models appear to form more diverse, less sparse, and more orthogonal convolution filters than their normal counterparts. The largest differences between robust and normal models are found in the deepest layers, and the very first convolution layer, which consistently and predominantly forms filters that can partially eliminate perturbations, irrespective of the architecture.*


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

![Activation of first stage filters](./plots/primary_thresholding.png)


## Data

Download the dataset from here https://zenodo.org/record/6414075.

## Citation 

If you find our work useful in your research, please consider citing:

```
@InProceedings{Gavrikov_2022a_CVPR,
    author    = {Gavrikov, Paul and Keuper, Janis},
    title     = {Adversarial Robustness Through the Lens of Convolutional Filters},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {139-147}
}
```
Dataset:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6414075.svg)](https://doi.org/10.5281/zenodo.6414075)

### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].
