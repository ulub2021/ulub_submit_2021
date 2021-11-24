# Unbiased Learning on Unknown Bias (submitted as Paper ID 9744 in CVPR2022)

Pytorch implementation of UBNet

# Dependencies and Environment
Dependencies can be installed via anaconda.
```
python>=3.7
pytorch==1.9.0
torchvision==0.10.0
numpy=1.18.5
adamp=0.3.0
opencv-python=4.5.3.56
cudatoolkit=11.1.74
```
Codes were tested locally on the following system configurations:
```
- OS:             Ubuntu 20.04 LTS
- GPU:            NVIDA GeForce RTX
- RAM:            256GB
- CPU:            Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz x 8
- NVIDIA_driver:  460.73.01
```

# Dataset Preparation

## CelebA-HQ
Download the CelebA-HQ and CelebA in the links as below.
- https://github.com/switchablenorms/CelebAMask-HQ
- https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (In-The-Wild Images)

## UTK face 
Download the UTK face dataset in the link as below.
- https://susanqq.github.io/UTKFace/ (aligned & cropped face)

## 9-Class ImageNet
Download the ImageNet dataset from the link as below.
Please follow the usual practice for collecting the ImageNet(ILSVRC2015).
- https://image-net.org/download.php

We expect the directory sturcture to be the following
```
dataset/
  celebA/
    celebA/
      img_celeba/     # CelebA images in wild
    celebA-HQ
      CelebA-HQ-img/  # CelebA-HQ images
  imagenet/
    train/            # Imagenet train images
    val/              # Imagenet val images
  utkface/
    utkcropped/       # UTKface aligned and cropped images
```

We grouped ImageNet to 9 classes as proposed in  [Adversarial examples are not bugs, they are features](https://proceedings.neurips.cc/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf).

We follow the evaluation protocol of [Learning De-biased Representations with Biased Representations](https://arxiv.org/abs/1910.02806) for unbiased accuracy.

# How to run

## Pretrained weights
Download the weights in https://drive.google.com/drive/folders/1_Dkr4CAPxWHbkOU7PIV3gbg9-oqotWRI?usp=sharing

## CelebA-HQ

### Training
base model
```
python base_model/main_base.py -e celebA_train --imagenet_pretrain --data_dir dataset --save_dir exp --data CelebA-HQ --is_train --model vgg11 --batch_size=32 --max_step=20 --lr=0.0001 --cuda --gpu=0 --lr_scheduler step --lr_decay_period=10
```

UBNet
```
python main.py -e celebA_ubnet_train --is_train --ubnet --cuda --checkpoint exp/celebA_train/checkpoint_step_19.pth --data CelebA-HQ --data_dir dataset --save_dir exp --lr=0.0001 --max_step=20 --gpu=0 --batch_size=32 --model vgg11 --lr_scheduler step --lr_decay_period=10
```
### Evaluation
| Method    	| Base Model   	| HEX          	| Rebias       	| **UBNet**       	|
|-----------	|--------------	|--------------	|--------------	|--------------	|
| ACC(UB1)  	| 99.38(±0.31) 	| 92.50(±0.67) 	| 99.05(±0.13) 	| **99.18(±0.18)** 	|
| ACC(UB2)  	| 51.22(±1.73) 	| 50.85(±0.37) 	| 55.57(±1.43) 	| **58.22(±0.64)** 	|
| ACC(test) 	| 75.30(±0.93) 	| 71.68(±0.50) 	| 77.31(±0.71) 	| **78.70(±0.24)** 	|

Note that we have reported the average of 3 results in the paper. The uploaded weight is from one of the 3 experiments; ACC(EB1) 99.29%, ACC(EB2) 58.74%, and hence ACC(Test) 79.02%. 

```
python main.py -e celebA_ubnet_test --ubnet --cuda --checkpoint_orth weights/celebA_ubnet/celeba_ubnet.pth --checkpoint weights/celebA_baseline/celeba_base.pth --data CelebA-HQ --data_dir dataset --save_dir exp --gpu=0 --batch_size=32 --model vgg11
```


## UTK face

### Training
base model
```
# skintone prediction with gender bias / training on UB1
python base_model/main_base.py -e utkface_skintone_gender_ub1_train --data_dir dataset --save_dir exp --is_train --model resnet18 --batch_size=512 --max_step=20 --lr=0.001 --imagenet_pretrain --cuda --gpu=0 --data utkface --data_var=0.2 --bias_type ub1 --cls_type skintone_gender --lr_scheduler step --lr_decay_period=10 --optim AdamP

# skintone prediction with gender bias / training on UB2
python base_model/main_base.py -e utkface_skintone_gender_ub2_train --data_dir dataset --save_dir exp --is_train --model resnet18 --batch_size=512 --max_step=20 --lr=0.001 --imagenet_pretrain --cuda --gpu=0 --data utkface --data_var=0.2 --bias_type ub2 --cls_type skintone_gender --lr_scheduler step --lr_decay_period=10 --optim AdamP

# gender prediction with skintone bias / training on UB1
python base_model/main_base.py -e utkface_gender_skintone_ub1_train --data_dir dataset --save_dir exp --is_train --model resnet18 --batch_size=512 --max_step=20 --lr=0.001 --imagenet_pretrain --cuda --gpu=0 --data utkface --data_var=0.2 --bias_type ub1 --cls_type gender_skintone --lr_scheduler step --lr_decay_period=10 --optim AdamP

# gender prediction with skintone bias / training on UB2
python base_model/main_base.py -e utkface_gender_skintone_ub2_train --data_dir dataset --save_dir exp --is_train --model resnet18 --batch_size=512 --max_step=20 --lr=0.001 --imagenet_pretrain --cuda --gpu=0 --data utkface --data_var=0.2 --bias_type ub2 --cls_type gender_skintone --lr_scheduler step --lr_decay_period=10 --optim AdamP
```

UBNet
```
# skintone prediction with gender bias / training on UB1
python main.py -e utkface_ubnet_skintone_gender_ub1_train --is_train --ubnet --cuda --use_pretrain True --checkpoint exp/utkface_skintone_gender_ub1_train/checkpoint_step_19.pth --data utkface --data_dir dataset --save_dir exp --lr=0.001 --max_step=20 --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub1 --cls_type skintone_gender --model resnet18 --lr_scheduler step --lr_decay_period=10 --optim AdamP

# skintone prediction with gender bias / training on UB2
python main.py -e utkface_ubnet_skintone_gender_ub2_train --is_train --ubnet --cuda --use_pretrain True --checkpoint exp/utkface_skintone_gender_ub2_train/checkpoint_step_19.pth --data utkface --data_dir dataset --save_dir exp --lr=0.001 --max_step=20 --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub2 --cls_type skintone_gender --model resnet18 --lr_scheduler step --lr_decay_period=10 --optim AdamP

# gender prediction with skintone bias / training on UB1
python main.py -e utkface_ubnet_gender_skintone_ub1_train --is_train --ubnet --cuda --use_pretrain True --checkpoint exp/utkface_gender_skintone_ub1_train/checkpoint_step_19.pth --data utkface --data_dir dataset --save_dir exp --lr=0.001 --max_step=20 --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub1 --cls_type gender_skintone --model resnet18 --lr_scheduler step --lr_decay_period=10 --optim AdamP

# gender prediction with skintone bias / training on UB2
python main.py -e utkface_ubnet_gender_skintone_ub2_train --is_train --ubnet --cuda --use_pretrain True --checkpoint exp/utkface_gender_skintone_ub2_train/checkpoint_step_19.pth --data utkface --data_dir dataset --save_dir exp --lr=0.001 --max_step=20 --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub2 --cls_type gender_skintone --model resnet18 --lr_scheduler step --lr_decay_period=10 --optim AdamP
```

### Evaluation
skintone prediction with gender bias

|   Trained  	|      UB1     	|      UB1     	|      UB2     	|      UB2     	|
|:----------:	|:------------:	|:------------:	|:------------:	|:------------:	|
|  Evaluated 	|      UB2     	|     Test     	|      UB1     	|     Test     	|
| Base Model 	| 80.97(±0.79) 	| 86.67(±0.38) 	| 81.43(±0.11) 	| 85.94(±0.77) 	|
| HEX        	| 80.69(±0.80) 	| 86.51(±0.56) 	| 81.01(±1.57) 	| 86.75(±1.57) 	|
| Rebias     	| 82.02(±0.84) 	| 86.24(±0.31) 	| 82.02(±0.90) 	| 86.39(±0.51) 	|
| **UBNet**      	| **82.07(±1.55)** 	| **87.75(±0.58)** 	| **82.69(±0.80)** 	| **87.36(±0.46)** 	|

Note that we have reported the average of 3 results in the paper. The uploaded weight is from one of the 3 experiments; ACC(UB1-UB2) 84.44%, ACC(UB1-Test) 87.91%, ACC(UB2-UB1) 88.66%, ACC(UB2-Test) 84.67%, where Acc([TRAIN_SET]-[VALIDATION_SET]).

```
# trained on UB1
python main.py -e utkface_ubnet_skintone_gender_ub1_test --ubnet --cuda --use_pretrain True --checkpoint_orth weights/utkface_ubnet/utkface_skintone_gender_ub1_ubnet.pth --data utkface --data_dir dataset --save_dir exp --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub1 --cls_type skintone_gender --model resnet18

# trained on UB2
python main.py -e utkface_ubnet_skintone_gender_ub2_test --ubnet --cuda --use_pretrain True --checkpoint_orth weights/utkface_ubnet/utkface_skintone_gender_ub2_ubnet.pth --data utkface --data_dir dataset --save_dir exp --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub2 --cls_type skintone_gender --model resnet18
```
gender prediction with skintone bias

|   Trained  	|      UB1      |	     UB1     	|      UB2     	|      UB2     	|
|:----------:	|:------------:	|:------------:	|:------------:	|:------------:	|
|  Evaluated 	|      UB2     	|     Test     	|      UB1     	|     Test     	|
| Base Model 	| 80.97(±0.79) 	| 86.67(±0.38) 	| 81.43(±0.11) 	| 85.94(±0.77) 	|
| HEX        	| 80.69(±0.80) 	| 86.51(±0.56) 	| 81.01(±1.57) 	| 86.75(±1.57) 	|
| Rebias     	| 82.02(±0.84) 	| 86.24(±0.31) 	| 82.02(±0.90) 	| 86.39(±0.51) 	|
| **UBNet**      	| **82.07(±1.55)** 	| **87.75(±0.58)** 	| **82.69(±0.80)** 	| **87.36(±0.46)** 	|

Note that we have reported the average of 3 results in the paper. The uploaded weight is from one of the 3 experiments; ACC(UB1-UB2) 83.84%, ACC(UB1-Test) 87.74%, ACC(UB2-UB1) 87.66%, and ACC(UB2-Test) 83.60%
```
# trained on UB1
python main.py -e utkface_ubnet_gender_skintone_ub1_test --ubnet --cuda --use_pretrain True --checkpoint_orth weights/utkface_ubnet/utkface_gender_skintone_ub1_ubnet.pth --data utkface --data_dir dataset --save_dir exp --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub1 --cls_type gender_skintone --model resnet18

# trained on UB2
python main.py -e utkface_ubnet_gender_skintone_ub2_test --ubnet --cuda --use_pretrain True --checkpoint_orth weights/utkface_ubnet/utkface_gender_skintone_ub2_ubnet.pth --data utkface --data_dir dataset --save_dir exp --gpu=0 --batch_size=512 --data_var=0.2 --bias_type ub2 --cls_type gender_skintone --model resnet18
```

## 9-Class ImageNet

### Training
base model
```
python base_model/main_base.py -e imagenet_train --data_dir dataset --save_dir exp --data imagenet --is_train --model resnet18 --batch_size=512 --max_step=120 --lr=0.0001 --cuda --gpu=0 --n_class=9 --data imagenet --lr_scheduler cosine
```
UBNet
```
python main.py -e imagenet_ubnet_train --is_train --ubnet --cuda --checkpoint exp/imagenet_train/checkpoint_step_119.pth --data imagenet --data_dir dataset --save_dir exp --lr=0.0001 --max_step=120 --gpu=0 --batch_size=512 --model resnet18 --n_class=9 --lr_scheduler cosine
```

### Evaluation

| Method   	| Base 	      |       SI 	    |        LM	      | RUBi 	| Rebias 	| **UBNet**	|
|----------	|------------	|--------------	|---------------	|------	|--------	|-------	|
| ACC(Biased)   	| 90.8       	| 88.4         	| 64.1          	| 90.5 	| 91.9   	| **91.9**	|
| ACC(Unbiased) 	| 88.8       	| 86.6         	| 62.7          	| 88.6 	| 90.5   	| **91.5**  	|

Note that we have reported the average of 3 results in the paper. The uploaded weight is from one of the 3 experiments; ACC(Biased) 92.19%, ACC(Unbiased) 91.60%.
```
python main.py -e imagenet_ubnet_test --imagenet_pretrain --ubnet --cuda --checkpoint_orth weights/imagenet_ubnet/imagenet_ubnet.pth --checkpoint weights/imagenet_baseline/imagenet_baseline.pth --data imagenet --data_dir dataset --save_dir exp --gpu=0 --batch_size=512 --model resnet18 --n_class=9
```
