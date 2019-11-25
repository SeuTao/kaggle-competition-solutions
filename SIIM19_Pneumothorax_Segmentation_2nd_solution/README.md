# SIIM-ACR Pneumothorax Segmentation

# 2nd place solution 

## Model
### segmentation
- unet (se-resnext50, se-resnext101) from [\[pretrained-models\]](https://github.com/Cadene/pretrained-models.pytorch)
- unet (efficientnet-b3, efficientnet-b5) from [\[EfficientNet-PyTorch\]](https://github.com/lukemelas/EfficientNet-PyTorch)
- deeplabv3 (se-resnext50) from [\[semantic-segmentation\]](https://github.com/NVIDIA/semantic-segmentation)
### classification 
- unet (se-resnext50, se-resnext101) from [\[pretrained-models\]](https://github.com/Cadene/pretrained-models.pytorch)
- unet (efficientnet-b3) from  [\[EfficientNet-PyTorch\]](https://github.com/lukemelas/EfficientNet-PyTorch)
### Augmentations
Used following transforms from \[[albumentations\]](https://github.com/albu/albumentations)
```python
RESIZE_SIZE = 1024 # or 768
train_transform = albumentations.Compose([
        albumentations.Resize(RESIZE_SIZE, RESIZE_SIZE),
        albumentations.OneOf([
            albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        ]),
        albumentations.OneOf([
            albumentations.Blur(blur_limit=4, p=1),
            albumentations.MotionBlur(blur_limit=4, p=1),
            albumentations.MedianBlur(blur_limit=4, p=1)
        ], p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])
```
### Loss Function
#### classification 

- Cls loss: BCE + focal loss
- Seg loss: BCE 
#### segmentation

- Seg loss: dice loss
## Training Method
### image size
Train image size: 768 or 1024
test image size: 768 or 1024
### stochastic weight averaging
\[[swa\]](https://github.com/timgaripov/swa)
### ensemble
- classification: stacking 
- segmentation: average 
### pseudo labels
We predict masks on chexpert dataset using trained model, 
and then add these pseudo labels(about 1000) to the network and fine-tune model. There was no significant improvement.

## File structure
    ├── configs
    │   ├── seg_path_configs.json
    ├── data              
    │   ├── chexpert_data
    │   │   ├── chexpert_img
    │   │   ├── chexpert_pesudo_label
    │   ├── cls_fold_5_all_images
    │   │   ├── 5_fold_file
    │   ├── cls_fold_5_all_images_p_label_chexpert
    │   │   ├── 5_fold_file
    |   ├── competition_data
    │   │   ├── test_png
    │   │   ├── train_png
    │   │   ├── sample_submission.csv
    │   │   ├── train-rle.csv
    │   └── det_fold_5
    │   │   ├── 5_fold_file
    ├── models_snapshot
    ├── result
    ├── semantic_segmentation
    │   ├── deeplab_model
    ├── src_unet_cls
    │   ├── classification_model_code
    ├── src_unet_seg
    │   ├── segmentation_model_code
    ├── README.md
    └── requirements.txt

## Install
```bash
pip install -r requirements.txt
```

## How to run code
segmentation model training:
```bash
cd src_unet_seg
```
```
python B_train_model.py -backbone unet_se50 -img_size 1024 -tbs 4 -vbs 2 -use_chex 1 -save_path seg_unet_se50_1024
python B_train_model.py -backbone unet_se101 -img_size 1024 -tbs 4 -vbs 2 -use_chex 0 -save_path seg_unet_se101_1024
python A_train_model.py -backbone unet_ef3 -img_size 1024 -tbs 6 -vbs 2 -use_chex 1 -save_path seg_unet_ef3_1024
python A_train_model.py -backbone unet_ef5 -img_size 768 -tbs 4 -vbs 2 -use_chex 0 -save_path seg_unet_ef5_768
python A_train_model.py -backbone deeplab_se50 -img_size 1024 -tbs 4 -vbs 2 -use_chex 1 -save_path seg_deeplab_se50_1024
```
stochastic weight averaging
```bash
python swa_models.py -i seg_unet_se50_1024/ -o ./se50_swa_{}.pth.tar -e0 43 -e1 34 -e2 39 --model_num unet_se50 --batch-size 4
python swa_models.py -i seg_unet_se101_1024/ -o ./se101_swa_{}.pth.tar -e0 43 -e1 34 -e2 39 --model_num unet_se101 --batch-size 4
python swa_models.py -i seg_unet_ef3_1024/ -o ./ef3_swa_{}.pth.tar -e0 43 -e1 34 -e2 39 --model_num unet_ef3 --batch-size 6
python swa_models.py -i seg_unet_ef5_768/ -o ./ef5_swa_{}.pth.tar -e0 43 -e1 34 -e2 39 --model_num unet_ef5 --batch-size 4
python swa_models.py -i seg_deeplab_se50_1024/ -o ./deep_se50_swa_{}.pth.tar -e0 43 -e1 34 -e2 39 --model_num deeplab_se50 --batch-size 4
```
Inference:
```bash
python predict_768.py 
python predict_1024.py
```
Ensemble:
```bash
python ensemble_5_model.py 
```
classification model training:
```bash
cd src_unet_cls
python train_model.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 16 -vbs 8 -save_path diy_model_se_resnext50_32x4d_768_normal
python train_model.py -backbone diy_model_se_resnext50_32x4d -img_size 1024 -tbs 8 -vbs 4 -save_path diy_model_se_resnext50_32x4d_1024_normal
python train_model.py -backbone EfficientNet_3_unet -img_size 1024 -tbs 16 -vbs 8 -save_path EfficientNet_3_unet_1024_normal
python train_model.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 16 -vbs 8 -save_path diy_model_se_resnext50_32x4d_768_add_chexpert
```
stochastic weight averaging
```bash
python swa.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 32 -vbs 8 -cp diy_model_se_resnext50_32x4d_768_normal
python swa.py -backbone diy_model_se_resnext50_32x4d -img_size 1024 -tbs 32 -vbs 8 -cp diy_model_se_resnext50_32x4d_1024_normal
python swa.py -backbone EfficientNet_3_unet -img_size 1024 -tbs 32 -vbs 8 -cp EfficientNet_3_unet_768_normal
python swa.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 32 -vbs 8 -cp diy_model_se_resnext50_32x4d_768_add_chexpert
```
Inference:
```bash
python predict.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 4 -vbs 4 -spth diy_model_se_resnext50_32x4d_768_normal
python predict.py -backbone diy_model_se_resnext50_32x4d -img_size 1024 -tbs 4 -vbs 4 -spth diy_model_se_resnext50_32x4d_1024_normal
python predict.py -backbone EfficientNet_3_unet -img_size 1024 -tbs 4 -vbs 4 -spth EfficientNet_3_unet_1024_normal
python predict.py -backbone diy_model_se_resnext50_32x4d -img_size 768 -tbs 4 -vbs 4 -spth diy_model_se_resnext50_32x4d_768_add_chexpert
```

Submit:
```bash
# stacking
python stacking.py
```

### Leaderboard:
- stage1: 0.8883
- stage2: 0.8665
