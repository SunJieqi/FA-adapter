# **Few-shot Classification with Fork Attention Adapter**

PyTorch implementation of Few-shot Classification with Fork Attention Adapter.

## Datasets

You must first specify the value of `data_path` in `config.yml`. This should be the absolute path of the folder where you plan to store all the data.

We follow [FRN](https://github.com/Tsingularity/FRN) setting to use the same data settings for training.

## Training scripts

CUB cropped/CUB

```
python train.py \
    --opt sgd \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 150 \
    --decay_epoch 70 120 \
    --val_epoch 20 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 15 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --resnet \
    --gpu 0 \
    --pre
```

mini-ImageNet

```
python train.py \
    --opt sgd \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 30 \
    --decay_epoch 15 25 \
    --stage 2 \
    --val_epoch 5 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 15 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --resnet \
    --gpu 0 \
    --pre
```

Or Running the shell script `train.sh` will train and evaluate the model with hyperparameters matching our paper. Explanations for these hyperparameters can be found in `trainers/trainer.py`.

For the first-phase training weights, you can download from [FRN](https://github.com/Tsingularity/FRN) directly, and modified the *pretrained_model_path*  in train.py

After training concludes, test accuracy and 95% confidence interval are logged in the std output and `*.log` file. To re-evaluate a trained model, run `test.py`, setting the internal `model_path` variable to the saved model `*.pth` you want to evaluate.

## Contact

We have tried our best to verify the correctness of our released data, code. However, there are a large number of experiment settings, all of which have been extracted and reorganized from our original codebase. There may be some undetected bugs or errors in the current release. If you encounter any issues or have questions about using this code, please feel free to contact us via sunjieqi1017@163.com.

## Acknowledgment

Our project references the codes in the following repos.

- [FRN](https://github.com/Tsingularity/FRN)

If you use the code in this repo for your work, please cite the following bib entries:

```
@article{SUN2024110805,
title = {Few-shot classification with Fork Attention Adapter},
journal = {Pattern Recognition},
volume = {156},
pages = {110805},
year = {2024},
author = {Jieqi Sun and Jian Li}
}
```

