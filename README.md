# Fine-tuning-CLIP

!(https://github.com/yangengineering/Fine-tuning-clip/blob/main/overview.png)

## Fine-tune CLIP to your own dataset
This code focuses on fine-tuning CLIP as a way to perform better on abnormal datasets.

CLIP was trained on ImageNet, so it is sensitive to common categories, such as cats, dogs, birds, and so on. But we use CLIP for classification on uncommon datasets, such as the industrial fault dataset, and CLIP's zero-shot performance is very poor.

So we provide a fine-tuned code to adapt to our own dataset and achieve better performance.

### Requirements

We have trained and tested our models on `Ubuntu 18.0`, `CUDA 11.0`, `Python 3.7`

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install -r requirements.txt
```
### Datasets

You can add images of your own datasets in the datasets folder. And add the paths and labels of the images in the data folder.

### Train and test

```bash
cd rkvan
sh scripts/run1.sh
```
The results are on the `rkvan/ouputs`.

## Acknowledgement

Our code base is build on top of [CLIP](https://github.com/openai/CLIP). 
