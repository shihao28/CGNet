# Cross-Granularity (CG) Network for Vehicle Make and Model Recognition
Cross-Granularity Network is made up of a backbone Convolutional Neural Network (CNN) and CG module. The CG module is a multi-scale feature generation module that reinforces the perception ability of the network by harnessing both high-level semantic information and low-level fine-grained details. It augments lateral connections from the convolutional blocks of CNN to gather the feature maps from all pyramid levels and subsequently coalesces them in a non-linear manner. Our proposal reports state-of-the-art performances on several publicly available datasets.

# Results
| Dataset       | Accuracy |
|---------------|----------|
| CompCarsWeb   | 98.3%    |
| Stanford Cars | 95.4%    |
| Car-FG3K      | 86.4%    |
| CompCarsSV    | 99.1%    |

# Installation 
1. pip install -r requirements.txt

# Train
1. Alter training configuration in config/train_config.yml
2. python train.py

# Evaluation
1. Alter evaluation configuration in config/eval_config.yml
2. python eval.py

# Acknowledgement
Thank you to the following authors for sharing their datasets
- Yang et al.: [CompCarsWeb and CompCarsSV](https://ieeexplore.ieee.org/document/7299023)
- Krause et al.: [Stanford Cars](https://ieeexplore.ieee.org/document/6755945)
- Wu et al.: [Car-FG3K](https://dl.acm.org/doi/10.1007/978-3-030-98355-0_20)
