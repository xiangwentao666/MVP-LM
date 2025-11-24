# MVP-LM
Official Implementation of paper: 

[\[ICCV2025\]Advancing Visual Large Language Model for Multi-granular Versatile Perception](https://arxiv.org/abs/2507.16213)

# Checkpoint Download
Checkpoints can be downloaded at [MVP-LM-checkpoint](https://pan.baidu.com/s/1nm84oBWeJUYZj-y03i4uYw?pwd=6666)

# Installation
```
pip install -r requirements.txt
pip install -e /PATH-to-mvp-lm/mvp-lm/psalm/model/mask_decoder/Openseed_Simplify/modeling/pixel_decoder/ops
pip install torch==1.13.1+cu116
pip install torchaudio==0.13.1+cu116
pip install torchvision==0.14.1+cu116
```

# Code-related
The source code of the shell script required to run the code contains many personalized logic elements, aiming to improve efficiency, establish a standard operating procedure (SOP) for each execution, and automate the model training and inference process. You may extract the most essential command for launching the Python process and incorporate your own automation logic as needed.

# Aknowledgements
This project is based on the following great works: [Mask2Former](https://github.com/facebookresearch/Mask2Former), [PSALM](https://github.com/zamling/PSALM) and [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD).

