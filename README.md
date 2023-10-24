# Low Field Enhancement (TODO: update the project name to the paper's name)

[ArXiv Paper][Demo][Published Paper]

This project utilizes the SCUNet (Swin-Conv-UNet) denoising network to enhance low field images, making them comparable to high field images. The project is for research purposes only and cannot be used for commercial purposes



The data pair synthesis and image enhancement pipeline 
----------

<img src="figs/Fig1.png" width="900px"/>


*New data synthesis pipeline for real image denoising*: For a high quality image, a randomly shuffled
degradation sequence is performed to produce a noisy image. Meanwhile, the resizing and reverse-forward tone mapping are performed
to produce a corresponding clean image. A paired noisy/clean training patches are then cropped for training deep blind denoising model.


*Swin-Conv-UNet(SCUNet) denoising network*  The SCUNet model utilized the swin-conv (SC) block as the primary component of a UNet backbone, combining the advantages of residual convolution and transformer mechanisms to enhance both local and non-local modeling. The first step is to pretrain the SCUNet model using the pretraining dataset including MRI images from 966 subjects, which is showed at the upper half of the figure (inside the red dotted line). The pretrained SCUNet model was finetuned using small-scale paired LF-MRI and HF-MRI images at the second step (inside the purple dotted line). The mobile LF-MRI images were inputted into the pretrained SCUNet model to generate SynthMRI images. The blue border highlights the forward propagation stage of the SCUNet model. The loss function of L1 and structural similarity index (SSIM) were applied to train and finetune the SCUNet model (with green border).


Running Environment and Reproduction
----------
Python 3.8 and PyTorch 1.7

Preprocessing scripts: src/data_util.py
Training scripts: scr
Inference script: 


Side note
----------
The project is for research purposes only and cannot be used for commercial purposes.
Please cite 
TODO: paper name 