IMPORTANT ---> This is a project study, with the goal to understand and using a real world GAN architecture. The origininal paper is at this link https://arxiv.org/abs/1812.02288 .

1 ---> In the script folder, there are two Python scripts used locally for data preparation, which are then passed to the Google Cloud infrastructure - Colab, where I performed model training and testing.

2 ---> In the model folder, the trained models are stored; each subfolder contains a model trained with a different latent dimension, as this was the focus of the study. Additionally, there is also a model trained as suggested by the original study.

3 ---> The notebook contains the code for training and testing the models.
Note: To test on Colab or locally, simply uncomment/comment the lines in the code that define the paths marked with #COLAB.

4 ---> In the data folder, there are the original (64x64) data and those used in the experiments.
