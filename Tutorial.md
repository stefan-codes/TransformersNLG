# NLG tutorial using Transformers with Tensorflow-Gpu

## Instal GPU requirements
https://www.tensorflow.org/install/gpu

### GPU Drivers
Just make sure the drivers are up to date.

### CUDA Toolkit
At as of the moment, I installed 10.1 update 2 (cuda_10.1.243_win10_network.exe). 

### CuDNN
Extract the files into the cuda dir and created a tool\cuda folder in C:\\.

### Set up the enviornment path
Add the enviornment path as instructed in the documentation.

## Create a virtual environment
It was recommendet to use virtualenv instead of the build in venv. Not sure why at this point. The following commands are for windows.

#### Create an env
```shell
virtualenv "name"
```

#### Activate the env
```shell
.\env\Scripts\activate
```

#### Deactivate the env (FYI)
```shell
deactivate
```

## Create requirements.txt
I installed pipreqs as suggested on stackoverlow.
https://github.com/bndr/pipreqs

#### Install
```shell
pip install pipreqs
```

#### Create
```shell
pipreqs /home/project/location
```

## Install modules
