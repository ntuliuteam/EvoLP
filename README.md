# EvoLP
This repository contains a Python implementation for "EvoLP: Self-Evolving Latency Predictor for Model
Compression in Real-Time Edge Systems"

## Dependencies

An [*environment.yml*](environment.yml) has been uploaded for creating a conda environment.

Create a shared folder (Samba file server) and mount (CIFS Utilities) it on the used edge device and the high-performance for-training server. Then download the code to the shared folder:
```shell
~ $ git clone git@github.com:ntuliuteam/EvoLP.git
```

If you do not have permission to mount, please use SCP.

## Predictor Generation - Model analyses and sampling (executed on the edge device)
```shell
~ $ cd ./Predictor_Gen/
~ $ mkdir Samples && mkdir Samples/vgg16
~ $ python search_space.py --arch vgg16 --input_size 224 --sample 1000 --save ./Samples/vgg16/vgg16.csv
~ $ python latency_measure_on_device.py --load ./Samples/vgg16/vgg16.csv --save ./Samples/vgg16/vgg16_result_<devicename>.csv --cuda 1
~ $ python result_process.py --load ./Samples/vgg16/vgg16_result_<devicename>.csv
```

## Predictor Generation - MLP training (executed on the server)
```shell
~ $ cd ./Predictor/
~ $ python matlab_script_generate.py --sample_folder ../Predictor_Gen/Samples/vgg16/ --save_folder ./vgg16_<devicename>/
```
Run *matlab_train.m* in Matlab.

## Model Compression with the Latency Predictor (executed on the server)
```shell
~ $ cd Predictor_EDLAB
```
In the *RlLatency class* in *predictor.py*, modify *self.device* to include your &lt;devicename&gt; and *self.py_path* to point to the path of python in your &lt;devicename&gt. Then train the model with predictor:
```shell
~ $ cd Model_Compress
~ $ python train_imagenet_example.py -a vgg16 --save ./logs-vgg16-latency40-tx2 --lr 0.01 <PATH/TO/imagenet_100/> --our 1 -sr --ssr 0.001 --epochs 90 --batch-size 128 --latency 40 --device tx2 --zerobn 30
```

## Latency & Accuracy Verify 
### Remove Zerorized Weights (executed on the server)
The Imagenet can be downloaded from [IMAGENET_2012](https://image-net.org/), and the classes we used are listed in the
[*imagenet100.txt*](imagenet100.txt).

The pre-trained weights used in the paper can be downloaded from [*here*](https://drive.google.com/file/d/1ImrFcsiEnAUSyUOPMooxSGLSAWidXT6y/view?usp=sharing) (Edge GPU1 is Nvidia Jetson TX2; Edge GPU2 is Nvidia Jetson Nano). Then unzip this file into the Model_Compress folder.
```shell
~ $ cd ./Model_Compress/
~ $ gdown https://drive.google.com/uc?id=1ImrFcsiEnAUSyUOPMooxSGLSAWidXT6y 
~ $ unzip logs.zip -d ./
```

Remove the zerorized weights and test the accuracy of pre-trained models.
```shell
~ $ cd ./Remove_Zero/
~ $ python remove.py --arch vgg16 --resume ../Model_Compress/logs-vgg16-latency40-tx2/0/model_best.pth.tar --save ./vgg16_tx2_40.pth.tar <PATH/TO/imagenet_100/>
~ $ python remove.py --arch inception_v3 --resume ../Model_Compress/logs-inceptionv3-latency50-tx2/0/model_best.pth.tar --save ./inceptionv3_tx2_50.pth.tar <PATH/TO/imagenet_100/>
~ $ python remove.py --arch resnet50_new --resume ../Model_Compress/logs-resnet50-latency35-tx2/0/model_best.pth.tar --save ./resnet50_tx2_35.pth.tar <PATH/TO/imagenet_100/>
~ $ python remove.py --arch mobilenet_v1 --resume ../Model_Compress/logs-mobilenetv1-latency20-nano/0/model_best.pth.tar --save ./mobilenetv1_nano_20.pth.tar <PATH/TO/imagenet_100/>
```

### Latency Evaluation (executed on the edge device)
```shell
~ $ cd ./Remove_Zero/
~ $ python latency_evaluate.py --arch vgg16 --input_size 224 --resume ./vgg16_tx2_40.pth.tar
```