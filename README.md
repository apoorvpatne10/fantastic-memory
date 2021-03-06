# fantastic-memory

Check out my [kernel](https://www.kaggle.com/apoorvwatsky/a-minimalistic-model-to-get-started-with) for MIRACL-VC1 on kaggle. It'd be a good place to start. I've worked on a simple model with a few layers to get started with.

Lip Reading using STCNNs and Bi-GRUs. Currently under development.

Project is based on [LipNet](https://arxiv.org/pdf/1611.01599.pdf).

Dataset used : [GRID CORPUS](http://spandh.dcs.shef.ac.uk/gridcorpus/).

![demo](https://i.imgur.com/BHG2yjp.gif)

## Achievements

* Out of 750+ applicants across India, this project made it to the final [round](http://csiawards.inapp.in/wp-content/uploads/2019/04/CSI-InApp-Awards-_2019-Final-Round-ShortList.pdf) (top 16) of CSI National Project Competition 2019. 

* IIC (Institution's Innovation Council) [Proof of Concept](http://abesit.in/wp-content/uploads/2019/05/IIC-PoC-Idea-Submission-Notice-8.pdf) : Project shortlisted for Mentorship Session/Boot Camp at Regional Level (Pune) on 26th July 2019.

## Install the dependencies

Create a new virtual environment and install the dependencies:

`pip install -r requirements.txt`

## Lip Extraction and phrase prediction

Make sure that the video's framerate is 25fps. Pass the video as an argument to `predict` module.

```
(lipnet) apoorv@apoorv:~/Work/fantastic-memory$ python predict.py -h
Using TensorFlow backend.
usage: predict.py [-h] path

Add path to video

positional arguments:
  path        Mention the path to the video

optional arguments:
  -h, --help  show this help message and exit

```

Something like:
```
Example : python predict.py /home/Desktop/abc.mpg|xyz.mp4 
```

This will be followed by generation of all frames of the video passed inside `demo_data/` along with its numpy sequence in ```frame_numpy_sequence``` directory and the decoded text based on the speaker's lip movement.

After this, execute `final_output.py` for final output.

## Bibtex
    @article{assael2016lipnet,
	  title={LipNet: End-to-End Sentence-level Lipreading},
	  author={Assael, Yannis M and Shillingford, Brendan and Whiteson, Shimon and de Freitas, Nando},
	  journal={GPU Technology Conference},
	  year={2017}
	}

## LipNet Architecture

![lipnet](https://i.imgur.com/R0FfyLY.png)
*[LipNet architecture](https://arxiv.org/pdf/1611.01599.pdf). A sequence of T frames is used as input, and is processed by 3 layers
of STCNN, each followed by a spatial max-pooling layer. The features extracted are processed by
2 Bi-GRUs; each time-step of the GRU output is processed by a linear layer and a softmax. This
end-to-end model is trained with CTC.*


## LipNet Model
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
the_input (InputLayer)           (None, 57, 100, 50, 3 0                                            
____________________________________________________________________________________________________
zero1 (ZeroPadding3D)            (None, 59, 104, 54, 3 0                                            
____________________________________________________________________________________________________
conv1 (Conv3D)                   (None, 57, 50, 25, 32 7232                                         
____________________________________________________________________________________________________
batc1 (BatchNormalization)       (None, 57, 50, 25, 32 128                                          
____________________________________________________________________________________________________
actv1 (Activation)               (None, 57, 50, 25, 32 0                                            
____________________________________________________________________________________________________
spatial_dropout3d_1 (SpatialDrop (None, 57, 50, 25, 32 0                                            
____________________________________________________________________________________________________
max1 (MaxPooling3D)              (None, 57, 25, 12, 32 0                                            
____________________________________________________________________________________________________
zero2 (ZeroPadding3D)            (None, 59, 29, 16, 32 0                                            
____________________________________________________________________________________________________
conv2 (Conv3D)                   (None, 57, 25, 12, 64 153664                                       
____________________________________________________________________________________________________
batc2 (BatchNormalization)       (None, 57, 25, 12, 64 256                                          
____________________________________________________________________________________________________
actv2 (Activation)               (None, 57, 25, 12, 64 0                                            
____________________________________________________________________________________________________
spatial_dropout3d_2 (SpatialDrop (None, 57, 25, 12, 64 0                                            
____________________________________________________________________________________________________
max2 (MaxPooling3D)              (None, 57, 12, 6, 64) 0                                            
____________________________________________________________________________________________________
zero3 (ZeroPadding3D)            (None, 59, 14, 8, 64) 0                                            
____________________________________________________________________________________________________
conv3 (Conv3D)                   (None, 57, 12, 6, 96) 165984                                       
____________________________________________________________________________________________________
batc3 (BatchNormalization)       (None, 57, 12, 6, 96) 384                                          
____________________________________________________________________________________________________
actv3 (Activation)               (None, 57, 12, 6, 96) 0                                            
____________________________________________________________________________________________________
spatial_dropout3d_3 (SpatialDrop (None, 57, 12, 6, 96) 0                                            
____________________________________________________________________________________________________
max3 (MaxPooling3D)              (None, 57, 6, 3, 96)  0                                            
____________________________________________________________________________________________________
time_distributed_1 (TimeDistribu (None, 57, 1728)      0                                            
____________________________________________________________________________________________________
bidirectional_1 (Bidirectional)  (None, 57, 512)       3048960                                      
____________________________________________________________________________________________________
bidirectional_2 (Bidirectional)  (None, 57, 512)       1181184                                      
____________________________________________________________________________________________________
dense1 (Dense)                   (None, 57, 28)        14364                                        
____________________________________________________________________________________________________
softmax (Activation)             (None, 57, 28)        0                                            
____________________________________________________________________________________________________
the_labels (InputLayer)          (None, 32)            0                                            
____________________________________________________________________________________________________
input_length (InputLayer)        (None, 1)             0                                            
____________________________________________________________________________________________________
label_length (InputLayer)        (None, 1)             0                                            
____________________________________________________________________________________________________
ctc (Lambda)                     (None, 1)             0                                            
====================================================================================================
Total params: 4,572,156.0
Trainable params: 4,571,772.0
Non-trainable params: 384.0
____________________________________________________________________________________________________
```

## C3D Model we'll be using soon:

```
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1 (Conv3D)               (None, 22, 90, 90, 64)    1792      
_________________________________________________________________
pool1 (MaxPooling3D)         (None, 22, 45, 45, 64)    0         
_________________________________________________________________
conv2 (Conv3D)               (None, 22, 45, 45, 128)   221312    
_________________________________________________________________
pool2 (MaxPooling3D)         (None, 11, 22, 22, 128)   0         
_________________________________________________________________
conv3a (Conv3D)              (None, 11, 22, 22, 256)   884992    
_________________________________________________________________
conv3b (Conv3D)              (None, 11, 22, 22, 256)   1769728   
_________________________________________________________________
pool3 (MaxPooling3D)         (None, 5, 11, 11, 256)    0         
_________________________________________________________________
conv4a (Conv3D)              (None, 5, 11, 11, 512)    3539456   
_________________________________________________________________
conv4b (Conv3D)              (None, 5, 11, 11, 512)    7078400   
_________________________________________________________________
pool4 (MaxPooling3D)         (None, 2, 5, 5, 512)      0         
_________________________________________________________________
conv5a (Conv3D)              (None, 2, 5, 5, 512)      7078400   
_________________________________________________________________
conv5b (Conv3D)              (None, 2, 5, 5, 512)      7078400   
_________________________________________________________________
zero_padding3d_1 (ZeroPaddin (None, 2, 7, 7, 512)      0         
_________________________________________________________________
pool5 (MaxPooling3D)         (None, 1, 3, 3, 512)      0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0         
_________________________________________________________________
fc6 (Dense)                  (None, 4096)              18878464  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
fc7 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dropout_2 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                40970     
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 63,353,226
Trainable params: 63,353,226
Non-trainable params: 0
_________________________________________________________________
```
