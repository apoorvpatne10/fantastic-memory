# fantastic-memory

Lip Reading using STCNNs and Bi-GRUs. Currently under development.

Project is based on [LipNet](https://github.com/apoorvpatne10/fantastic-memory/blob/master/lipNet.pdf).

Dataset used : [MIRACL-VC1](https://www.kaggle.com/apoorvwatsky/miraclvc1).

## Model Summary (Temporary)

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv3d_4 (Conv3D)            (None, 9, 43, 43, 8)      1736      
_________________________________________________________________
max_pooling3d_4 (MaxPooling3 (None, 3, 14, 14, 8)      0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 3, 14, 14, 8)      0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 4704)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                301120    
_________________________________________________________________
dropout_6 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 32)                2080      
_________________________________________________________________
dropout_7 (Dropout)          (None, 32)                0         
_________________________________________________________________
dense_6 (Dense)              (None, 10)                330       
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0         
=================================================================
Total params: 305,266
Trainable params: 305,266
Non-trainable params: 0
```
_____________
## Initial training results

model.fit(X_train, y_train, validation_data=(X, y), batch_size=batch_si,
      epochs=num_epochs, shuffle=True)
Train on 1300 samples, validate on 100 samples
Epoch 1/8
1300/1300 [==============================] - 8s 6ms/step - loss: 4.5109 - mean_squared_error: 0.1800 - acc: 0.2992 - val_loss: 4.5063 - val_mean_squared_error: 0.1800 - val_acc: 0.2300
...
...
Epoch 8/8
1300/1300 [==============================] - 5s 4ms/step - loss: 3.6097 - mean_squared_error: 0.1402 - acc: 0.0985 - val_loss: 3.7023 - val_mean_squared_error: 0.1204 - val_acc: 0.3140
Out[30]: <keras.callbacks.History at 0x7f2ffb6ba390>
____________________________________________________
