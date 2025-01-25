import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError

np.random.seed(678)
tf.random.set_seed(5678)

class ConLayerLeft(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerLeft, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, in_channels, out_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1):
        layer = tf.nn.conv2d(inputs, self.w, strides=[1, stride, stride, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

class ConLayerRight(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerRight, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, out_channels, in_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]
        
        layer = tf.nn.conv2d_transpose(inputs, self.w, output_shape=output_shape,
                                     strides=[1, 1, 1, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

def normalize_mri(image):
    return (image - np.mean(image)) / (np.std(image) + 1e-8)

def augment_mri(image, mask):
    # Random rotation
    angle = np.random.uniform(-20, 20)
    image = tf.keras.preprocessing.image.random_rotation(image, angle)
    mask = tf.keras.preprocessing.image.random_rotation(mask, angle)
    
    # Random brightness
    image = tf.image.random_brightness(image, 0.2)
    
    return image, mask

# Data loading
data_location = "./BrainMRI/training/images/"
train_data = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".nii" in filename.lower() or ".dcm" in filename.lower():
            train_data.append(os.path.join(dirName, filename))

data_location = "./BrainMRI/training/masks/"
train_data_gt = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".nii" in filename.lower() or ".dcm" in filename.lower():
            train_data_gt.append(os.path.join(dirName, filename))

# Image preprocessing
train_images = np.zeros(shape=(128, 256, 256, 1))
train_labels = np.zeros(shape=(128, 256, 256, 1))

for file_index in range(len(train_data)):
    mri_slice = normalize_mri(resize(mpl_imread(train_data[file_index]), (256, 256)))
    train_images[file_index, :, :] = np.expand_dims(mri_slice, axis=2)
    
    mask_slice = resize(mpl_imread(train_data_gt[file_index]), (256, 256))
    train_labels[file_index, :, :] = np.expand_dims(mask_slice, axis=2)

# Model parameters
num_epochs = 100
init_lr = 0.0001
batch_size = 2

class BrainMRICNN(tf.keras.Model):
    def __init__(self, layers):
        super(BrainMRICNN, self).__init__()
        self.layer_list = layers
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

# Layer configurations
layer_configs = [
    (32, 3), (32, 3), (64, 3),
    (64, 3), (128, 3), (128, 3),
    (64, 3), (64, 3), (32, 3),
    (32, 3), (16, 3), (8, 3),
    (4, 3), (1, 3)
]

# Initialize layers
current_channels = 1
layers = []
for i, (out_channels, kernel_size) in enumerate(layer_configs):
    if i == len(layer_configs) - 1:
        layers.append(ConLayerRight(kernel_size, current_channels, out_channels))
    else:
        layers.append(ConLayerLeft(kernel_size, current_channels, out_channels))
    current_channels = out_channels

# Create and compile model
model = BrainMRICNN(layers)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
             loss='mse',
             metrics=['mae'])

# Training loop
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size]
        
        with tf.GradientTape() as tape:
            predictions = model(current_batch)
            loss_value = tf.reduce_mean(tf.square(predictions - current_label))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f' Iter: {iter}, Cost: {loss_value.numpy():.8f}', end='\r')
    
    print('\n-----------------------')
    train_images, train_labels = shuffle(train_images, train_labels)
    
    # Visualization every 2 epochs
    if iter % 2 == 0:
        test_batch = train_images[:2]
        predictions = model(test_batch)
        
        for idx in range(2):
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.squeeze(test_batch[idx]), cmap='gray')
            plt.title('MRI Slice')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(train_labels[idx]), cmap='jet')
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(predictions[idx]), cmap='jet')
            plt.title('Prediction')
            plt.axis('off')
            
            plt.savefig(f'mri_results/iteration_{iter}_sample_{idx}.png')
            plt.close()

# Save the trained model
model.save('brain_mri_segmentation_model')
