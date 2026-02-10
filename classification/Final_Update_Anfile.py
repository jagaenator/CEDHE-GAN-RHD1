Importing Necessory packages:
import numpy as np
import os
import sys
import datetime
import matplotlib.pyplot as plt
import cv2
import datetime
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Utils
import h5py
from google.colab import drive
drive.mount('/content/drive')
os.getcwd()
cd '/content/drive/MyDrive/170450 Segmentation Heart disease image'
data = h5py.File("Dataset_2/image_dataset.hdf5", "r")
Images = data["train 2ch frames"][:,:,:,:]
Masks = data["train 2ch masks"][:,:,:,:]
print(Images.shape, Masks.shape)
Images[0].shape
plt.imshow(Images[0],cmap ='gray')
# Visualization of masks:
#Input_images = os.listdir('write_dir/Mitotic_fig/masks')
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
# Loop through the first 9 images:
for i, image_file in enumerate(Images[:9]):
    # Display the image
    axes[i].imshow(image_file,cmap='gray')
    axes[i].axis('off')
    #axes[i].set_title("size-{}".format(image.shape))
  # Visualization of masks:
#Input_images = os.listdir('write_dir/Mitotic_fig/masks')
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
# Loop through the first 9 images:
for i, image_file in enumerate(Masks[:9]):
    # Display the image
    axes[i].imshow(image_file)
    axes[i].axis('off')
    #axes[i].set_title("size-{}".format(image.shape))
  type(Images)
# Define Gaussian blur kernel size and sigma
kernel_size = (5, 5)  # Kernel size
sigma = 1.0  # Standard deviation for the Gaussian kernel

# Iterate through each image and apply Gaussian blur
preprocessed_images = np.empty_like(Images)  # Create an empty array to store blurred images

for i in range(Images.shape[0]):
    # Extract the 2D image from the 3D shape (384, 384, 1)
    image = Images[i, :, :, 0]

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    # Store the blurred image back in the array
    preprocessed_images[i, :, :, 0] = blurred_image

print("Gaussian blur applied to all images successfully.")
preprocessed_images.shape
# Visualization of masks:
#Input_images = os.listdir('write_dir/Mitotic_fig/masks')
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()
# Loop through the first 9 images:
for i, image_file in enumerate(preprocessed_images[:9]):
    # Display the image
    axes[i].imshow(image_file,cmap='gray')
    axes[i].axis('off')
    #axes[i].set_title("size-{}".format(image.shape))
  import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow.keras.backend as K
from tensorflow import argmax
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
train_images, test_images, train_masks, test_masks = train_test_split(preprocessed_images, Masks)
def generalized_dice_loss(y_true, y_pred, smooth=1e-7, num_classes=4):

    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return 1.0-K.mean((2. * intersect / (denom + smooth)))
  def multiclass_dice(y_true, y_pred, smooth=1e-7, num_classes=4):
    '''
    Multiclass Dice score. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:])
    y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))
    def dice_lv(y_true, y_pred, smooth=1e-7, num_classes=4):
    '''
    Multiclass Dice score. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,1:2])
    y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,1:2])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))
    def dice_la(y_true, y_pred, smooth=1e-7, num_classes=4):
    '''
    Multiclass Dice score. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,3:4])
    y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,3:4])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))
def dice_myo(y_true, y_pred, smooth=1e-7, num_classes=4):
    '''
    Multiclass Dice score. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=4)[...,2:3])
    y_pred_f = K.flatten(K.one_hot(argmax(y_pred, axis=3), num_classes=4)[...,2:3])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))
  drop = 0.25

def ResBlock(input_tensor, filters):

    conv_1 = Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')
    conv_1a = conv_1(input_tensor) # Shared weights conv layer
    batch_1 = BatchNormalization()(conv_1a)
    relu_1  = Activation("relu")(batch_1)
    drop_1  = Dropout(drop)(relu_1)
    conv_1b = conv_1(drop_1) # Shared weights conv layer
    batch_1 = BatchNormalization()(conv_1b)
    return batch_1

def Dual_UNet(input_size = (256, 256, 1), num_classes=2, filters=30):

    # X's denote standard flow
    # XNUM denote ResBlock outputs

    # "First" UNet

    # Input branch
    inputs = Input(input_size)
    X = Conv2D(filters=filters, kernel_size=3, activation="relu", padding = 'same', kernel_initializer = 'he_normal')(inputs)

    # Down branch
    X1 = ResBlock(input_tensor=X, filters=filters) # ResBlock located in the first layer of the paper scheme
    X = Conv2D(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X1)
    X = Activation("relu")(X) # This ReLU is not shown in the paper scheme

    X2 = ResBlock(input_tensor=X, filters=filters*2)
    X = Conv2D(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X2)
    X = Activation("relu")(X)

    X3 = ResBlock(input_tensor=X, filters=filters*4)
    X = Conv2D(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X3)
    X = Activation("relu")(X)

    X4 = ResBlock(input_tensor=X, filters=filters*8)
    X = Conv2D(filters=filters*16, kernel_size=3, strides=2, kernel_initializer='he_normal')(X4)
    X = Activation("relu")(X)

    # Bottom block
    X = ResBlock(input_tensor=X, filters=filters*16)

    # Up branch
    X = Conv2DTranspose(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X4])
    # X = Activation("relu")(X) # This ReLU is commented in the paper code
    X5 = ResBlock(input_tensor=X, filters=filters*8)

    X = Conv2DTranspose(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X5)
    X = Add()([X, X3])
    # X = Activation("relu")(X)
    X6 = ResBlock(input_tensor=X, filters=filters*4)

    X = Conv2DTranspose(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X6)
    X = Add()([X, X2])
    # X = Activation("relu")(X)
    X7 = ResBlock(input_tensor=X, filters=filters*2)

    X = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, output_padding=1, kernel_initializer='he_normal')(X7)
    X = Add()([X, X1])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters)

    # Top block (bottle-neck)
    X8 = ResBlock(input_tensor=X, filters=filters)
    X = ResBlock(input_tensor=X, filters=filters)
    X = Add()([X, X8])

    # "Second" UNet

    # Down branch
    X9 = ResBlock(input_tensor=X, filters=filters)
    X = Conv2D(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Activation("relu")(X)
    X = Add()([X7, X])

    X10 = ResBlock(input_tensor=X, filters=filters*2)
    X = Conv2D(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Activation("relu")(X)
    X = Add()([X6, X])

    X11 = ResBlock(input_tensor=X, filters=filters*4)
    X = Conv2D(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Activation("relu")(X)
    X = Add()([X5, X])

    X12 = ResBlock(input_tensor=X, filters=filters*8)
    X = Conv2D(filters=filters*16, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Activation("relu")(X)

    # Bottom block
    X = ResBlock(input_tensor=X, filters=filters*16)

    # Up branch
    X = Conv2DTranspose(filters=filters*8, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X12])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*8)

    X = Conv2DTranspose(filters=filters*4, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X11])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*4)

    X = Conv2DTranspose(filters=filters*2, kernel_size=3, strides=2, kernel_initializer='he_normal')(X)
    X = Add()([X, X10])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters*2)

    X = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, kernel_initializer='he_normal', output_padding=1)(X)
    X = Add()([X, X9])
    # X = Activation("relu")(X)
    X = ResBlock(input_tensor=X, filters=filters)

    # Final block
    X = Conv2D(filters=num_classes, kernel_size=1, kernel_initializer='he_normal')(X)
    # X = Activation("relu")(X)
    X = Activation("softmax")(X)
    #X = Conv2D(1, 1)(X)

    model = Model(inputs, X)

    return model
  model_unet = Dual_UNet(input_size=(384, 384, 1), num_classes=4, filters=20
                         earlystop = EarlyStopping(monitor='val_multiclass_dice', min_delta=0, patience=5,
                          verbose=1, mode="max", restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor='val_multiclass_dice', factor=0.2, patience=2,
                              verbose=1, mode="max", min_lr=1e-5)
model_unet.summary()
model_unet.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=[multiclass_dice, "accuracy"])
history = model_unet.fit(x=train_images,
                    y=train_masks,
                    validation_data=[test_images, test_masks],
                    batch_size=5,
                    epochs=50,
                    callbacks=[earlystop, reduce_lr])
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
plot_graphs(history, "multiclass_dice")
# Convert to DataFrame
df = pd.DataFrame(history.history)

# Save to CSV
df.to_csv("training_metrics_50ep_1.csv", index=False)

print("Training metrics saved to 'training_metrics.csv'")
def display_image_grid(test_frames, test_masks, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = 10
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i in range(10):
        img = test_frames[i,:,:,0]
        mask = test_masks[i,:,:,0]

        ax[i, 0].imshow(img, cmap='gray')
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask)
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()


predicted_masks = []
for i in range(10):
    prediction = model_unet.predict(test_images[i:i+1,:,:,:])
    prediction = prediction.reshape([384, 384, 4])
    y = tf.convert_to_tensor(prediction)
    predicted_masks.append(tf.math.argmax(prediction, axis = 2))

display_image_grid(test_images, test_masks, predicted_masks=predicted_masks)

predicted_masks = []
for i in range(len(preprocessed_images)):
    prediction = model_unet.predict(preprocessed_images[i:i+1,:,:,:])
    prediction = prediction.reshape([384, 384, 4])
    y = tf.convert_to_tensor(prediction)
    predicted_masks.append(tf.math.argmax(prediction, axis = 2))
    plt.imsave(f'Segmented_Output/{i}.png',predicted_masks[i])
    print(f'Segmented_Output{i}.png is saved')
  import cv2
import numpy as np

def quantize_colors(image, num_bins=8):
    """Quantizes an image into fixed bins per channel (HSV space)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    bins = np.linspace(0, 256, num_bins + 1)  # Define bin edges
    quantized = np.digitize(hsv, bins) - 1  # Assign each pixel to a bin (0 to num_bins-1)
    return quantized

def compute_ccm(image, distance=1, num_bins=8):
    """Computes the Color Co-occurrence Matrix (CCM)"""
    quantized_image = quantize_colors(image, num_bins)
    h, w, _ = quantized_image.shape

    # Initialize CCM with valid bin size
    ccm = np.zeros((num_bins, num_bins), dtype=np.float32)

    # Iterate over image pixels (excluding edges)
    for y in range(h):
        for x in range(w - distance):  # Ensure index does not go out of bounds
            color1 = quantized_image[y, x, 0]  # Get bin index of first pixel (H Channel)
            color2 = quantized_image[y, x + distance, 0]  # Neighboring pixel (H Channel)

            if 0 <= color1 < num_bins and 0 <= color2 < num_bins:  # Valid indices check
                ccm[color1, color2] += 1  # Increment co-occurrence count

    # Normalize CCM
    ccm /= np.sum(ccm) if np.sum(ccm) > 0 else 1  # Avoid division by zero 
    return np.array(ccm.flatten(), dtype=np.float32)
preprocessed_images[0].shape
# Path to the folder containing images
folder_path = "Segmented_Output"

# List all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize an empty list to store images
image_list = []

# Load images and convert them to NumPy arrays
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = cv2.imread(img_path)  # Read image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_list.append(img)

# Convert list of images to a single NumPy array
Segmented_Image = np.array(image_list)

# Print shape of the final NumPy array
print("Loaded Image Array Shape:", Segmented_Image.shape)  # (num_images, height, width, channels)
import os
import pandas as pd

# Prepare storage for features and labels
feature_list = []
label_list = []

for img_file in Segmented_Image:
        ccm_feat = compute_ccm(img_file)
        feature_list.append(ccm_feat)
                       

# Save to CSV
features_df = pd.DataFrame(feature_list)
csv_output_path = 'color_co-ocurrence_features_.csv'
features_df.to_csv(csv_output_path, index=False)
print(f"CCM feature extraction complete.")  
features_df=pd.read_csv('color_co-ocurrence_features_.csv')
import os
import pandas as pd
feature_df =pd.read_csv('color_co-ocurrence_features_with_target_label.csv')
feature_df.shape
feature_df.head(15)
features_df1=feature_df.copy()
features_df1['Heart Disease'].value_counts()
x1 = features_df1.drop('Heart Disease',axis=1)
y1 = features_df1['Heart Disease']
print(x1.shape)
print(y1.shape)
from sklearn.preprocessing import StandardScaler,train_test_split

sc=StandardScaler()
scaled_x1=sc.fit_transform(x1)
from sklearn.model_selection import train_test_split, GridSearchCV
X_train,X_test,y_train,y_test = train_test_split(scaled_x1,y1,test_size=0.20,random_state=42,shuffle=True)  
print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,LSTM

from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy,categorical_crossentropy,CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy,categorical_accuracy,CategoricalAccuracy,sparse_categorical_accuracy,Accuracy
rom tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
callbacks = [
    ModelCheckpoint(filepath='cnn_classification_model2.keras', save_best_only=True, monitor='val_loss', mode='min',verbose =1),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)]
import seaborn as sns
history_plot = history.history

# Plot Training & Validation Loss
sns.set_theme(style='darkgrid')
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history_plot['loss'], label='Train Loss')
plt.plot(history_plot['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
#plt.grid()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history_plot['accuracy'], label='Train Accuracy')
plt.plot(history_plot['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim=(0,1)
plt.title('Training & Validation Accuracy')
plt.legend()
#plt.grid()

#plt.show()
predicted = model2.predict(X_test)

predicted = np.argmax(predicted, axis=1)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

accuracy = accuracy_score(y_test, predicted)
#print('Accuracy : ',accuracy)
precision = precision_score(y_test, predicted, average='macro')
recall = recall_score(y_test, predicted, average='macro')
f1 = f1_score(y_test, predicted, average='macro')

print('Accuracy : ',accuracy)
print('Precision:',precision)
print('Recall:',recall)
print('F1-score:',f1)
print("Classification Report:\n", classification_report(y_test, predicted))
cm= confusion_matrix(y_test,predicted)
cm_percentage = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True) * 100
class_names=['Heart Disease Symptoms','NO Heart Disease symptoms']
import seaborn as sns
plt.figure(figsize = (8,6))
sns.heatmap(cm_percentage,annot =True,fmt=".2f",cmap="Blues",xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel("True Label")
plt.title("Confusion Matrix") 
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Rows = Patients, Columns = Models
data = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 1]
])

patients = [f"P{i+1}" for i in range(data.shape[0])]
models = ['CNN', 'GAN-CEDHE', 'Proposed']

plt.figure(figsize=(6, 4))
sns.heatmap(
    data,
    annot=True,
    cmap="Reds",
    cbar=False,
    xticklabels=models,
    yticklabels=patients
)

plt.title("Binary Heatmap of RHD Detection (0=Normal, 1=RHD)")
plt.xlabel("Models")
plt.ylabel("Patient Samples")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Confusion Matrix
cm = np.array([
    [96, 4],   # Normal
    [6, 94]    # RHD
])

classes = ['Normal', 'RHD']

fig, ax = plt.subplots()
im = ax.imshow(cm)

# Ticks & labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix Heatmap")

# Annotate cell values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j],
                ha="center", va="center", fontsize=12)

fig.colorbar(im)
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with your values)
data = np.array([
    [0.92, 0.88, 0.90],
    [0.85, 0.91, 0.89],
    [0.87, 0.86, 0.93]
])

fig, ax = plt.subplots()
im = ax.imshow(data)

# Axis labels
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(['Model-1', 'Model-2', 'Model-3'])
ax.set_yticklabels(['Sample-1', 'Sample-2', 'Sample-3'])

# Annotate values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.2f}",
                ha="center", va="center")

ax.set_title("Performance Heatmap")
fig.colorbar(im)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Rows: Patients, Columns: Models
data = np.array([
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 0],
    [1, 1, 0, 1]
])

patients = ['P1', 'P2', 'P3', 'P4']
models = ['CNN', 'U-Net', 'Dual U-Net', 'Proposed']

fig, ax = plt.subplots()
im = ax.imshow(data)

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(patients)))
ax.set_xticklabels(models)
ax.set_yticklabels(patients)

ax.set_title("Binary RHD Detection Heatmap")
ax.set_xlabel("Models")
ax.set_ylabel("Patient Samples")

# Annotate 0/1
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, data[i, j],
                ha="center", va="center")

fig.colorbar(im)
plt.tight_layout()
plt.show()
plt.savefig("heatmap.png", dpi=300)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
img_dir  = "dataset/images"
gt_dir   = "dataset/masks_gt"
pred_dir = "dataset/masks_pred"

# Load filenames
files = sorted(os.listdir(img_dir))
num_samples = len(files)

# Create grid
fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))

for i, fname in enumerate(files):

    # Load images
    img  = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(os.path.join(pred_dir, fname), cv2.IMREAD_GRAYSCALE)

    # Resize for safety
    gt = cv2.resize(gt, img.shape[::-1])
    pred = cv2.resize(pred, img.shape[::-1])

    # Error heatmap
    error = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    error = (error - error.min()) / (error.max() + 1e-6)

    # Display
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title("Image")

    axes[i, 1].imshow(gt)
    axes[i, 1].set_title("Ground Truth")

    axes[i, 2].imshow(pred)
    axes[i, 2].set_title("Prediction")

    axes[i, 3].imshow(error)
    axes[i, 3].set_title("Error Heatmap")

    for j in range(4):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("segmentation_heatmap_grid.png", dpi=300)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load ground truth and predicted masks (grayscale / label masks)
gt = cv2.imread("ground_truth.png", cv2.IMREAD_GRAYSCALE)
pred = cv2.imread("predicted_mask.png", cv2.IMREAD_GRAYSCALE)

# Ensure same size
gt = cv2.resize(gt, pred.shape[::-1])

# Normalize masks
gt = gt.astype(np.float32)
pred = pred.astype(np.float32)

# Absolute error map
error_map = np.abs(gt - pred)

# Normalize for visualization
error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-6)

# Plot heatmap
plt.figure(figsize=(5,5))
plt.imshow(error_map)
plt.colorbar(label="Segmentation Error Intensity")
plt.title("Segmentation Error Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()
import cv2
import os
import sys

gt_path   = "ground_truth.png"
pred_path = "predicted_mask.png"

# Check existence
if not os.path.exists(gt_path):
    sys.exit(f"ERROR: Ground truth mask not found at {gt_path}")

if not os.path.exists(pred_path):
    sys.exit(f"ERROR: Predicted mask not found at {pred_path}")

# Read images
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

# Check read success
if gt is None:
    sys.exit("ERROR: Failed to load ground truth image")

if pred is None:
    sys.exit("ERROR: Failed to load predicted mask")

# Resize safely
gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
img_dir  = "C:\\Users\\Jagadesh\\OneDrive\\Desktop\\download"
gt_dir   = "C:\\Users\\jagadesh\\OneDrive\\Desktop\masks_gt"
pred_dir = "C:\\Users\\jagadesh\\OneDrive\\Desktop\masks_pred"

# Load filenames
files = sorted(os.listdir(img_dir))
num_samples = len(files)

# Create grid
fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))

for i, fname in enumerate(files):

    # Load images
    img  = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
    gt   = cv2.imread(os.path.join(gt_dir, fname), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(os.path.join(pred_dir, fname), cv2.IMREAD_GRAYSCALE)

    # Resize for safety
    gt = cv2.resize(gt, img.shape[::-1])
    pred = cv2.resize(pred, img.shape[::-1])

    # Error heatmap
    error = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    error = (error - error.min()) / (error.max() + 1e-6)

    # Display
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title("Image")

    axes[i, 1].imshow(gt)
    axes[i, 1].set_title("Ground Truth")

    axes[i, 2].imshow(pred)
    axes[i, 2].set_title("Prediction")

    axes[i, 3].imshow(error)
    axes[i, 3].set_title("Error Heatmap")

    for j in range(4):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("segmentation_heatmap_grid.png", dpi=300)
plt.show()
