from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import seaborn as sns 
from tensorflow.keras.callbacks import TensorBoard 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/adam_base_epoch00008") 


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    r'C:/Users/amkan/Downloads/archive',
    target_size=(200,200),
    batch_size=50,
    class_mode='categorical' 
)
import os
safe = 0
cancer = 0

# Get the current user's home directory
home_dir = os.path.expanduser("~")

# Define the relative path from the home directory to the data directory
relative_data_dir = 'Downloads/archive/models'

# Construct the full path to the data directory
data_dir = os.path.join(home_dir, relative_data_dir)

class_names = ['Healthy','Cancerous']
print("Classes found:", class_names)

num_samples_per_class = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in class_names}

print("Number of samples for each class:")
print(num_samples_per_class)

import os
import random
import matplotlib.pyplot as plt
from PIL import Image





images_to_display = {}

# Load and display one image from each class
plt.figure(figsize=(15, 10))
for idx, class_name in enumerate(class_names):
    # Get the random image file from the directory
    image_files = os.listdir(os.path.join(data_dir, class_name))
    
    random_image_index = random.randint(0, len(image_files) - 1)
    print(f"{random_image_index}/{len(image_files)}")

    if len(image_files) > 0:
        # Load the first image from the directory
        image_path = os.path.join(data_dir, class_name, image_files[random_image_index])
        img = Image.open(image_path)
        images_to_display[class_name] = img

        # Plot the image
        plt.subplot(1, len(class_names), idx + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

plt.tight_layout()
plt.show()






from sklearn.model_selection import train_test_split


image_files = []
labels = []


for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    
    class_files = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
    image_files.extend(class_files)
    labels.extend([class_name] * len(class_files)) 

image_files, labels = np.array(image_files), np.array(labels)
indices = np.arange(len(image_files))
np.random.shuffle(indices)
image_files = image_files[indices]
labels = labels[indices]

train_files, temp_files, train_labels, temp_labels = train_test_split(
    image_files, labels, test_size=0.30, random_state=42)  

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.10, random_state=42) 

print("Training set size:", len(train_files))
print("Validation set size:", len(val_files))
print("Test set size:", len(test_files))


from pandas import DataFrame

train_df = DataFrame({'filename': train_files, 'class': train_labels})
val_df = DataFrame({'filename': val_files, 'class': val_labels})
test_df = DataFrame({'filename': test_files, 'class': test_labels})

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=50,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=50,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=(200, 200),
    batch_size=50,
    class_mode='categorical'
)

#Batch check
print("Found classes:", train_generator.class_indices)
print("Number of images found:", train_generator.samples)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# Get the class indices
class_indices = train_generator.class_indices

# Invert the dictionary to get the class labels
class_labels = {v: k for k, v in class_indices.items()}

# Get the class labels from the generator
labels = train_generator.classes

# Plotting the histogram of class distribution
plt.figure(figsize=(10, 6))

# Calculate the number of classes
num_classes = len(class_labels)

# Plot the histogram
hist, bins, _ = plt.hist(labels, bins=num_classes, alpha=0.7, color='blue', edgecolor='black')

# Calculate the middle positions of the bars for aligning the ticks
middle_positions = 0.5 * (bins[:-1] + bins[1:])

# Set the ticks to align with the middle of each bar
plt.xticks(middle_positions, list(class_labels.values()), rotation='vertical')

plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Distribution of Classes in Training Data')
plt.tight_layout()
plt.show()


# Get the class indices
class_indices =val_generator.class_indices

# Invert the dictionary to get the class labels
class_labels = {v: k for k, v in class_indices.items()}

# Get the class labels from the generator
labels = val_generator.classes

# Plotting the histogram of class distribution
plt.figure(figsize=(10, 6))

# Calculate the number of classes
num_classes = len(class_labels)

# Plot the histogram
hist, bins, _ = plt.hist(labels, bins=num_classes, alpha=0.7, color='blue', edgecolor='black')

# Calculate the middle positions of the bars for aligning the ticks
middle_positions = 0.5 * (bins[:-1] + bins[1:])

# Set the ticks to align with the middle of each bar
plt.xticks(middle_positions, list(class_labels.values()), rotation='vertical')

plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Distribution of Classes in Validation Data')
plt.tight_layout()
plt.show()


# Get the class indices
class_indices =test_generator.class_indices

# Invert the dictionary to get the class labels
class_labels = {v: k for k, v in class_indices.items()}

# Get the class labels from the generator
labels = test_generator.classes

# Plotting the histogram of class distribution
plt.figure(figsize=(10, 6))

# Calculate the number of classes
num_classes = len(class_labels)

# Plot the histogram
hist, bins, _ = plt.hist(labels, bins=num_classes, alpha=0.7, color='blue', edgecolor='black')

# Calculate the middle positions of the bars for aligning the ticks
middle_positions = 0.5 * (bins[:-1] + bins[1:])

# Set the ticks to align with the middle of each bar
plt.xticks(middle_positions, list(class_labels.values()), rotation='vertical')

plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.title('Distribution of Classes in Test Data')
plt.tight_layout()
plt.show()
# Count the number of images in each directory
num_train_images = len(train_generator.filenames)
num_test_images = len(test_generator.filenames)
num_val_images = len(val_generator.filenames)

# Create a bar chart to visualize the distribution of images
labels = ['Train', 'Test', 'Validation']
num_images = [num_train_images, num_test_images, num_val_images]

plt.bar(labels, num_images, color='skyblue')
plt.xlabel('Datasets')
plt.ylabel('Number of Images')
plt.title('Distribution of Images in Datasets')

# Add labels with the number of images on the bars
for i, v in enumerate(num_images):
    plt.text(i, v + 10, str(v), ha='center', va='top')

plt.show()


import random
import os
from PIL import Image

# Number of images you want to show from each class
num_images_per_class = 5

# Load and display random images from each class
plt.figure(figsize=(15, 10))
for idx, class_name in enumerate(class_names):
    # Get the list of image files from the directory
    image_files = os.listdir(os.path.join(data_dir, class_name))
    
    # Randomly select 5 images from the list
    random_image_indices = random.sample(range(len(image_files)), num_images_per_class)
    
    # Iterate over the randomly selected image indices
    for j, random_image_index in enumerate(random_image_indices):
        # Load the image
        image_path = os.path.join(data_dir, class_name, image_files[random_image_index])
        img = Image.open(image_path)
        
        # Plot the image
        plt.subplot(num_images_per_class, len(class_names), j * len(class_names) + idx + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')

plt.tight_layout()
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters, kernel_size, strides=(1, 1)):
    # Shortcut branch
    shortcut = x

    # Main branch
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection
    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Max pooling layer
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Deep residual blocks
    for _ in range(3):
        x = residual_block(x, filters=64, kernel_size=(3, 3))

    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = layers.Dense(1000, activation='relu')(x)

    # Output layer
    outputs = layers.Dense(2, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
input_shape = (224, 224, 3)  # Example input shape for breast cancer scan
model = create_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Generate dummy data for training
import numpy as np
x_train = np.random.randn(1000, 224, 224, 3)
y_train = np.random.randint(0, 2, size=(1000,))

# Train the model
epochs = 30
batch_size = 50
steps_per_epoch = len(x_train) // batch_size
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, steps_per_epoch=steps_per_epoch)


# Save the trained model to the specified location
model.save('C:/Users/amkan/Downloads/archive/ai4all.h5')



import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('C:/Users/amkan/Downloads/archive/ai4all.h5')

# Define the class labels
class_labels = ['Healthy', 'Cancerous']

# Function to predict the class of an image
def predict_image_class(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image data

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

numOfCases = 0
correct = 0
# Loop through each file in the directory
for filename in os.listdir('C:/Users/amkan/Downloads/archive/testcase'):
    # Get the full path to the file
    f = os.path.join('C:/Users/amkan/Downloads/archive/testcase', filename)
    
    # checking if it is a file
    if os.path.isfile(f):
        # Predict the class of the image
        predicted_class = predict_image_class(f)
        print("File:", filename, "Predicted class:", predicted_class)

        numOfCases+=1
        if(predicted_class in filename):
            correct += 1

print(f"Prediction Accuracy: {(correct/numOfCases)*100} among 600 files")




