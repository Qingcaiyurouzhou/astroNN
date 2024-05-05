from keras import utils
import numpy as np
from sklearn.model_selection import train_test_split
import pylab as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from astroNN.datasets import galaxy10

from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.regularizers import l2

def residual_block(input_tensor, filters, strides=(1, 1)):
    filters1, filters2 = filters
    
    x = Conv2D(filters1, kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    
    if strides != (1, 1):
        input_tensor = Conv2D(filters2, kernel_size=(1, 1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)
    
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    
    return x

def ResNet(input_shape=(69, 69, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    x = residual_block(x, [64, 64], strides=(1, 1))
    x = residual_block(x, [64, 64], strides=(1, 1))
    x = residual_block(x, [128, 128], strides=(2, 2))
    x = residual_block(x, [128, 128], strides=(1, 1))
    x = residual_block(x, [256, 256], strides=(2, 2))
    x = residual_block(x, [256, 256], strides=(1, 1))
    x = residual_block(x, [512, 512], strides=(2, 2))
    x = residual_block(x, [512, 512], strides=(1, 1))
    
    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    return model

# Load Galaxy10 dataset
images, labels = galaxy10.load_data()

# Convert labels to categorical 10 classes
labels = utils.to_categorical(labels, 10)

# Convert to desirable types
labels = labels.astype(np.float32)
images = images.astype(np.float32)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, shuffle=True)

#train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)

# Create ResNet model
resnet_model = ResNet()

# Compile the model
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = resnet_model.fit(train_images, train_labels, batch_size=64, epochs=30, validation_data=(test_images, test_labels))

# Save the model
resnet_model.save('resnet_model.h5')

# Plot training history (loss and accuracy)
plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = resnet_model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict labels for the test set
predicted_labels = resnet_model.predict(test_images)
predicted_classes = np.argmax(predicted_labels, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
