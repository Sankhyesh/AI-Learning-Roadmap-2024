 

## Notes: Multi-Class Image Classification with ANNs on MNIST using Keras

This document outlines the process of building and training an **Artificial Neural Network (ANN)** for classifying handwritten digits from the **MNIST dataset**. This is a classic **multi-class classification** task, aiming to assign one of ten labels (digits 0-9) to each input image.

### 1. Core Objective & Dataset Introduction

* **Goal:** To understand and implement a solution for a multi-class classification problem using an ANN with the Keras library, focusing on the practical steps involved in handling image data, building a model, training it, and evaluating its performance.
* **Dataset: MNIST**
    * A widely-used benchmark dataset in machine learning and computer vision.
    * It contains **70,000 grayscale images of handwritten digits**.
        * 60,000 images for the training set.
        * 10,000 images for the test set.
    * Each image is **low-resolution (28x28 pixels)**.
    * **Significance:** MNIST serves as an excellent entry point for learning image classification due to its well-structured nature and manageable complexity.

    *Visual Aid Suggestion:* A montage showing several examples of each digit (0 through 9) from the MNIST dataset would effectively illustrate the data variability.

### 2. Setting up the Environment & Loading Data

First, we import the necessary libraries. **TensorFlow** is an open-source machine learning platform, and **Keras** is its high-level API for building and training neural networks.

```python
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
```

* **Significance:** These imports provide the tools for creating the neural network layers (`Sequential`, `Dense`, `Flatten`), loading the dataset (`keras.datasets.mnist`), and visualizing data (`matplotlib.pyplot`).

Next, we load the MNIST dataset. Keras provides a convenient function for this.

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

* This command unpacks the dataset into training data/labels (`X_train`, `y_train`) and testing data/labels (`X_test`, `y_test`).
* `X_train` and `X_test` will contain the pixel values of the images.
* `y_train` and `y_test` will contain the actual digit labels (0-9) for these images.

Let's inspect the shape of our test data:

```python
X_test.shape
```

* **Expected Output:**
    ```
    (10000, 28, 28)
    ```
* **Significance:** This output confirms that `X_test` contains 10,000 images, and each image is 28 pixels in height and 28 pixels in width. `X_train.shape` would similarly be `(60000, 28, 28)`.

Let's look at some of the training labels:

```python
y_train
```

* **Expected Output (example):**
    ```
    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
    ```
* **Significance:** This shows that `y_train` is an array of integers, where each integer is the label for the corresponding image in `X_train`. For instance, the first training image is a '5', the second is a '0', and so on.

We can visualize one of the images using `matplotlib`. Let's display the third image in the training set (index 2):

```python
plt.imshow(X_train[2])
plt.show() # Ensures the plot is displayed
```

* **Expected Output:**
    * A plot window will appear displaying the handwritten digit corresponding to `X_train[2]`. For example, if `y_train[2]` is '4', this will show an image of a handwritten '4'.
    *(Image: A 28x28 grayscale image of a handwritten digit, taken from the training set at index 2.)*
* **Significance:** Visualizing the data is a crucial step to ensure it's loaded correctly and to get an intuitive understanding of what the model will be working with.

### 3. Data Preprocessing: Normalization

Neural networks generally perform better when input data is scaled to a small range. The pixel values in MNIST images range from 0 (black) to 255 (white). We'll **normalize** these values to a range of **0 to 1** by dividing by 255.0.

```python
X_train = X_train / 255
X_test = X_test / 255
```

Let's check the pixel values of the first training image after normalization:

```python
X_train[0]
```

* **Expected Output:**
    ```
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)
    ```
    (A 28x28 array where all pixel values are now between 0.0 and 1.0)
* **Significance:** Normalization helps in faster convergence during training and prevents features with larger values from disproportionately influencing the learning process.

### 4. Building the Artificial Neural Network (ANN)

We will use the **Keras Sequential API**, which allows us to build a model layer by layer.

```python
model = Sequential()
```

The first layer we add is a **`Flatten` layer**. This converts the 2D image data (28x28 matrix) into a 1D array (784 elements), which can then be fed into dense layers.

```python
model.add(Flatten(input_shape=(28, 28)))
```

* `input_shape=(28,28)`: This specifies the dimensions of the input data for the first layer.

Next, we add **`Dense` layers** (fully connected layers):
* A hidden layer with **128 neurons** and **ReLU (Rectified Linear Unit) activation**.
* Another hidden layer with **32 neurons** and **ReLU activation**. (This is an addition compared to the initial single hidden layer discussed in some materials, often explored to see effects on performance).
* An output layer with **10 neurons** (one for each digit class 0-9) and **Softmax activation**.

```python
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu')) # Additional hidden layer
model.add(Dense(10, activation='softmax'))
```

* **Significance of Layers:**
    * **Flatten Layer:** Prepares the image data for the dense layers.
    * **Dense Hidden Layers (with ReLU):** These are the core learning layers. ReLU is a common activation function that helps with non-linearities and can mitigate the vanishing gradient problem. The number of neurons (128, 32) are hyperparameters that can be tuned.
    * **Dense Output Layer (with Softmax):** Essential for multi-class classification. Softmax converts the network's raw output scores into a probability distribution across the 10 classes, ensuring the probabilities sum to 1. The class with the highest probability is the model's prediction.

Let's view the **model summary**:

```python
model.summary()
```

* **Expected Output:**
    ```
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense (Dense)               (None, 128)               100480    
                                                                     
     dense_1 (Dense)             (None, 32)                4128      
                                                                     
     dense_2 (Dense)             (None, 10)                330       
                                                                     
    =================================================================
    Total params: 104938 (410.00 KB)
    Trainable params: 104938 (410.00 KB)
    Non-trainable params: 0 (0.00 B)
    _________________________________________________________________
    ```
* **Significance:** The summary shows each layer's type, its output shape (`None` indicates the batch size can vary), and the number of trainable parameters (weights and biases). This helps verify the architecture and understand its complexity. For example, the first dense layer has $784 \text{ inputs} \times 128 \text{ units} + 128 \text{ biases} = 100480$ parameters.

### 5. Compiling the Model

Before training, the model needs to be compiled. This step configures the learning process.

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
```

* **Key Components:**
    * **`loss='sparse_categorical_crossentropy'`**: This loss function is suitable for multi-class classification where true labels are integers (0, 1, ..., 9). It measures how well the model is performing.
    * **`optimizer='Adam'`**: Adam (Adaptive Moment Estimation) is an efficient and widely used optimization algorithm that adjusts the network's weights to minimize the loss.
    * **`metrics=['accuracy']`**: We want to monitor the **accuracy** (proportion of correctly classified images) during training and evaluation.

### 6. Training the Model

The model is trained using the `fit()` method on the training data.

```python
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)
```

* **Parameters:**
    * `X_train, y_train`: The training images and their corresponding labels.
    * `epochs=25`: An epoch is one complete pass through the entire training dataset. The model will see the training data 25 times.
    * `validation_split=0.2`: Reserves 20% of the training data to be used as a validation set. The model's performance on this validation set is monitored at the end of each epoch, which helps in detecting overfitting.
* **Expected Output (during execution):**
    ```
    Epoch 1/25
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3000 - accuracy: 0.9125 - val_loss: 0.1562 - val_accuracy: 0.9537
    Epoch 2/25
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1285 - accuracy: 0.9615 - val_loss: 0.1159 - val_accuracy: 0.9653
    ...
    Epoch 25/25
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0150 - accuracy: 0.9948 - val_loss: 0.1300 - val_accuracy: 0.9750
    ```
    (The exact loss/accuracy values will vary slightly each time you run it).
* **Significance:** The `history` object stores the training progress (loss and metrics for each epoch on both training and validation sets). The logs show the model improving over epochs.

### 7. Evaluating Performance & Making Predictions

First, let's get the model's predictions on the test set. `model.predict()` returns an array of probabilities for each image for each of the 10 classes.

```python
y_prob = model.predict(X_test)
```

We then find the class with the highest probability for each test image using `argmax()`. This gives us the predicted labels.

```python
y_pred = y_prob.argmax(axis=1)
```

Now, we can calculate the accuracy on the test set using `sklearn.metrics.accuracy_score`.

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

* **Expected Output:**
    ```
    0.97...
    ```
    (A float value representing the accuracy, e.g., `0.9745` meaning ~97.45% accuracy).
* **Significance:** This is the final measure of how well our trained model generalizes to new, unseen data.

We can plot the training history (loss and accuracy over epochs) to visualize the learning process.

Plotting training and validation loss:

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

* **Expected Output:**
    * A line plot showing two curves: one for training loss and one for validation loss across the 25 epochs. Ideally, both should decrease. If validation loss starts to increase while training loss decreases, it's a sign of overfitting.
    *(Image: A plot with epoch on the x-axis and loss on the y-axis. Two lines representing training loss and validation loss.)*

Plotting training and validation accuracy:

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

* **Expected Output:**
    * A line plot showing two curves: one for training accuracy and one for validation accuracy. Both should ideally increase and then plateau. A large gap between training and validation accuracy can also indicate overfitting.
    *(Image: A plot with epoch on the x-axis and accuracy on the y-axis. Two lines representing training accuracy and validation accuracy.)*
* **Significance of Plots:** These plots are crucial for diagnosing the training process, identifying overfitting or underfitting, and deciding if adjustments like early stopping, regularization, or architectural changes are needed.

Let's test the model on a single image from the test set. First, display the image:

```python
plt.imshow(X_test[1])
plt.show()
```

* **Expected Output:**
    * A plot window will display the second image (index 1) from the `X_test` dataset.
    *(Image: A 28x28 grayscale image of a handwritten digit, taken from the test set at index 1.)*

Now, predict this single image. Note that `model.predict()` expects a batch of images, so we reshape the single image `X_test[1]` to have a shape of `(1, 28, 28)`.

```python
model.predict(X_test[1].reshape(1, 28, 28)).argmax(axis=1)
```

* **Expected Output (example, if `X_test[1]` is the digit '2'):**
    ```
    array([2])
    ```
* **Significance:** This demonstrates how to use the trained model to predict the class of a new, single instance.

### 8. Discussion & Potential Improvements

* **Overfitting:** As seen in the plots, if training accuracy significantly surpasses validation accuracy, or if validation loss starts increasing, the model might be overfitting. This means it's learning the training data too well (including noise) and not generalizing effectively to unseen data. The provided model with an additional hidden layer and 25 epochs might show some signs of this depending on the run.
* **Improving Performance & Mitigating Overfitting:**
    * **Architectural Changes:** Experiment with the number of layers or neurons. Simpler models sometimes generalize better.
    * **Regularization Techniques:** Techniques like **Dropout** (randomly dropping neurons during training), **L1/L2 regularization** (adding penalties to the loss function based on weight magnitudes) can help reduce overfitting. These are common next steps.
    * **Early Stopping:** Monitor validation loss and stop training when it no longer improves for a certain number of epochs.
    * **Data Augmentation:** For image tasks, creating more training data by slightly altering existing images (rotating, shifting, zooming) can improve generalization.
* **Convolutional Neural Networks (CNNs):** For image data, CNNs usually outperform standard ANNs (Dense layers only). CNNs have specialized layers (convolutional, pooling) designed to capture spatial hierarchies of features in images, making them more effective for tasks like digit recognition.

### Stimulating Learning Prompts

1.  **Impact of Optimizers:** The `Adam` optimizer was used. What might happen if you tried a different optimizer, like `SGD` (Stochastic Gradient Descent) or `RMSprop`? How might learning rate adjustments affect the training process with these optimizers?
2.  **Error Analysis:** If you were to examine the images that the model misclassifies, what kinds of patterns or characteristics in those images might you expect to find? How could this analysis inform further model improvements?

[End of Notes]
 