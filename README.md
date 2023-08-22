# Plant Leaf Classification Using Convolutional Neural Network (CNN)

This project focuses on classifying plant leaf images across different species using a deep learning approach. Utilizing a dataset from Kaggle, the system demonstrates effective leaf classification which can be employed in a range of applications.

## 📝 Table of Contents
- [Architecture and Methodology](#architecture-and-methodology)
- [Convolutional Neural Network](#convolutional-neural-network)
- [Experimental Results Analysis](#experimental-results-analysis)
- [Image Visualization and Prediction](#image-visualization-and-prediction)


## 🏗 Architecture and Methodology
The project is bifurcated into two phases:

### Phase 1 (Sequential Model):
- A two-layer convolutional neural network that only considers image inputs.
- Uses the Keras Sequential model API.
- Input layer accepts an array of image IDs.
- Features rectified using the ReLU operation after each convolution layer.
- Concludes with a fully connected MLP layer employing a softmax activation function.

### Phase 2 (Functional Model):
- Aims to fuse image data and pre-extracted features.
- Uses the Keras Functional API model for two different data types.
- Merges output from the last convolution layer with pre-extracted features.
- Trained using a Keras generator with an image augmenter generator and pre-extracted features array.

## 🧩 Convolutional Neural Network
CNN, a dominant method in deep learning, is widely recognized for image recognition tasks. The key operations in the CNN model include:
1. Convolution
2. Non Linearity (ReLU)
3. Max Pooling
4. Classification (Fully Connected Layer)

![Architecture of CNN](path_to_cnn_figure)

### Convolution:
- Extracts features from input images.
- Two convolutional layers used, the first takes a 96x96x1 image and a 5x5 filter matrix.
  
### ReLU:
- Ensures non-linearity among features.
- Replaces all negative pixel values in the feature map with zero.

### Max Pooling:
- Reduces feature map dimensions while retaining significant information.

### Fully Connected Layer:
- Functions as a traditional MLP using softmax activation function for the output layer.

## 📊 Experimental Results Analysis
Results are analyzed based on input structures:

### Sequential Model with Image Dataset:
- Achieved a training accuracy of 97%.
- Training and validation accuracy and loss visualized in ![Figure 2](path_to_figure_2).

### Functional API Model with Image and Pre-extracted Features:
- Demonstrated significant improvement over the sequential model.
- Approximate training accuracy of 99.0%.
- Training and validation accuracy and loss visualized in ![Figure 3](path_to_figure_3).

The dataset size being smaller than ideal for a robust CNN model may account for the lower performance of the sequential model. However, the inclusion of pre-extracted features in the functional model showed a marked improvement.

## 🌌 Image Visualization and Prediction
- The combined CNN model successfully classified raw images.
- Visualizations showcase the impact of CNN filters on different convolutional layers as seen in ![Figure 4](path_to_figure_4).
  
### Prediction:
- Given the absence of class labels in the Kaggle test dataset, a cross-validation approach was employed.
- The model predicted species like `Quercus Afares` accurately, providing top 3 predictions for images to determine the species.

