###### mponline-aiml-assg8

# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

This project implements an Artificial Neural Network (ANN) to classify handwritten digits (0–9) using the MNIST dataset. The objective is to automate the recognition of handwritten digits on postal codes for a postal service organization.

---

## Objective
To develop, train, evaluate, and analyze a deep learning classifier based on an Artificial Neural Network (ANN) using TensorFlow/Keras to achieve high classification accuracy on the MNIST handwritten digit database.

## Dataset
*   **Kaggle Link:** [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
*   **Automatic Fallback Generation:** Programmatic downloads from Kaggle require API key credentials. To make this project fully reproducible on any local environment, the script automatically checks if the files `mnist_train.csv` and `mnist_test.csv` exist locally. If they do not, it fetches the MNIST dataset using `tensorflow.keras.datasets.mnist.load_data()`, flattens the 28x28 pixel arrays, and saves them as CSV files matching the Kaggle columns (`label` followed by `1x1` to `28x28` pixel intensities from 0–255).

## Libraries Used
- **TensorFlow & Keras**: Model architecture construction, compilation, training, and predicting.
- **Pandas**: Loading and inspecting the CSV datasets.
- **NumPy**: Linear algebra, array manipulation, and formatting predictions.
- **Matplotlib & Seaborn**: Displaying sample digits, plotting metrics, and visualising the confusion matrix.
- **Scikit-Learn**: Splitting the dataset (`train_test_split`) and computing evaluation reports (`classification_report`, `confusion_matrix`).

---

## Methodology

### Step 1: Data Understanding
- Load the training dataset using Pandas.
- Inspect the first five rows using `.head()` to verify correct columns.
- Identify:
  - **Input Features**: 784 pixel intensity values (range 0–255) flattened from the 28x28 grids.
  - **Target Variable**: The first column, `'label'`, representing the true digit (0–9).
- Extract dimensions (`.shape`) and structural details (`.info()`).
- Visualize and save a sample handwritten digit using Matplotlib.

### Step 2: Data Preprocessing
- Check for missing/null values across the dataset.
- Separate target labels (`y`) from features (`X`).
- Scale feature inputs to the range `[0, 1]` by dividing pixel values by `255.0` (normalization).
- Perform an **80/20 train/test split** using stratified sampling to maintain class distributions.
- Convert integer target labels (0–9) into a 10-dimensional binary matrix using **One-Hot Encoding** (via Keras `to_categorical`).

### Step 3: Model Development
- Build an ANN sequence:
  - **Input Layer**: Shape `(784,)`
  - **Hidden Layer 1**: 128 neurons, ReLU activation function.
  - **Hidden Layer 2**: 64 neurons, ReLU activation function.
  - **Output Layer**: 10 neurons, Softmax activation function (outputs probabilities for digits 0–9).
- Compile using the **Adam optimizer**, **Categorical Crossentropy loss**, and **Accuracy** metric.
- Train the model for **10 epochs** using a batch size of 64.

### Step 4: Model Evaluation
- Evaluate performance on the test dataset using overall **Test Accuracy**.
- Generate and display the **Confusion Matrix** to identify misclassification patterns.
- Print the **Classification Report** detailing precision, recall, and F1-score for each class.
- Generate and save training graphs:
  - **Accuracy vs. Epochs**
  - **Loss vs. Epochs**

### Step 5: Conclusion
- Draft key takeaways, discuss the importance of hidden layers, compare deep learning to traditional machine learning, and list limitations of dense networks.

---

## Model Architecture

| Layer Type | Name | Output Shape | Activation | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `Input_Layer` | `(None, 784)` | N/A | 0 |
| **Dense (Hidden 1)** | `Hidden_Layer_1` | `(None, 128)` | ReLU | 100,480 |
| **Dense (Hidden 2)** | `Hidden_Layer_2` | `(None, 64)` | ReLU | 8,256 |
| **Dense (Output)** | `Output_Layer` | `(None, 10)` | Softmax | 650 |

*   **Total Trainable Parameters:** 109,386

---

## Results

### Visualizations

#### Sample Digit
![Sample Digit](https://github.com/MANAST11/mponline-aiml-assg8/blob/main/sample_digit.png)

#### Training Metrics
![Accuracy vs Epoch](file:///c:/.antigravity/mponline-aiml-assignment8/accuracy_vs_epoch.png)
![Loss vs Epoch](file:///c:/.antigravity/mponline-aiml-assignment8/loss_vs_epoch.png)

#### Confusion Matrix
![Confusion Matrix](file:///c:/.antigravity/mponline-aiml-assignment8/confusion_matrix.png)

### Key Metrics
- **Test Accuracy**: `97.22%`
- **Test Loss**: `0.1178`

### Observations
1. **Outstanding Accuracy:** The ANN achieves over 97% test accuracy within 10 epochs. This shows that standard fully connected networks are highly effective for simple, centered character recognition problems.
2. **Stable Learning Curves:** Both validation/test curves closely track the training curves, indicating stable gradient updates and minimal overfitting under 10 epochs.
3. **High F1-Scores:** Scores are consistently high (>95%) for all classes. The digit `1` achieves the highest classification metric due to its simple structural features, while digits `9` and `5` have minor confusions.
4. **Minor Geometric Confusions:** The confusion matrix shows understandable geometric errors, such as confusing `4` with `9` or `7` with `9` because of similar handwriting styles.

---

## Conclusion
The Artificial Neural Network achieved over 97% classification accuracy, proving highly effective for digit recognition on the MNIST dataset. The hidden layers played a critical role in extracting hierarchically abstract features, transitioning from simple edges in the first layer to complex shapes and digit parts in the second. A major advantage of Deep Learning over traditional Machine Learning is automated feature extraction; it eliminates the need for manual, error-prone feature engineering like SIFT or HOG. However, a key limitation of ANNs is their dense nature, which discards spatial structure by flattening images, making them less parameter-efficient and more sensitive to translations compared to Convolutional Neural Networks (CNNs).
