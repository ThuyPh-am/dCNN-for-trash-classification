# dCNN-for-trash-classification
An end-to-end deep learning project for classifying waste images using Convolutional Neural Networks (CNNs), with a focus on handling class imbalance and improving model generalization.
---
## Overview
This project builds a robust image classification pipeline to automatically categorize different types of waste. It includes data preprocessing, model development, hyperparameter tuning, and performance optimization.

Key challenges addressed:

- Class imbalance in real-world datasets
- Overfitting in deep CNN models
- Efficient training with limited computational resources
---
##  Dataset

- Source: Kaggle (provided by CCHANG, 2018)
- Task: Multi-class image classification of waste categories
- Challenge: Significant class imbalance (especially underrepresented "trash" class)

---

### Exploratory Data Analysis
Initial analysis revealed:

- Strong class imbalance across categories
- Risk of biased learning toward majority classes
- Need for augmentation and weighting strategies

### Data Pipeline Development

A robust data pipeline was implemented using:

- Image normalization
- Data augmentation (rotation, flipping, etc.)
- Batch size experimentation for performance vs efficiency trade-off

### Handling Imbalanced Data

To address imbalance:

- Computed **class weights**
- Applied during training to penalize bias toward majority classes

This improved the model’s ability to learn minority class representations.
---
##  Model Development
### Baseline Model Development

- Initial CNN architecture built as a performance benchmark
- Provided a reference for further improvements
### Model Enhancement Through Experimentation

- Added deeper layers to improve feature extraction
- Tuned architecture to reduce overfitting
- Adjusted training strategies for better generalization
---

### Handling Model Output

Methods were implemented to enhance the interpretability of model predictions, providing insights into alternative potential classifications. This approach aimed to improve understanding and trust in model outputs.
---
##  Hyperparameter Tuning
Used **Keras Tuner** to optimize:

- Number of layers
- Learning rate
- Optimizer choice

This systematic search improved both performance and training efficiency.
---

##  Model Interpretability

Implemented techniques to:

- Analyze prediction confidence
- Provide alternative class probabilities

This improves transparency and trust in model outputs.

---
##  Additional Experiments

- Attempted training on additional datasets to improve robustness
- Faced challenges due to:
  - Computational constraints
  - Increased pipeline complexity

---

##  Results

- Achieved strong classification performance despite dataset imbalance
- Demonstrated effectiveness of:
  - Data augmentation
  - Class weighting
  - Hyperparameter tuning

---

##  Limitations

- Class imbalance still impacts minority class performance
- Limited computational resources restricted large-scale experiments
- Model may require further tuning for real-world deployment

---

##  Future Work

- Explore advanced architectures (e.g., EfficientNet, ResNet)
- Improve data augmentation strategies
- Apply techniques like:
  - Focal Loss
  - Oversampling / SMOTE
- Optimize for edge deployment

---

##  Tech Stack

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy, Matplotlib

---

### Source of Data

The dataset used in this project was obtained from Kaggle, provided by CCHANG in 2018. You can find the dataset [here](https://www.kaggle.com/ds/81794).
