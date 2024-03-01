# dCNN-for-trash-classification


### Exploratory Data Analysis

The initial exploration revealed significant class imbalance, with the "trash" class being notably underrepresented compared to others. This imbalance poses a challenge for model training, potentially leading to overfitting and underfitting issues.

### Data Pipeline Development

A robust data pipeline was developed, incorporating data augmentation techniques and normalization to enhance model performance. Batch size experiments were conducted to find an optimal balance between model performance and training efficiency.

### Handling Imbalanced Data

Class imbalance was addressed by computing class weights, ensuring equal attention to all classes during model training. This approach aimed to mitigate the impact of minority classes on model performance.

### Baseline Model Development

A baseline model was established, serving as a foundation for subsequent enhancements. Initial model architecture and hyperparameters were chosen to set a performance baseline for comparison.

### Model Enhancement Through Experimentation

Iterative experimentation was conducted to address overfitting and improve model generalization. This involved adding more layers to the model and adjusting hyperparameters to optimize performance.

### Handling Model Output

Methods were implemented to enhance the interpretability of model predictions, providing insights into alternative potential classifications. This approach aimed to improve understanding and trust in model outputs.

### Hyperparameter Tuning with Keras Tuner

Hyperparameters were optimized using Keras Tuner, exploring various configurations to improve model efficiency and performance. Experimentation focused on parameters such as the number of layers, optimizer functions, and learning rates.

### Re-train the Model on Different Data Source

Additional data sources were explored to diversify the training dataset and improve model robustness. However, resource limitations and complexity challenges were encountered during this process.

### Resource Management and Future Directions

Efforts were made to optimize code and manage computational resources effectively. Future work includes exploring alternative architectures, data augmentation techniques, and addressing class imbalance challenges.

### Conclusion of Methodology

The methodology employed a systematic approach to dataset exploration, model development, and iterative experimentation. While significant progress was made, further improvements are needed to enhance model predictability and robustness for real-world applications.

### Source of Data

The dataset used in this project was obtained from Kaggle, provided by CCHANG in 2018. You can find the dataset [here](https://www.kaggle.com/ds/81794).
