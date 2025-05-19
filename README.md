
---

### Alzheimer's MRI Classification

This project builds two machine learning models to classify brain MRI images into one of three categories based on dementia severity:

* **Mild Demented**
* **Moderate Demented**
* **Non-Demented**

It uses approximately **34,000 brain MRI images** to train and validate deep learning models for multi-class classification. The pipeline includes preprocessing, data balancing, model training, and hyperparameter tuning using Optuna.

---

### Dataset Description

The dataset includes MRI brain scans stored in class-labeled folders:

* `Mild Demented`
* `Moderate Demented`
* `Non Demented`

Data imbalance is handled by:

* **Undersampling** the overrepresented class.
* **Oversampling** minority classes using image augmentation (`ImageDataGenerator`).

---

### Model Architecture

We use a customizable **Artificial Neural Network (ANN)** model built using `tf.keras.Sequential`.

Key architectural components:

* Configurable number of **hidden layers** (via `num_layers`)
* **Activation functions**: `"relu"`, `"sigmoid"`, or `"tanh"`
* **Dropout** layers after each dense layer (to prevent overfitting)
* **Output Layer** with `softmax` for 3-class classification
* **Loss Function**: `sparse_categorical_crossentropy`
* **Optimizer**: `"adam"`, `"sgd"`, or `"rmsprop"` (with tunable learning rate)

---

### Hyperparameter Tuning

We use **Optuna**, a powerful hyperparameter optimization library, to tune:

* Number of layers
* Units per layer
* Dropout rate
* Activation function
* Optimizer
* Learning rate

Optuna:

* Randomly samples values
* Remembers best trial using `study.best_trial`
* Returns the best configuration to build the final model

```python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)
best_params = study.best_trial.params
final_model = objective(optuna.trial.FixedTrial(best_params))
```

---

### Evaluation Metrics

The primary evaluation metric is:

* **Validation Accuracy**

Model performance is monitored using training/validation accuracy and loss.

---
### Results

* The tuned ANN model achieved a validation accuracy of **93.72%** on the resampled dataset.

---
