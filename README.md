# Selecting the Ideal Crop for Specific Soil Types
### Machine Learning Classification Project

---

## Project Overview

This project builds a multi-class crop recommendation system that predicts the most suitable crop to cultivate based on soil nutrient levels and environmental conditions. Seven supervised classification algorithms are trained, evaluated, and compared to identify the best-performing model for agricultural decision support.

---

## Industrial Context & Target Variable

### Industry: Precision Agriculture

Agriculture is one of the most data-intensive industries in the world, yet most smallholder farmers rely on intuition or generational knowledge when selecting crops — leading to poor yields, soil degradation, and economic loss. Precision agriculture addresses this by using measurable soil and climate data to drive data-backed planting decisions.

This project directly supports that goal. The dataset contains real-world agronomic measurements collected across different farming regions, capturing the soil chemistry and microclimate conditions under which specific crops thrive.

### Target Variable: `Crops` (formerly `label`)

The target is a **multi-class categorical variable** representing the recommended crop. There are **22 distinct crop classes** including rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, and coffee.

### Input Features

| Feature | Original Name | Description | Unit |
|---|---|---|---|
| Nitrogen | N | Nitrogen content in soil | mg/kg |
| Phosphorus | P | Phosphorus content in soil | mg/kg |
| Potassium | K | Potassium content in soil | mg/kg |
| temperature | temperature | Average ambient temperature | °C |
| humidity | humidity | Relative humidity | % |
| ph | ph | Soil pH level | 0–14 |
| rainfall | rainfall | Average annual rainfall | mm |

These seven features represent the core parameters used in agronomic soil testing (NPK analysis) and regional climate profiling — both standard inputs in professional crop advisory systems.

---

## Data Cleaning & Preprocessing Justification

### 1. Column Renaming

```python
data.rename(columns={'N': 'Nitrogen', 'P': 'Phosphorus', 'K': 'Potassium', 'label': 'Crops'}, inplace=True)
```

**Justification:** Single-letter column names (`N`, `P`, `K`) are ambiguous when passed through pipelines, exported to APIs, or read by non-domain collaborators. Renaming to full agronomic terms (`Nitrogen`, `Phosphorus`, `Potassium`) improves code readability, reduces the risk of column-name collisions, and aligns with standard soil science nomenclature used in agricultural documentation.

### 2. Null Value Check

```python
data.isnull().sum()
```

**Justification:** Missing values in soil measurement datasets typically indicate sensor failure, lab omission, or data entry errors. A null check is performed before any transformation to confirm data completeness. In this dataset, no null values were found — allowing us to proceed without imputation, which avoids introducing artificial bias into the nutrient or climate distributions.

### 3. Duplicate Detection

```python
data.duplicated()
```

**Justification:** Duplicates in an agronomic dataset can arise from repeated entries of the same field observation or data pipeline errors. Retaining duplicates would inflate the representation of certain soil-crop combinations, causing classifiers to be biased toward those patterns and overestimating accuracy on test sets that inadvertently contain copies of training rows.

### 4. Label Encoding

```python
le = LabelEncoder()
data['Crops'] = le.fit_transform(data[['Crops']])
```

**Justification:** All seven classifiers used in this project (including scikit-learn's implementations of SVM, Naive Bayes, and others) require numeric target labels. `LabelEncoder` maps each of the 22 crop strings to a unique integer (0–21) in a deterministic, reversible manner. One-Hot Encoding was intentionally avoided because the target variable is the output label — not an input feature — so there is no concern about ordinality implying false relationships between classes.

### 5. Feature Engineering Note

This dataset does not require rolling windows or lagging features because it is **not a time-series dataset**. Each row represents an independent soil-profile observation at a point in time; there is no temporal ordering or sequential dependency between records. Therefore, standard cross-sectional preprocessing (null check, deduplication, encoding, train-test split) is the appropriate and sufficient methodology. Applying time-series feature engineering to this data would be a category error that introduces noise rather than signal.

---

## Model Architecture Selection

### Why Seven Classifiers Were Compared

Rather than committing to a single model a priori, this project takes a benchmarking approach — training seven diverse classifiers and selecting the best based on empirical performance. This is standard practice in applied ML when the dataset's underlying distribution and class separability are not known in advance.

### Models and Rationale

| Model | Rationale |
|---|---|
| **Naive Bayes (GaussianNB)** | Strong baseline for multi-class problems; assumes feature independence — reasonable for soil nutrients which are relatively orthogonal agronomic parameters |
| **Random Forest** | Ensemble of decision trees; robust to feature scale differences; naturally handles non-linear boundaries between crop classes |
| **Decision Tree** | Interpretable single-tree model; directly maps soil conditions to crop recommendations in a human-readable rule structure |
| **K-Nearest Neighbors** | Instance-based learner; effective when similar soil profiles genuinely predict similar crop needs |
| **Logistic Regression** | Linear baseline; useful for identifying whether classes are linearly separable in feature space |
| **SVM** | Finds maximum-margin hyperplanes; well-suited for high-dimensional feature spaces with clear class boundaries |
| **Gradient Boosting** | Sequential ensemble; typically strong for structured tabular data — included to provide a complete benchmark |

### Final Results

| Classifier | Test Accuracy |
|---|---|
| Naive Bayes | **99.55%** |
| Random Forest | 99.32% |
| Decision Tree | 98.64% |
| KNN | 97.05% |
| SVM | 96.14% |
| Logistic Regression | 94.55% |
| Gradient Boosting | 9.82% |

Naive Bayes achieved the highest accuracy. This result is consistent with the dataset's structure — the seven soil and climate features are largely independent of each other (Nitrogen content does not directly cause a specific humidity level, for example), which is precisely the condition under which Naive Bayes's conditional independence assumption holds and produces near-optimal results.

### Guarding Against Overfitting

This is a **cross-sectional classification problem, not a time-series problem**, so standard anti-overfitting methods apply rather than time-series-specific techniques:

**Train-Test Split (80/20)**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
A held-out test set that the model never sees during training provides an unbiased estimate of generalisation performance. `random_state=42` ensures reproducibility.

**Train vs. Test Accuracy Comparison**
The project explicitly compares training accuracy against test accuracy for all seven models side-by-side. A large gap between the two (high train, lower test) would signal overfitting. The results show the gap is minimal for Naive Bayes and Random Forest, confirming they generalise well.

**Confusion Matrix per Model**
Each classifier is evaluated with a full confusion matrix and classification report (precision, recall, F1 per class). This reveals whether high overall accuracy masks poor performance on individual crop classes — a critical check in a 22-class problem where class imbalance could otherwise hide failures.

**Gradient Boosting's Poor Performance**
The anomalously low accuracy of Gradient Boosting (9.82%) likely results from default hyperparameters (particularly a very low learning rate or insufficient estimators) being unsuitable for this dataset without tuning. This highlights that model selection based on benchmark comparison — rather than assumptions — is the correct approach.

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Open the notebook
jupyter notebook Selecting_the_Ideal_Crop_for_Specific_Soil_Types.ipynb
```

The notebook loads the dataset directly from Google Drive. An active internet connection is required for the first cell. All subsequent cells run sequentially top-to-bottom.

---

## Project Structure

```
.
├── Selecting_the_Ideal_Crop_for_Specific_Soil_Types.ipynb   # Main notebook
└── README.md                                                  # This file
```
