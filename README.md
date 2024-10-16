# GR Affinity Prediction

## Project Overview
Glucocorticoid receptors (GR) play a crucial role in various physiological processes in humans. This study utilizes data from the TOX21 database to predict the GR binding affinity of various compounds through feature engineering and machine learning models. The goal is to uncover specific structural characteristics that affect GR binding affinity.

## Dataset
We used the TOX21 database, focusing on the TOX21_GR_BLA_Agonist_ratio and TOX21_GR_BLA_Antagonist_ratio AC50 values. Compounds were classified into five categories based on their GR binding affinity.

## Methodology
### Feature Engineering
Four feature engineering methods were combined with five machine learning algorithms and one neural network to determine the best predictive model. Models included:
- XGBoost (using Permutation Importance for feature selection)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- AdaBoost
- Artificial Neural Network (ANN)

### Model Performance
The XGBoost model demonstrated the strongest generalization ability, achieving an AUC > 0.85 and an F1-score > 0.75 on the test set.

### SHAP Analysis
SHAP analysis identified the following molecular descriptors as significant contributors to GR binding affinity:
- Hydrophilicity (MolLogP), Molecular Weight (MolWt), BCUT2D_MWLOW
- Molecular Complexity (CIC0, ATSC7se, ZMIC2)
- Surface Electrostatic Properties (PEOE_VSA6)

### Project Structure
- `data/`: Contains the TOX21 dataset.
- `src/`: Source code for data processing, model training, and SHAP analysis.
- `models/`: Trained model files.
- `results/`: Analysis results and visualizations.
- `docs/`: Detailed documentation of the methodology.

## Installation & Usage
### 1. Clone the Repository
```bash
git clone https://github.com/LLEO97/GR_affinity.git
cd GR_affinity
