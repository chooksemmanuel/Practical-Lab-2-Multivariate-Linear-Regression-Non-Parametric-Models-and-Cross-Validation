# Practical Lab 2: Multivariate Linear Regression, Non-Parametric Models and Cross-Validation

**Course:** CSCN 8010 — Foundations of Machine Learning Framework  
**Student:** Emmanuel Ihejiamaizu  
**Student Number:** 9080005  
**Institution:** Conestoga College  
**Program:** Applied AI & Machine Learning

---

## Objective

Build a predictive model for diabetes disease progression using the [Scikit-Learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (442 patients, 10 baseline features). The model serves as a clinical screening tool to help physicians identify patients at risk.

## Dataset

- **442 samples** with 10 baseline physiological features: age, sex, BMI, blood pressure, and 6 blood serum measurements (s1–s6)
- **Target:** Quantitative measure of disease progression one year after baseline (range: 25–346)
- Features are pre-scaled (mean-centered, divided by std × √n)

## Project Structure

```
├── Lab2_Emmanuel_Ihejiamaizu_9080005.ipynb   # Main notebook with all code, plots & analysis
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules
└── README.md                                  # This file
```

## Models Evaluated

### Part 1 — Data Exploration & Preparation
- Exploratory Data Analysis: descriptive statistics, scatter plots, histograms, correlation heatmap
- Data quality assessment and train/validation/test split (75/10/15)

### Part 2 — Univariate Polynomial Regression (BMI)
- 6 polynomial models (degree 0–5) using BMI as the sole predictor
- Best model: **Degree 1 (Linear)** — selected via Occam's Razor given negligible R² differences across degrees
- Model equation, prediction, and trainable parameter analysis included

### Part 3 — Multivariate Models (All Features)
| Model | Val R² | Notes |
|-------|--------|-------|
| Polynomial (deg=2) | **0.60** | Best overall regression model |
| kNN (k=10) | 0.51 | Strong non-parametric baseline |
| kNN (k=5) | 0.49 | Slightly higher variance |
| Decision Tree (depth=3) | 0.41 | Conservative but stable |
| Decision Tree (depth=6) | 0.16 | Overfitting observed |
| Polynomial (deg=3) | << 0 | Catastrophic overfitting (286 features vs 330 samples) |
| Logistic Regression (C=1.0) | — | **82% accuracy** on binary risk classification |

## Evaluation Metrics

- **R² (Coefficient of Determination):** Proportion of variance explained
- **MAE (Mean Absolute Error):** Average absolute prediction error
- **MAPE (Mean Absolute Percentage Error):** Relative error as percentage
- **Accuracy** (for Logistic Regression classification)

## Key Findings

1. **Multivariate Polynomial Regression (degree 2)** is the best regression model, capturing feature interactions while avoiding overfitting.
2. **BMI is the strongest single predictor** (r = 0.59), but combining all features boosts Val R² from 0.45 to 0.60.
3. **The bias-variance tradeoff** is clearly demonstrated — degree 3 polynomial and deep decision trees overfit severely.
4. **Logistic Regression (C=1.0)** achieves 82% validation accuracy for binary high/low risk classification, offering a clinically actionable alternative.

## How to Run

```bash
# Clone the repository
git clone https://github.com/chooksemmanuel/Practical-Lab-2-Multivariate-Linear-Regression-Non-Parametric-Models-and-Cross-Validation.git

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open in VS Code
code .
# Then open the .ipynb file and Run All
```

## Tools & Technologies

Python 3 | VS Code | Jupyter Notebook | Scikit-Learn | Pandas | Matplotlib | Seaborn | NumPy
