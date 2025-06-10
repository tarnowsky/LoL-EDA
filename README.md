# League of Legends Early Game Analytics: Predicting Victory

## Project Overview

This repository contains a two-phase data science project analyzing early-game statistics from **League of Legends (LoL)** to identify and model the key factors that predict the **victory of the blue team**. The analysis focuses on the first 10 minutes of gameplay in high-elo matches (Diamond I to Master tier).

The project includes:
1. **Exploratory Data Analysis (EDA):** Identifying and interpreting features most strongly correlated with match outcomes.
2. **Predictive Modeling:** Building and evaluating machine learning models to forecast blue team victory based on early-game statistics.

## Dataset Description

The dataset comprises **9,879 unique games** with **40 features**, capturing a snapshot of gameplay at the 10-minute mark. Each row corresponds to a single match identified by a `gameId`.

**Notable attributes:**
- `blueWins`: Binary target variable (1 = blue team wins, 0 = red team wins).
- **Team-level stats:** `Kills`, `Deaths`, `Assists`, `EliteMonsters`, `WardsPlaced`, `TotalGold`, `AvgLevel`, `TotalExperience`, `TowersDestroyed`, etc., prefixed by `blue` or `red`.
- **Differential stats:** `blueGoldDiff`, `blueExperienceDiff` (blue advantage over red in gold/XP).

## Phase 1: Exploratory Data Analysis (EDA)

### Key Findings

- **Data Quality:** No missing values; good internal consistency.
- **Target Balance:** `blueWins` is well balanced (49.9% blue victories), suitable for binary classification.
- **Strong Predictors:**
  - `blueGoldDiff` (corr = 0.51) and `blueExperienceDiff` (corr = 0.49) – early economic lead is highly predictive.
  - `blueKills`, `blueAssists`, `blueDragons`, and `blueTowersDestroyed` also show positive correlation.
- **Outliers:** Present but meaningful, reflecting natural variation in match flow.

### Strategic Implications

Winning often correlates with:
- Superior **gold and XP** gains.
- Early **aggression** and teamfight success.
- **Objective control** (dragons, towers).

## Phase 2: Predictive Modeling

### Modeling Objectives

- Build machine learning models to predict blue team victory using early-game features.
- Evaluate models based on accuracy, precision, recall, and interpretability.

### Methods Used

- **Preprocessing:** Feature selection, scaling, data splitting.
- **Models Implemented:**
  - **Logistic Regression**
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
- **Evaluation:** Performance assessed on test data using confusion matrix and classification report.
- **Visualization:** Feature importance, decision paths, and prediction probabilities explored.

### Key Insights

- **Economic stats** (gold/XP diff) consistently rank among top predictors.
- Decision Trees provided good interpretability, highlighting simple rules tied to early-game leads.
- Random Forests offered improved accuracy at the cost of interpretability.

## How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tarnowsky/LoL-EDA.git
   cd LoL-EDA
   ```

2. **Install dependencies:**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Phase 1 (EDA):**
   ```bash
   python First\ Part/data_analisys.py
   ```

4. **Run Phase 2 (Modeling):**
   Open and run:
   ```bash
   Second\ Part/second_phase_codebase.ipynb
   ```

5. **View Reports:**
   - `First Part/Raport.pdf`: EDA summary
   - `Second Part/report.pdf`: Modeling phase summary

## Project Structure

```
.
├── data
│   └── high_diamond_ranked_10min.csv
├── First Part
│   ├── data_analisys.py
│   ├── histograms/
│   ├── Raport.pages
│   └── Raport.pdf
├── Second Part
│   ├── second_phase_codebase.ipynb
│   ├── decision_tree.png
│   ├── report.pages
│   └── report.pdf
├── requirements.txt
└── README.md
```
---
