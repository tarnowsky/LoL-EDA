# League of Legends Early Game Analytics: Predicting Victory

## Project Overview

This repository hosts an Exploratory Data Analysis (EDA) project focused on identifying key factors in the first 10 minutes of League of Legends (LoL) games that predict the ultimate victory of the blue team. The dataset comprises detailed statistics from high-elo matches (Diamond I - Master tier), providing a rich ground for uncovering impactful early-game dynamics.

The core objective is to understand which early-game metrics (gold, experience, kills, objectives, etc.) are most indicative of a win, laying the groundwork for potential predictive modeling.

## Dataset Description

The dataset contains **9,879 unique games** and **40 attributes**, detailing various statistics captured at the 10-minute mark. Each row represents a single game with a unique `gameId`.

**Key attributes include:**
* `blueWins`: The target variable (1 if blue team wins, 0 otherwise).
* **Team Statistics (prefixed `blue` or `red`):** `WardsPlaced`, `WardsDestroyed`, `FirstBlood`, `Kills`, `Deaths`, `Assists`, `EliteMonsters` (Dragons, Heralds), `TowersDestroyed`, `TotalGold`, `AvgLevel`, `TotalExperience`, `TotalMinionsKilled`, `TotalJungleMinionsKilled`, `CSPerMin`, `GoldPerMin`.
* **Differential Statistics:** `blueGoldDiff`, `blueExperienceDiff` (representing gold/experience advantage over the opposing team).

## Exploratory Data Analysis (EDA) Highlights

The EDA phase revealed critical insights into the dataset and game dynamics:

* **Data Quality:** The dataset is of **high quality**, with no missing values and strong internal consistency (e.g., `kills` of one team matching `deaths` of the other).
* **Target Variable Balance:** The `blueWins` variable is well-balanced, with **49.90% blue team victories**, making it suitable for classification tasks.
* **Key Predictors of Victory:**
    * **Economic Advantage:** `blueGoldDiff` (correlation: 0.51) and `blueExperienceDiff` (correlation: 0.49) are the strongest indicators of victory, highlighting the paramount importance of early-game economic superiority. Winning blue teams averaged **over 1200 gold and 900 experience advantage**.
    * **Aggression & Objectives:** Higher `blueKills` (correlation: 0.34) and control over objectives like `blueDragons` (correlation: 0.21) and `blueTowersDestroyed` (correlation: 0.12) are significantly correlated with winning. Notably, winning blue teams destroyed **243% more towers** on average.
* **Outlier Analysis:** While some outliers were observed (e.g., in `Heralds` (18.80% for blue team) and `TowersDestroyed` (4.70% for blue team) counts), they are considered natural variations in game events and do not compromise data integrity.


## Strategic Implications

The findings suggest that early-game strategy in League of Legends for high-tier players should prioritize:
1.  **Securing Gold and Experience Leads:** This is the most crucial factor for achieving victory.
2.  **Aggressive Play:** Focusing on kills and assists to gain early momentum and pressure.
3.  **Objective Control:** Prioritizing dragons, heralds, and early tower pushes to translate advantages into tangible map control.

## Future Work

This EDA serves as the foundation for future work. Next steps include:
* Feature engineering and selection for predictive modeling.
* Developing and evaluating various machine learning classification models (e.g., Logistic Regression, Decision Trees, Random Forests) to predict game outcomes.
* Hyperparameter tuning and model optimization to improve prediction accuracy.
* In-depth analysis of model performance and interpretability.
* Consideration of role-specific analysis to understand individual player impact in the early game.

## How to Use This Repository

To replicate the analysis and explore the data yourself:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tarnowsky/LoL-EDA.git](https://github.com/tarnowsky/LoL-EDA.git)
    cd LoL-EDA
    ```
2.  **Install dependencies:**
    Ensure you have Python installed. Then, install the required libraries. It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Execute the Python script to perform the EDA and generate all the visualizations and console outputs.
    ```bash
    python data_analisys.py
    ```
4.  **Explore the report:**
    Open the `Report.pdf` file to view the detailed analysis and findings.

## Contact

For any questions, suggestions, or collaborations, feel free to open an issue or contact me directly.

---
