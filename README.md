# Formula 1 Podium Prediction

**Advanced Computer Lab, Milestone 1**
Team 23: Rana Ali – 58-11078, Abdelrahman Mohamed – 58-10649, Shahd Alaa – 58-22017, Amr Mostafa – 58-14976.

## The task

For every driver in every race, predict whether he finishes on the podium (`positionOrder` ≤ 3), using only information available **after qualifying and before the race starts**. Results, status, points, laps, pit stops and standings of the race itself are known only afterwards, so they are never used as features of that race.

Data: [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020), the Kaggle/Ergast dataset with 14 linked tables.

## Repository contents

| File | What it is |
|---|---|
| `acl-final.ipynb` | The run-all notebook: audit, cleaning, data-engineering answers, features, models, ablation, explanations and the inference function, with every output from one committed Kaggle run |
| `f1_modeling_table.csv` | The modelling table: one row per (raceId, driverId), 24,739 rows |
| `Milestone1_ACL_Report.pdf` | The analytical report |

### The modelling table (`f1_modeling_table.csv`)

- **24,739 rows, unique on (raceId, driverId).** The notebook checks this before saving.
- **Columns:**
  - identifiers: `raceId`, `driverId`, `constructorId`, `year`, `race_order`;
  - `split` (train 1950–2019, validation 2020–21, test 2022–24);
  - `dedup_ambiguous`: marks the 24 shared-drive pairs whose starting car the data cannot show;
  - `podium`: the target;
  - the **33 model input columns**.
- **The features are saved before filling and scaling.** The median fill and the scaling are learned from the training seasons only, after the split. Missing values in older seasons are therefore expected; for example, qualifying data starts in 1994. They are marked by the availability flags (`has_quali`, `has_lap_times`, `has_team_standings`, `has_driver_history`, `has_teammate`).
- **Duplicates.** In 1950–1978, shared drives give 85 repeated (race, driver) pairs. The kept row is the driver's starting car, which is the row with the fewest laps; it gets the driver's official podium label. Section 5 of the notebook shows the evidence and the cases where this rule fails.

## Notebook structure

0. **How we judge a choice:** every decision is tested on predictions of seasons the model was not fitted on.
1. Setup and loading
2. Audit before cleaning: missing-value markers, keys, foreign keys
3. Semantic typing and outliers
4. Race chronology
5. One row per driver per race: the duplicate rule and its evidence
6. Scope of the table: the Indianapolis 500 and drivers who never started
7. Qualifying data
8. Train / validation / test split, by season
9. Building the candidate features: 48 candidates, with a leak test
10. Feature selection: rolling folds, two model types, 24 features kept
11. Preprocessing of the final features, with the preprocessing-order experiments
12. The modelling table, saved as `f1_modeling_table.csv`
13. The three data-engineering questions
14. Predictive models: baselines, three modelling attempts, final comparison, feature-group ablation
15. Explaining the models: logistic regression weights, permutation importance, SHAP, LIME
16. Inference function, shown on a complete record and on a partly missing one
17. Summary and known limitations

## How to run

**On Kaggle (recommended):**
1. Add the dataset `Formula 1 World Championship (1950 - 2024)` to the notebook.
2. Turn **Internet on**: the first cell installs `shap` and `lime`.
3. Run **Save & Run All**. A full run takes about 80–100 minutes on Kaggle's CPU.

**Locally:** set `DATA` in section 1 to the folder that holds the 14 CSV files, then install:

```
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn keras tensorflow shap lime
```

The split is by season, so it involves no randomness. Every random step uses a fixed seed (`SEED`): model training, the five network seeds, the bootstrap ranges, permutation importance, SHAP and LIME. A run on Kaggle therefore reproduces the saved outputs.
