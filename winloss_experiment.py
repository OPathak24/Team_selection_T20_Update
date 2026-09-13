import json
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FILE = "team_composition_updated_new.json"

RANDOM_SEEDS = [42, 52, 62, 72, 82]

TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024
TEST_END_YEAR = 2025

TOTAL_PLAYERS = 11


# ============================================================
# LOAD MATCH DATA
# ============================================================

def load_match_data():

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# PREPARE COMPLETE DATASET
# ============================================================

def prepare_dataset(match_data):

    rows = []

    for match in match_data:

        result = str(
            match.get("Result", "")
        ).strip().lower()

        # Keep only valid Win/Loss observations
        if result not in ["win", "loss"]:
            continue

        row = {

            "Match_No":
                match.get("Match_No"),

            "Year":
                match.get("Year"),

            "Opponent":
                match.get("Opponent"),

            "Pitch_Type":
                match.get("Pitch_Type"),

            "HomeAway":
                match.get("HomeAway"),

            "Venue":
                match.get("Venue"),

            "Team_Rank":
                match.get("Team_Rank", 10),

            "Result":
                result
        }

        # ----------------------------------------------------
        # TEAM COMPOSITION
        # ----------------------------------------------------

        comp = match.get(
            "Team_Composition",
            {}
        )

        for role, count in comp.items():

            row[role] = count

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:

        raise ValueError(
            "No valid Win/Loss records found."
        )

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df["Team_Rank"] = pd.to_numeric(
        df["Team_Rank"],
        errors="coerce"
    )

    # Essential information
    df = df.dropna(
        subset=[
            "Year",
            "Opponent",
            "Pitch_Type",
            "HomeAway",
            "Team_Rank"
        ]
    )

    # ========================================================
    # IDENTIFY ROLE COLUMNS
    # ========================================================

    excluded_columns = [
        "Match_No",
        "Year",
        "Opponent",
        "Pitch_Type",
        "HomeAway",
        "Venue",
        "Team_Rank",
        "Rank_Tier",
        "Result"
    ]

    role_cols = [
        col
        for col in df.columns
        if col not in excluded_columns
    ]

    # ========================================================
    # CLEAN ROLE COUNTS
    # ========================================================

    df[role_cols] = (
        df[role_cols]
        .fillna(0)
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    # ========================================================
    # CLEAN CATEGORICAL VALUES
    # ========================================================

    categorical_columns = [
        "Opponent",
        "Pitch_Type",
        "HomeAway",
        "Venue"
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = (
        df
        .sort_values(
            ["Year", "Match_No"],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    return df, role_cols


# ============================================================
# FIT RANK TIERS USING TRAINING DATA ONLY
# ============================================================

def fit_rank_tier_boundaries(train_df):

    q1 = train_df["Team_Rank"].quantile(0.25)
    q2 = train_df["Team_Rank"].quantile(0.50)
    q3 = train_df["Team_Rank"].quantile(0.75)

    qd = (q3 - q1) / 2

    lower_boundary = q2 - qd
    upper_boundary = q2 + qd

    return lower_boundary, upper_boundary


def apply_rank_tier(
    df,
    lower_boundary,
    upper_boundary
):

    df = df.copy()

    def classify_rank(rank):

        if rank <= lower_boundary:

            return "Top"

        elif rank <= upper_boundary:

            return "Mid"

        else:

            return "Low"

    df["Rank_Tier"] = (
        df["Team_Rank"]
        .apply(classify_rank)
    )

    return df


# ============================================================
# CREATE COMMON CHRONOLOGICAL SPLIT
# ============================================================

def create_common_split(match_data):

    full_df, role_cols = prepare_dataset(
        match_data
    )

    print("\n")
    print("=" * 75)
    print("DATASET")
    print("=" * 75)

    print(
        f"Total usable matches: {len(full_df)}"
    )

    print(
        f"Year range: "
        f"{int(full_df['Year'].min())} "
        f"to "
        f"{int(full_df['Year'].max())}"
    )

    print(
        f"Wins: "
        f"{(full_df['Result'] == 'win').sum()}"
    )

    print(
        f"Losses: "
        f"{(full_df['Result'] == 'loss').sum()}"
    )

    # ========================================================
    # EXACT CHRONOLOGICAL SPLIT
    # ========================================================

    train_all = full_df[
        full_df["Year"] <= TRAIN_END_YEAR
    ].copy()

    test_df = full_df[
        (
            full_df["Year"] >= TEST_START_YEAR
        )
        &
        (
            full_df["Year"] <= TEST_END_YEAR
        )
    ].copy()

    # ========================================================
    # FIT RANK TIER ONLY ON TRAINING DATA
    # ========================================================

    lower_boundary, upper_boundary = (
        fit_rank_tier_boundaries(
            train_all
        )
    )

    train_all = apply_rank_tier(
        train_all,
        lower_boundary,
        upper_boundary
    )

    test_df = apply_rank_tier(
        test_df,
        lower_boundary,
        upper_boundary
    )

    # ========================================================
    # MODEL A — WIN ONLY
    # ========================================================

    win_train_df = train_all[
        train_all["Result"] == "win"
    ].copy()

    # ========================================================
    # MODEL B — WIN + LOSS
    # ========================================================

    win_loss_train_df = train_all[
        train_all["Result"].isin(
            ["win", "loss"]
        )
    ].copy()

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    if len(train_all) == 0:

        raise ValueError(
            "No training matches before 2024."
        )

    if len(test_df) == 0:

        raise ValueError(
            "No 2024-2025 test matches found."
        )

    if len(win_train_df) == 0:

        raise ValueError(
            "No historical winning matches "
            "before 2024."
        )

    # ========================================================
    # PRINT SPLIT INFORMATION
    # ========================================================

    print("\n")
    print("=" * 75)
    print("CHRONOLOGICAL EXPERIMENT")
    print("=" * 75)

    print("\nTRAINING PERIOD")
    print(
        f"{TRAIN_END_YEAR} and earlier"
    )

    print(
        f"Matches: {len(train_all)}"
    )

    print("\nWIN-ONLY TRAINING")

    print(
        f"Matches: {len(win_train_df)}"
    )

    print(
        f"Years: "
        f"{int(win_train_df['Year'].min())}"
        f"-"
        f"{int(win_train_df['Year'].max())}"
    )

    print("\nWIN + LOSS TRAINING")

    print(
        f"Matches: {len(win_loss_train_df)}"
    )

    print(
        f"Wins: "
        f"{(win_loss_train_df['Result'] == 'win').sum()}"
    )

    print(
        f"Losses: "
        f"{(win_loss_train_df['Result'] == 'loss').sum()}"
    )

    print(
        f"Years: "
        f"{int(win_loss_train_df['Year'].min())}"
        f"-"
        f"{int(win_loss_train_df['Year'].max())}"
    )

    print("\nCOMMON FUTURE TEST SET")

    print(
        f"Matches: {len(test_df)}"
    )

    print(
        f"Years: "
        f"{int(test_df['Year'].min())}"
        f"-"
        f"{int(test_df['Year'].max())}"
    )

    print(
        "\nRank-tier boundaries learned "
        "from training data only:"
    )

    print(
        f"Top boundary: "
        f"{lower_boundary:.4f}"
    )

    print(
        f"Mid boundary: "
        f"{upper_boundary:.4f}"
    )

    return (
        win_train_df,
        win_loss_train_df,
        test_df,
        role_cols
    )


# ============================================================
# CREATE FEATURES AND TARGETS
# ============================================================

def create_xy(
    train_df,
    test_df,
    role_cols
):

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    # Keep the feature set identical to the original LightGBM experiment.
    # The reviewer comparison must differ only in the training observations
    # (Win-only vs Win+Loss), not in the predictive features.
    categorical_features = [
        "Opponent",
        "Pitch_Type",
        "HomeAway",
        "Rank_Tier"
    ]

    numerical_features = []

    feature_columns = (
        categorical_features
        +
        numerical_features
    )

    X_train_raw = train_df[
        feature_columns
    ].copy()

    X_test_raw = test_df[
        feature_columns
    ].copy()

    y_train = (
        train_df[role_cols]
        .values
        .astype(float)
    )

    y_test = (
        test_df[role_cols]
        .values
        .astype(float)
    )

    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),

            (
                "numeric",
                "passthrough",
                numerical_features
            )
        ],

        remainder="drop"
    )

    X_train = preprocessor.fit_transform(
        X_train_raw
    )

    X_test = preprocessor.transform(
        X_test_raw
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

def train_lightgbm(
    X_train,
    y_train,
    seed
):

    base_model = lgb.LGBMRegressor(

        objective="regression",

        random_state=seed,

        n_estimators=100,

        learning_rate=0.05,

        verbosity=-1
    )

    model = MultiOutputRegressor(
        base_model
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# RAW PREDICTION
# ============================================================

def get_predictions(
    model,
    X_test
):

    predictions = model.predict(
        X_test
    )

    # Role counts cannot be negative
    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


# ============================================================
# CONSTRAIN PREDICTIONS TO A VALID XI
# ============================================================

def reconcile_to_eleven(
    predictions,
    total_players=11
):

    predictions = np.maximum(
        predictions,
        0
    )

    final_predictions = []

    for row in predictions:

        # ----------------------------------------------------
        # Convert continuous predictions to integer counts
        # ----------------------------------------------------

        base = np.floor(row).astype(int)

        fractions = row - base

        difference = (
            total_players
            -
            base.sum()
        )

        # ----------------------------------------------------
        # If total is too small, add players to largest
        # fractional components
        # ----------------------------------------------------

        if difference > 0:

            order = np.argsort(
                -fractions
            )

            for idx in order:

                if difference <= 0:
                    break

                base[idx] += 1
                difference -= 1

        # ----------------------------------------------------
        # If total is too large, remove players from the
        # smallest fractional components, while avoiding
        # negative counts
        # ----------------------------------------------------

        elif difference < 0:

            order = np.argsort(
                fractions
            )

            for idx in order:

                if difference >= 0:
                    break

                if base[idx] > 0:

                    base[idx] -= 1
                    difference += 1

        final_predictions.append(
            base
        )

    return np.array(
        final_predictions,
        dtype=float
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    y_test,
    predictions,
    role_cols,
    model_name,
    seed,
    prediction_type
):

    results = []

    # ========================================================
    # ROLE-WISE
    # ========================================================

    for i, role in enumerate(
        role_cols
    ):

        y_true = y_test[:, i]

        y_pred = predictions[:, i]

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        if len(
            np.unique(y_true)
        ) > 1:

            r2 = r2_score(
                y_true,
                y_pred
            )

        else:

            r2 = np.nan

        results.append({

            "Model":
                model_name,

            "Seed":
                seed,

            "Prediction_Type":
                prediction_type,

            "Role":
                role,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2
        })

    # ========================================================
    # OVERALL
    # ========================================================

    y_true_flat = y_test.flatten()

    y_pred_flat = predictions.flatten()

    overall_mae = mean_absolute_error(
        y_true_flat,
        y_pred_flat
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_true_flat,
            y_pred_flat
        )
    )

    overall_r2 = r2_score(
        y_true_flat,
        y_pred_flat
    )

    results.append({

        "Model":
            model_name,

        "Seed":
            seed,

        "Prediction_Type":
            prediction_type,

        "Role":
            "OVERALL",

        "MAE":
            overall_mae,

        "RMSE":
            overall_rmse,

        "R2":
            overall_r2
    })

    return pd.DataFrame(
        results
    )


# ============================================================
# VALIDITY CHECK FOR TEAM COMPOSITION
# ============================================================

def composition_validity(
    predictions,
    total_players=11
):

    integer_predictions = (
        np.rint(predictions)
        .astype(int)
    )

    correct_total = (
        integer_predictions.sum(axis=1)
        ==
        total_players
    )

    valid_rows = (
        correct_total
        &
        (
            integer_predictions >= 0
        ).all(axis=1)
    )

    validity_percentage = (
        valid_rows.mean()
        *
        100
    )

    return validity_percentage


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # LOAD DATA
    # ========================================================

    match_data = load_match_data()

    print(
        f"Raw JSON records: "
        f"{len(match_data)}"
    )

    # ========================================================
    # COMMON SPLIT
    # ========================================================

    (
        win_train_df,
        win_loss_train_df,
        test_df,
        role_cols
    ) = create_common_split(
        match_data
    )

    # ========================================================
    # PRINT TARGET INFORMATION
    # ========================================================

    print("\n")
    print("=" * 75)
    print("TARGET ROLES")
    print("=" * 75)

    print(role_cols)

    print(
        f"\nNumber of target roles: "
        f"{len(role_cols)}"
    )

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    all_results = []
    paired_comparisons = []

    all_predictions = []

    # ========================================================
    # FIVE RANDOM SEEDS
    # ========================================================

    for seed in RANDOM_SEEDS:

        print("\n")
        print("-" * 75)
        print(
            f"SEED = {seed}"
        )
        print("-" * 75)

        # ====================================================
        # MODEL A — WIN ONLY
        # ====================================================

        (
            X_train_win,
            X_test_win,
            y_train_win,
            y_test_win,
            preprocessor_win
        ) = create_xy(
            win_train_df,
            test_df,
            role_cols
        )

        model_win = train_lightgbm(
            X_train_win,
            y_train_win,
            seed
        )

        predictions_win_raw = (
            get_predictions(
                model_win,
                X_test_win
            )
        )

        predictions_win_valid = (
            reconcile_to_eleven(
                predictions_win_raw,
                TOTAL_PLAYERS
            )
        )

        # ----------------------------------------------------
        # RAW REGRESSION METRICS
        # ----------------------------------------------------

        metrics_win_raw = (
            evaluate_predictions(
                y_test_win,
                predictions_win_raw,
                role_cols,
                "Win-only",
                seed,
                "Raw"
            )
        )

        # ----------------------------------------------------
        # CONSTRAINED METRICS
        # ----------------------------------------------------

        metrics_win_valid = (
            evaluate_predictions(
                y_test_win,
                predictions_win_valid,
                role_cols,
                "Win-only",
                seed,
                "XI-constrained"
            )
        )

        all_results.extend([
            metrics_win_raw,
            metrics_win_valid
        ])

        # ====================================================
        # MODEL B — WIN + LOSS
        # ====================================================

        (
            X_train_all,
            X_test_all,
            y_train_all,
            y_test_all,
            preprocessor_all
        ) = create_xy(
            win_loss_train_df,
            test_df,
            role_cols
        )

        model_all = train_lightgbm(
            X_train_all,
            y_train_all,
            seed
        )

        predictions_all_raw = (
            get_predictions(
                model_all,
                X_test_all
            )
        )

        predictions_all_valid = (
            reconcile_to_eleven(
                predictions_all_raw,
                TOTAL_PLAYERS
            )
        )

        # ----------------------------------------------------
        # RAW REGRESSION METRICS
        # ----------------------------------------------------

        metrics_all_raw = (
            evaluate_predictions(
                y_test_all,
                predictions_all_raw,
                role_cols,
                "Win+Loss",
                seed,
                "Raw"
            )
        )

        # ----------------------------------------------------
        # PAIRED RAW COMPARISON
        # ----------------------------------------------------
        # Both models are evaluated on the exact same match-role observations.
        # The paired test compares their absolute errors directly.
        from scipy.stats import ttest_rel

        win_errors_flat = np.abs(
            y_test_win - predictions_win_raw
        ).reshape(-1)
        all_errors_flat = np.abs(
            y_test_all - predictions_all_raw
        ).reshape(-1)

        paired_test = ttest_rel(
            win_errors_flat,
            all_errors_flat
        )

        paired_comparisons.append({
            "Seed": seed,
            "Win_only_MAE": np.mean(win_errors_flat),
            "Win_Loss_MAE": np.mean(all_errors_flat),
            "MAE_Difference_Win_minus_WinLoss": (
                np.mean(win_errors_flat) - np.mean(all_errors_flat)
            ),
            "Paired_t_statistic": paired_test.statistic,
            "Paired_p_value": paired_test.pvalue,
            "Paired_Observations": len(win_errors_flat)
        })

        # ----------------------------------------------------
        # CONSTRAINED METRICS
        # ----------------------------------------------------

        metrics_all_valid = (
            evaluate_predictions(
                y_test_all,
                predictions_all_valid,
                role_cols,
                "Win+Loss",
                seed,
                "XI-constrained"
            )
        )

        all_results.extend([
            metrics_all_raw,
            metrics_all_valid
        ])

        # ====================================================
        # VALIDITY
        # ====================================================

        win_validity = composition_validity(
            predictions_win_valid,
            TOTAL_PLAYERS
        )

        all_validity = composition_validity(
            predictions_all_valid,
            TOTAL_PLAYERS
        )

        print(
            f"\nWin-only XI validity: "
            f"{win_validity:.2f}%"
        )

        print(
            f"Win+Loss XI validity: "
            f"{all_validity:.2f}%"
        )

        # ====================================================
        # CURRENT SEED OVERALL RESULTS
        # ====================================================

        print("\nWin-only overall — RAW:")

        print(
            metrics_win_raw[
                metrics_win_raw["Role"]
                == "OVERALL"
            ].to_string(
                index=False
            )
        )

        print(
            "\nWin+Loss overall — RAW:"
        )

        print(
            metrics_all_raw[
                metrics_all_raw["Role"]
                == "OVERALL"
            ].to_string(
                index=False
            )
        )

        print(
            "\nWin-only overall — XI constrained:"
        )

        print(
            metrics_win_valid[
                metrics_win_valid["Role"]
                == "OVERALL"
            ].to_string(
                index=False
            )
        )

        print(
            "\nWin+Loss overall — XI constrained:"
        )

        print(
            metrics_all_valid[
                metrics_all_valid["Role"]
                == "OVERALL"
            ].to_string(
                index=False
            )
        )

        # ====================================================
        # SAVE PREDICTIONS FOR THIS SEED
        # ====================================================

        for i in range(
            len(test_df)
        ):

            prediction_row = {

                "Seed":
                    seed,

                "Match_No":
                    test_df.iloc[i]["Match_No"],

                "Year":
                    test_df.iloc[i]["Year"],

                "Result":
                    test_df.iloc[i]["Result"],

                "Model":
                    "Win-only"
            }

            # Raw predictions
            for j, role in enumerate(
                role_cols
            ):

                prediction_row[
                    f"{role}_Raw"
                ] = predictions_win_raw[i, j]

                prediction_row[
                    f"{role}_XI"
                ] = predictions_win_valid[i, j]

                prediction_row[
                    f"{role}_Actual"
                ] = y_test_win[i, j]

            all_predictions.append(
                prediction_row
            )

            prediction_row = {

                "Seed":
                    seed,

                "Match_No":
                    test_df.iloc[i]["Match_No"],

                "Year":
                    test_df.iloc[i]["Year"],

                "Result":
                    test_df.iloc[i]["Result"],

                "Model":
                    "Win+Loss"
            }

            for j, role in enumerate(
                role_cols
            ):

                prediction_row[
                    f"{role}_Raw"
                ] = predictions_all_raw[i, j]

                prediction_row[
                    f"{role}_XI"
                ] = predictions_all_valid[i, j]

                prediction_row[
                    f"{role}_Actual"
                ] = y_test_all[i, j]

            all_predictions.append(
                prediction_row
            )

    # ========================================================
    # COMBINE ALL METRICS
    # ========================================================

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # ========================================================
    # SUMMARY ACROSS SEEDS
    # ========================================================

    summary_df = (
        results_df
        .groupby(
            [
                "Model",
                "Prediction_Type",
                "Role"
            ]
        )
        .agg(

            MAE_mean=(
                "MAE",
                "mean"
            ),

            MAE_SD=(
                "MAE",
                "std"
            ),

            RMSE_mean=(
                "RMSE",
                "mean"
            ),

            RMSE_SD=(
                "RMSE",
                "std"
            ),

            R2_mean=(
                "R2",
                "mean"
            ),

            R2_SD=(
                "R2",
                "std"
            )
        )
        .reset_index()
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    overall = summary_df[
        summary_df["Role"]
        == "OVERALL"
    ].copy()

    # ========================================================
    # PRINT FINAL COMPARISON
    # ========================================================

    print("\n\n")
    print("=" * 75)
    print(
        "FINAL WIN-ONLY VS WIN+LOSS COMPARISON"
    )
    print("=" * 75)

    print(
        overall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # PAIRED STATISTICAL COMPARISON
    # ========================================================

    paired_df = pd.DataFrame(paired_comparisons)

    print("\n\n")
    print("=" * 75)
    print("PAIRED WIN-ONLY VS WIN+LOSS RAW COMPARISON")
    print("=" * 75)
    print(
        paired_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ========================================================
    # ROLE-WISE SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 75)
    print("ROLE-WISE SUMMARY")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame(
        all_predictions
    )

    predictions_df.to_csv(
        "LightGBM_win_vs_winloss_predictions.csv",
        index=False
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    results_df.to_csv(
        "LightGBM_win_vs_winloss_detailed.csv",
        index=False
    )

    summary_df.to_csv(
        "LightGBM_win_vs_winloss_summary.csv",
        index=False
    )

    overall.to_csv(
        "LightGBM_win_vs_winloss_overall.csv",
        index=False
    )

    paired_df.to_csv(
        "LightGBM_win_vs_winloss_paired_comparison.csv",
        index=False
    )

    print("\n")
    print("=" * 75)
    print("RESULTS SAVED SUCCESSFULLY")
    print("=" * 75)

    print(
        "\nFiles created:"
    )

    print(
        "1. LightGBM_win_vs_winloss_predictions.csv"
    )

    print(
        "2. LightGBM_win_vs_winloss_detailed.csv"
    )

    print(
        "3. LightGBM_win_vs_winloss_summary.csv"
    )

    print(
        "4. LightGBM_win_vs_winloss_overall.csv"
    )