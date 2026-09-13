import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import t

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from context_sidebar_manual_selection import get_match_context
from statistical_score_calc import (
    run_statistical_score_calc,
    get_feature_target_from_final,
    run_statistical_bowling_score_calc,
    get_bowling_feature_target,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architectures are deliberately limited to compact MLPs appropriate for
# the existing model. The current model is 1x64.
ARCHITECTURES = {
    "1 × 32": [32],
    "1 × 64": [64],
    "1 × 128": [128],
    "2 × 32-16": [32, 16],
    "2 × 64-32": [64, 32],
    "2 × 128-64": [128, 64],
}

DEFAULT_SEEDS = [42, 123, 2024, 7, 99]

# ---------------------------------------------------------------------
# Training / evaluation protocol
# ---------------------------------------------------------------------
# These values document the existing experimental setup explicitly.
# They are kept separate from the architecture definition so the
# experimental protocol is reproducible without changing the model logic.
DEVELOPMENT_FRACTION = 0.85
TEST_FRACTION = 0.15
SPLIT_RANDOM_STATE = 42

# The final 15% test set is kept completely untouched during architecture
# selection. Five-fold cross-validation is performed only on the remaining
# 85% development data.
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# The existing trainer performs full-batch gradient descent: all training
# observations are used in one optimizer step per epoch.
BATCH_SIZE = None  # None = full-batch training

# Existing optimization settings; deliberately unchanged.
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 0.001
OPTIMIZER_NAME = "Adam"
LOSS_FUNCTION_NAME = "Mean Squared Error (MSE)"

# No explicit weight decay, dropout, L1/L2 penalty, or other regularization
# is used in the intact model.
REGULARIZATION = "None"

# No early stopping is used. Training stops after the fixed number of epochs.
STOPPING_CRITERION = "Fixed number of epochs (100); no early stopping"

# Architecture selection is based only on cross-validation performance on the
# 85% development data. The 15% test set is not used for selection.
SELECTION_PRIMARY_METRIC = "5-fold CV Validation R2 (mean across seeds)"
SELECTION_SECONDARY_METRIC = "5-fold CV Validation RMSE (mean across seeds)"
CROSS_VALIDATION_PROCEDURE = (
    "5-fold cross-validation on the 85% development data, with a separate "
    "15% final test set kept untouched until architecture selection is complete."
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iyengar_sudarsan_weights(X_np):
    """Exactly the weighting calculation used in Manual_Selection.py."""
    means = np.mean(X_np, axis=0)
    stds = np.std(X_np, axis=0)
    epsilon = 1e-8
    return means / (stds + epsilon)


class ContextualMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, iyengar_w):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for layer_index, hidden_dim in enumerate(hidden_layers):
            layer = nn.Linear(previous_dim, hidden_dim)

            # Preserve the existing Iyengar/Sudarsan initialization for
            # the first hidden layer. Additional hidden layers use Xavier.
            if layer_index == 0:
                weight_matrix = torch.tensor(
                    iyengar_w,
                    dtype=torch.float32
                ).reshape(1, -1).repeat(hidden_dim, 1)

                with torch.no_grad():
                    layer.weight.copy_(weight_matrix)
                    layer.bias.zero_()
            else:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

            layers.append(layer)
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        output = nn.Linear(previous_dim, 1)

        # Preserve the existing output initialization.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def metric_values(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, predictions),
    }


def train_model(X_train, y_train, input_dim, hidden_layers, iyengar_w,
                seed, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LEARNING_RATE):
    """Train one MLP using the intact optimization setup."""
    set_seed(seed)

    model = ContextualMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        iyengar_w=iyengar_w,
    ).to(DEVICE)

    X_train_t = torch.tensor(
        X_train, dtype=torch.float32, device=DEVICE
    )
    y_train_t = torch.tensor(
        y_train, dtype=torch.float32, device=DEVICE
    ).view(-1, 1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)

        loss.backward()
        optimizer.step()

    return model


def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.tensor(X, dtype=torch.float32, device=DEVICE)
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )


def cross_validate_and_evaluate(
    X_np,
    y_np,
    iyengar_w,
    hidden_layers,
    seed,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LEARNING_RATE,
):
    """
    Perform 5-fold CV on the 85% development data and evaluate the resulting
    fold models on their held-out validation folds.

    The separate 15% final test set is deliberately excluded here.
    """
    set_seed(seed)

    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)

    # First create the untouched final 15% test set.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_np,
        y_np,
        test_size=TEST_FRACTION,
        random_state=SPLIT_RANDOM_STATE,
        shuffle=True,
    )

    kfold = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    fold_train_metrics = []
    fold_val_metrics = []

    for fold_number, (train_idx, val_idx) in enumerate(
        kfold.split(X_dev), start=1
    ):
        X_train = X_dev[train_idx]
        y_train = y_dev[train_idx]
        X_val = X_dev[val_idx]
        y_val = y_dev[val_idx]

        model = train_model(
            X_train=X_train,
            y_train=y_train,
            input_dim=X_np.shape[1],
            hidden_layers=hidden_layers,
            iyengar_w=iyengar_w,
            seed=seed,
            epochs=epochs,
            lr=lr,
        )

        train_pred = predict_model(model, X_train)
        val_pred = predict_model(model, X_val)

        train_metrics = metric_values(y_train, train_pred)
        val_metrics = metric_values(y_val, val_pred)

        fold_train_metrics.append(train_metrics)
        fold_val_metrics.append(val_metrics)

    train_df = pd.DataFrame(fold_train_metrics)
    val_df = pd.DataFrame(fold_val_metrics)

    # Return the mean over the five folds for this independent seed.
    return {
        "Train_MAE": train_df["MAE"].mean(),
        "Train_RMSE": train_df["RMSE"].mean(),
        "Train_R2": train_df["R2"].mean(),
        "Validation_MAE": val_df["MAE"].mean(),
        "Validation_RMSE": val_df["RMSE"].mean(),
        "Validation_R2": val_df["R2"].mean(),
    }


def final_test_evaluation(
    X_np,
    y_np,
    iyengar_w,
    hidden_layers,
    seed,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LEARNING_RATE,
):
    """
    After architecture selection, retrain the selected architecture on the
    complete 85% development data and evaluate once on the untouched 15% test
    set for this independent seed.
    """
    set_seed(seed)

    X_np = np.asarray(X_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X_np,
        y_np,
        test_size=TEST_FRACTION,
        random_state=SPLIT_RANDOM_STATE,
        shuffle=True,
    )

    model = train_model(
        X_train=X_dev,
        y_train=y_dev,
        input_dim=X_np.shape[1],
        hidden_layers=hidden_layers,
        iyengar_w=iyengar_w,
        seed=seed,
        epochs=epochs,
        lr=lr,
    )

    test_pred = predict_model(model, X_test)
    test_metrics = metric_values(y_test, test_pred)

    return {
        "Test_MAE": test_metrics["MAE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_R2": test_metrics["R2"],
    }



def check_data_size(X_np, dataset_name):
    n = len(X_np)

    # 15% is reserved for the final test set and the remaining 85% must
    # support five CV folds.
    if n < 20:
        raise ValueError(
            f"{dataset_name} has only {n} rows. "
            "The 85% development / 15% test split is too small for "
            "a meaningful 5-fold cross-validation experiment."
        )


def run_experiment(X_np, y_np, weights, dataset_name, seeds):
    """
    Architecture sensitivity experiment.

    Stage 1:
      - Reserve 15% as an untouched final test set.
      - Perform 5-fold CV only on the remaining 85%.
      - Compare architectures using CV validation performance.

    Stage 2:
      - Select the best architecture using CV validation R2/RMSE.
      - Retrain only that selected architecture on the complete 85%
        development data.
      - Evaluate it on the untouched 15% test set across the independent seeds.
    """
    check_data_size(X_np, dataset_name)

    cv_records = []

    for architecture_name, hidden_layers in ARCHITECTURES.items():
        for seed in seeds:
            metrics = cross_validate_and_evaluate(
                X_np=X_np,
                y_np=y_np,
                iyengar_w=weights,
                hidden_layers=hidden_layers,
                seed=seed,
            )

            cv_records.append({
                "Dataset": dataset_name,
                "Architecture": architecture_name,
                "Hidden Layers": len(hidden_layers),
                "Neurons": "-".join(map(str, hidden_layers)),
                "Seed": seed,
                **metrics,
            })

    cv_raw = pd.DataFrame(cv_records)

    metric_columns = [
        "Train_MAE",
        "Train_RMSE",
        "Train_R2",
        "Validation_MAE",
        "Validation_RMSE",
        "Validation_R2",
    ]

    # Aggregate the five independent seeds. Each seed value is itself the
    # mean across the five CV folds.
    summary = (
        cv_raw.groupby(
            ["Dataset", "Architecture", "Hidden Layers", "Neurons"],
            as_index=False
        )[metric_columns]
        .agg(["mean", "std", "count"])
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # 95% CIs across the five independent seeds.
    for metric in metric_columns:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        ci_lower_col = f"{metric}_CI95_Lower"
        ci_upper_col = f"{metric}_CI95_Upper"

        summary[ci_lower_col] = np.nan
        summary[ci_upper_col] = np.nan

        valid = summary[count_col] > 1
        degrees_of_freedom = summary.loc[valid, count_col] - 1
        critical_value = t.ppf(0.975, degrees_of_freedom)
        standard_error = (
            summary.loc[valid, std_col] /
            np.sqrt(summary.loc[valid, count_col])
        )
        margin = critical_value * standard_error

        summary.loc[valid, ci_lower_col] = (
            summary.loc[valid, mean_col] - margin
        )
        summary.loc[valid, ci_upper_col] = (
            summary.loc[valid, mean_col] + margin
        )

    # Primary selection: highest mean 5-fold CV validation R2.
    # Secondary selection: lowest mean 5-fold CV validation RMSE.
    summary = summary.sort_values(
        by=["Validation_R2_mean", "Validation_RMSE_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # The first row is the architecture selected WITHOUT looking at the
    # final test set.
    selected_architecture = summary.iloc[0]["Architecture"]
    selected_hidden_layers = ARCHITECTURES[selected_architecture]

    # Final test evaluation is performed only after architecture selection.
    test_records = []
    for seed in seeds:
        test_metrics = final_test_evaluation(
            X_np=X_np,
            y_np=y_np,
            iyengar_w=weights,
            hidden_layers=selected_hidden_layers,
            seed=seed,
        )

        test_records.append({
            "Dataset": dataset_name,
            "Architecture": selected_architecture,
            "Hidden Layers": len(selected_hidden_layers),
            "Neurons": "-".join(map(str, selected_hidden_layers)),
            "Seed": seed,
            **test_metrics,
        })

    test_raw = pd.DataFrame(test_records)

    test_metric_columns = [
        "Test_MAE",
        "Test_RMSE",
        "Test_R2",
    ]

    test_summary = (
        test_raw.groupby(
            ["Dataset", "Architecture", "Hidden Layers", "Neurons"],
            as_index=False
        )[test_metric_columns]
        .agg(["mean", "std", "count"])
    )

    test_summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in test_summary.columns
    ]

    for metric in test_metric_columns:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        ci_lower_col = f"{metric}_CI95_Lower"
        ci_upper_col = f"{metric}_CI95_Upper"

        test_summary[ci_lower_col] = np.nan
        test_summary[ci_upper_col] = np.nan

        valid = test_summary[count_col] > 1
        degrees_of_freedom = test_summary.loc[valid, count_col] - 1
        critical_value = t.ppf(0.975, degrees_of_freedom)
        standard_error = (
            test_summary.loc[valid, std_col] /
            np.sqrt(test_summary.loc[valid, count_col])
        )
        margin = critical_value * standard_error

        test_summary.loc[valid, ci_lower_col] = (
            test_summary.loc[valid, mean_col] - margin
        )
        test_summary.loc[valid, ci_upper_col] = (
            test_summary.loc[valid, mean_col] + margin
        )

    # Attach final test results only to the selected architecture row.
    for col in test_summary.columns:
        if col.startswith("Test_"):
            summary[col] = np.nan

    selected_mask = summary["Architecture"] == selected_architecture
    for col in test_summary.columns:
        if col.startswith("Test_"):
            summary.loc[selected_mask, col] = test_summary.iloc[0][col]

    return cv_raw, summary, test_raw, selected_architecture



def main():
    st.set_page_config(
        page_title="MLP Architecture Sensitivity",
        page_icon="🏏",
        layout="wide",
    )

    st.title("MLP Architecture Sensitivity Analysis")

    st.info(
        "This is a separate experiment. It does not modify Manual_Selection.py "
        "or mlp_trainer.py. It uses the same statistical pipeline, features, "
        "Iyengar/Sudarsan weights, 85/15 development/test split with 5-fold "
        "cross-validation on the development data, full-batch Adam optimizer, "
        "learning rate 0.001, MSE loss, no explicit regularization and 100 epochs."
    )

    players_path = os.path.join(BASE_DIR, "Players.csv")
    teams_path = os.path.join(BASE_DIR, "TeamID.csv")

    players_df = pd.read_csv(players_path)
    teams_df = pd.read_csv(teams_path)

    # This is the SAME match-context interface used by Manual_Selection.py.
    match_context = get_match_context(players_df, teams_df)

    st.write(
        f"**Device:** `{DEVICE}`  |  "
        f"**Seeds:** `{DEFAULT_SEEDS}`"
    )

    with st.expander("Training and validation protocol"):
        st.markdown(
            f"""
            - **Data split:** 15% is reserved as an untouched final test set.
              The remaining 85% is used as development data.
            - **Cross-validation:** {CV_FOLDS}-fold cross-validation on the 85%
              development data, with `shuffle=True` and
              `random_state={CV_RANDOM_STATE}`. The final 15% test set is not
              used during architecture selection.
            - **Batch size:** Full-batch (all training observations per optimizer step).
            - **Optimizer:** {OPTIMIZER_NAME}.
            - **Learning rate:** `{DEFAULT_LEARNING_RATE}`.
            - **Loss:** {LOSS_FUNCTION_NAME}.
            - **Regularization:** {REGULARIZATION}; no dropout or weight decay is added.
            - **Stopping criterion:** {STOPPING_CRITERION}.
            - **Independent runs:** `{len(DEFAULT_SEEDS)}` seeds: `{DEFAULT_SEEDS}`.
            - **Architecture selection:** primary = {SELECTION_PRIMARY_METRIC};
              secondary = {SELECTION_SECONDARY_METRIC}. Test performance is used
              only for confirmation after selection.
            """
        )

    if not st.button("Run Architecture Sensitivity Analysis", type="primary"):
        st.warning(
            "Select the match context above, then click the button to start."
        )
        return

    # ---------------------------------------------------------------
    # BATTTING: exact production feature pipeline
    # ---------------------------------------------------------------
    with st.spinner("Generating the existing batting statistical features..."):
        final_df, final_weights, used_factors = run_statistical_score_calc(
            match_context
        )

    batting_raw_results = []
    batting_test_results = []
    batting_selected_architectures = []
    batting_summaries = []

    for pos in range(1, 8):
        pos_df = final_df[final_df["Position"] == pos].copy()

        if pos_df.empty:
            st.warning(f"Position {pos}: no rows available; skipped.")
            continue

        X_pos, y_pos, _ = get_feature_target_from_final(
            pos_df,
            used_factors
        )

        X_pos_np = X_pos.to_numpy(dtype=np.float32)
        y_pos_np = y_pos.to_numpy(dtype=np.float32).reshape(-1)

        try:
            iw_pos = iyengar_sudarsan_weights(X_pos_np)

            raw, summary, test_raw, selected_architecture = run_experiment(
                X_pos_np,
                y_pos_np,
                iw_pos,
                f"Batting Position {pos}",
                DEFAULT_SEEDS,
            )

            batting_raw_results.append(raw)
            batting_test_results.append(test_raw)
            batting_selected_architectures.append(
                (f"Batting Position {pos}", selected_architecture)
            )
            batting_summaries.append(summary)

        except ValueError as exc:
            st.warning(f"Batting Position {pos}: {exc}")

    # ---------------------------------------------------------------
    # BOWLING: exact production feature pipeline
    # ---------------------------------------------------------------
    with st.spinner("Generating the existing bowling statistical features..."):
        bowl_df, _, bowl_factors = run_statistical_bowling_score_calc(
            match_context
        )

    X_bowl, y_bowl, _ = get_bowling_feature_target(
        bowl_df,
        bowl_factors
    )

    X_bowl_np = X_bowl.to_numpy(dtype=np.float32)
    y_bowl_np = y_bowl.to_numpy(dtype=np.float32).reshape(-1)

    try:
        iw_bowl = iyengar_sudarsan_weights(X_bowl_np)

        bowl_raw, bowl_summary, bowl_test_raw, bowl_selected_architecture = run_experiment(
            X_bowl_np,
            y_bowl_np,
            iw_bowl,
            "Bowling",
            DEFAULT_SEEDS,
        )

    except ValueError as exc:
        bowl_raw = pd.DataFrame()
        bowl_test_raw = pd.DataFrame()
        bowl_selected_architecture = None
        bowl_summary = pd.DataFrame()
        st.warning(f"Bowling: {exc}")

    # ---------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------
    st.success("Architecture sensitivity analysis completed.")

    if batting_summaries:
        all_batting_summary = pd.concat(
            batting_summaries,
            ignore_index=True
        )

        st.header("Batting Architecture Results")
        st.dataframe(
            all_batting_summary.style.format(
                {
                    col: "{:.6f}"
                    for col in all_batting_summary.columns
                    if col.endswith("_mean") or col.endswith("_std") or "CI95_" in col
                }
            ),
            use_container_width=True,
        )

        st.subheader("Best Architecture by Batting Position (selected by 5-fold CV)")

        best_batting = (
            all_batting_summary
            .sort_values(
                ["Dataset", "Validation_R2_mean", "Validation_RMSE_mean"],
                ascending=[True, False, True],
            )
            .groupby("Dataset", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

        st.dataframe(
            best_batting[
                [
                    "Dataset",
                    "Architecture",
                    "Validation_R2_mean",
                    "Validation_R2_std",
                    "Validation_RMSE_mean",
                    "Test_R2_mean",
                    "Test_R2_std",
                    "Test_RMSE_mean",
                ]
            ].style.format(
                {
                    "Validation_R2_mean": "{:.6f}",
                    "Validation_R2_std": "{:.6f}",
                    "Validation_RMSE_mean": "{:.6f}",
                    "Test_R2_mean": "{:.6f}",
                    "Test_R2_std": "{:.6f}",
                    "Test_RMSE_mean": "{:.6f}",
                }
            ),
            use_container_width=True,
        )

    if not bowl_summary.empty:
        st.header("Bowling Architecture Results")

        st.dataframe(
            bowl_summary.style.format(
                {
                    col: "{:.6f}"
                    for col in bowl_summary.columns
                    if col.endswith("_mean") or col.endswith("_std") or "CI95_" in col
                }
            ),
            use_container_width=True,
        )

        st.subheader("Best Architecture for Bowling (selected by 5-fold CV)")

        st.dataframe(
            bowl_summary.head(1).style.format(
                {
                    col: "{:.6f}"
                    for col in bowl_summary.columns
                    if col.endswith("_mean") or col.endswith("_std") or "CI95_" in col
                }
            ),
            use_container_width=True,
        )

    # Final test results for the architecture selected using CV.
    if batting_test_results:
        st.header("Final Test Results — Selected Batting Architectures")
        st.dataframe(
            pd.concat(batting_test_results, ignore_index=True),
            use_container_width=True,
        )

    if not bowl_test_raw.empty:
        st.header("Final Test Results — Selected Bowling Architecture")
        st.dataframe(
            bowl_test_raw,
            use_container_width=True,
        )

    # Optional raw run-level CV results for reproducibility.
    with st.expander("Show individual seed results"):
        if batting_raw_results:
            st.subheader("Batting — individual runs")
            st.dataframe(
                pd.concat(batting_raw_results, ignore_index=True),
                use_container_width=True,
            )

        if not bowl_raw.empty:
            st.subheader("Bowling — individual runs")
            st.dataframe(
                bowl_raw,
                use_container_width=True,
            )

    st.caption(
        "Architecture selection is based only on 5-fold cross-validation "
        "performance on the 85% development data. The 15% final test set "
        "remains untouched until the architecture is fixed and is then used "
        "for final confirmation across the independent seeds."
    )


if __name__ == "__main__":
    main()
