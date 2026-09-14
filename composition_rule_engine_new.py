import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


def load_match_data():
    with open("team_composition_updated_new.json") as f:
        return json.load(f)


def prepare_dataset(match_data):
    rows = []

    for match in match_data:
        row = {
            "Match_No": match.get("Match_No"),
            "Opponent": match.get("Opponent"),
            "Pitch_Type": match.get("Pitch_Type"),
            "HomeAway": match.get("HomeAway"),
            "Venue": match.get("Venue"),
            "Team_Rank": match.get("Team_Rank", 10)
        }

        comp = match.get("Team_Composition", {})

        for role, count in comp.items():
            row[role] = count

        rows.append(row)

    df = pd.DataFrame(rows)

    df["Team_Rank"] = pd.to_numeric(
        df["Team_Rank"],
        errors="coerce"
    )

    Q1 = df["Team_Rank"].quantile(0.25)
    Q2 = df["Team_Rank"].quantile(0.50)
    Q3 = df["Team_Rank"].quantile(0.75)

    QD = (Q3 - Q1) / 2

    lower_boundary = Q2 - QD
    upper_boundary = Q2 + QD

    def classify_rank(rank):
        if pd.isna(rank):
            return np.nan

        if rank <= lower_boundary:
            return "Top"

        elif rank <= upper_boundary:
            return "Mid"

        else:
            return "Low"

    df["Rank_Tier"] = df["Team_Rank"].apply(
        classify_rank
    )

    role_cols = [
        col
        for col in df.columns
        if col not in [
            "Match_No",
            "Opponent",
            "Pitch_Type",
            "HomeAway",
            "Venue",
            "Team_Rank",
            "Rank_Tier"
        ]
    ]

    df[role_cols] = (
        df[role_cols]
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(
        subset=[
            "Opponent",
            "Pitch_Type",
            "HomeAway",
            "Rank_Tier"
        ]
    )

    return df, role_cols


def train_ml_model(df, role_cols):
    le_opponent = LabelEncoder()
    le_pitch = LabelEncoder()
    le_homeaway = LabelEncoder()
    le_rank = LabelEncoder()

    X_cat = pd.DataFrame({
        "Opponent": le_opponent.fit_transform(
            df["Opponent"]
        ),
        "Pitch_Type": le_pitch.fit_transform(
            df["Pitch_Type"]
        ),
        "HomeAway": le_homeaway.fit_transform(
            df["HomeAway"]
        ),
        "Rank_Tier": le_rank.fit_transform(
            df["Rank_Tier"]
        )
    })

    y = df[role_cols]

    model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            random_state=42
        )
    )

    model.fit(X_cat, y)

    return model, (
        le_opponent,
        le_pitch,
        le_homeaway,
        le_rank
    )


def get_predicted_role_counts(
    pitch,
    homeaway,
    opponent=None
):
    match_data = load_match_data()

    df, role_cols = prepare_dataset(
        match_data
    )

    opponent_rank = (
        df.loc[
            df["Opponent"] == opponent,
            "Team_Rank"
        ]
    )

    if opponent is not None and not opponent_rank.empty:
        selected_rank = opponent_rank.iloc[0]

        Q1 = df["Team_Rank"].quantile(0.25)
        Q2 = df["Team_Rank"].quantile(0.50)
        Q3 = df["Team_Rank"].quantile(0.75)

        QD = (Q3 - Q1) / 2

        lower_boundary = Q2 - QD
        upper_boundary = Q2 + QD

        if selected_rank <= lower_boundary:
            rank_tier = "Top"

        elif selected_rank <= upper_boundary:
            rank_tier = "Mid"

        else:
            rank_tier = "Low"

    else:
        rank_tier = None

    composition_map = {}

    for _, row in df.iterrows():
        key = (
            row["Pitch_Type"],
            row["HomeAway"],
            row["Rank_Tier"],
            row["Opponent"]
        )

        composition_map[key] = {
            role: row[role]
            for role in role_cols
        }

    input_context = (
        pitch,
        homeaway,
        rank_tier,
        opponent
    )

    def context_similarity(ctx1, ctx2):
        if ctx1[3] is None:
            return sum(
                1
                for a, b in zip(
                    ctx1[:3],
                    ctx2[:3]
                )
                if a == b
            )

        return sum(
            1
            for a, b in zip(
                ctx1,
                ctx2
            )
            if a == b
        )

    best_match = None
    best_score = -1

    for ctx, comp in composition_map.items():
        score = context_similarity(
            input_context,
            ctx
        )

        if score > best_score:
            best_score = score
            best_match = (
                ctx,
                comp
            )

    if (
        best_match is not None
        and best_score >= 2
    ):
        return best_match[1]

    model, encoders = train_ml_model(
        df,
        role_cols
    )

    (
        le_opponent,
        le_pitch,
        le_homeaway,
        le_rank
    ) = encoders

    most_common_opponent = (
        df["Opponent"].mode()[0]
    )

    opponent_for_ml = opponent

    if (
        opponent_for_ml is None
        or opponent_for_ml not in le_opponent.classes_
    ):
        opponent_for_ml = most_common_opponent

    if pitch not in le_pitch.classes_:
        raise ValueError(
            f"Unknown Pitch_Type: {pitch}"
        )

    if homeaway not in le_homeaway.classes_:
        raise ValueError(
            f"Unknown HomeAway value: {homeaway}"
        )

    if rank_tier is None:
        raise ValueError(
            "Unable to determine Rank Tier for the selected opponent."
        )

    if rank_tier not in le_rank.classes_:
        raise ValueError(
            f"Unknown Rank_Tier: {rank_tier}"
        )

    x_input = pd.DataFrame([{
        "Opponent": le_opponent.transform(
            [opponent_for_ml]
        )[0],
        "Pitch_Type": le_pitch.transform(
            [pitch]
        )[0],
        "HomeAway": le_homeaway.transform(
            [homeaway]
        )[0],
        "Rank_Tier": le_rank.transform(
            [rank_tier]
        )[0]
    }])

    y_pred = model.predict(
        x_input
    )[0]

    role_counts = {
        role: max(
            0,
            int(round(count))
        )
        for role, count in zip(
            role_cols,
            y_pred
        )
    }

    return role_counts


if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(
        page_title="Team Composition Predictor",
        page_icon="🏏",
        layout="centered"
    )

    st.title(
        "🏏 Team Composition Predictor"
    )

    match_data = load_match_data()

    df, role_cols = prepare_dataset(
        match_data
    )

    all_opponents = sorted(
        df["Opponent"]
        .dropna()
        .unique()
    )

    pitch_options = sorted(
        df["Pitch_Type"]
        .dropna()
        .unique()
    )

    homeaway_options = sorted(
        df["HomeAway"]
        .dropna()
        .unique()
    )

    st.subheader(
        "🎯 Select Match Context"
    )

    pitch = st.selectbox(
        "Pitch Type",
        pitch_options
    )

    homeaway = st.selectbox(
        "Home/Away",
        homeaway_options
    )

    opponent = st.selectbox(
        "Opponent",
        all_opponents
    )

    if st.button(
        "Submit",
        type="primary",
        use_container_width=True
    ):
        try:
            opponent_rank = df.loc[
                df["Opponent"] == opponent,
                "Team_Rank"
            ].iloc[0]

            Q1 = df["Team_Rank"].quantile(0.25)
            Q2 = df["Team_Rank"].quantile(0.50)
            Q3 = df["Team_Rank"].quantile(0.75)

            QD = (Q3 - Q1) / 2

            lower_boundary = Q2 - QD
            upper_boundary = Q2 + QD

            if opponent_rank <= lower_boundary:
                rank_tier = "Top"

            elif opponent_rank <= upper_boundary:
                rank_tier = "Mid"

            else:
                rank_tier = "Low"

            st.info(
                f"Opponent: {opponent}  |  "
                f"Rank: {int(opponent_rank)}  |  "
                f"Tier: {rank_tier}"
            )

            with st.spinner(
                "Generating team composition..."
            ):
                predicted_counts = (
                    get_predicted_role_counts(
                        pitch=pitch,
                        homeaway=homeaway,
                        opponent=opponent
                    )
                )

            st.success(
                "Team composition generated successfully."
            )

            composition_df = pd.DataFrame({
                "Role": list(
                    predicted_counts.keys()
                ),
                "Predicted Count": list(
                    predicted_counts.values()
                )
            })

            st.subheader(
                "🧮 Predicted Team Composition"
            )

            st.dataframe(
                composition_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(
                f"Unable to generate team composition: {e}"
            )
