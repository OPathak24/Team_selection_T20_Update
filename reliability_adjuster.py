import pandas as pd
import streamlit as st
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def display_professional_table(df, number_formats=None, height=None):
    """Render a professional spreadsheet-style HTML table in Streamlit."""
    import html

    if df is None or df.empty:
        st.info("No data available.")
        return

    display_df = df.copy().reset_index(drop=True)
    number_formats = number_formats or {}

    # Apply the requested numeric formatting before converting to HTML.
    for column, fmt in number_formats.items():
        if column not in display_df.columns:
            continue

        def format_value(value, fmt=fmt):
            if pd.isna(value):
                return "-"
            try:
                return fmt.format(value)
            except (ValueError, TypeError):
                return str(value)

        display_df[column] = display_df[column].map(format_value)

    # Escape all values so that table content is always rendered as text.
    columns = [html.escape(str(col)) for col in display_df.columns]

    rows_html = []
    for _, row in display_df.iterrows():
        cells = []
        for value in row:
            cells.append(
                f'<td>{html.escape(str(value))}</td>'
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="professional-table-wrapper">
        <div class="professional-table-scroll">
            <table class="professional-table">
                <thead>
                    <tr>
                        {''.join(f'<th>{col}</th>' for col in columns)}
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)


# Professional spreadsheet-style table design.
st.markdown(
    """
    <style>
    /* ============================================================
       Professional transparent / glass spreadsheet tables
       Applied to every table rendered by display_professional_table()
       ============================================================ */

    .professional-table-wrapper {
        width: 100%;
        margin: 12px 0 24px 0;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.42);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }

    .professional-table-scroll {
        width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        scrollbar-width: thin;
    }

    .professional-table {
        width: 100%;
        min-width: 720px;
        border-collapse: separate;
        border-spacing: 0;
        table-layout: auto;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        font-size: 14px;
        color: #1f2937;
        background: transparent;
    }

    .professional-table thead th {
        background: rgba(255, 255, 255, 0.24);
        color: #172033;
        font-weight: 700;
        text-align: left;
        padding: 14px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.48);
        border-right: 1px solid rgba(255, 255, 255, 0.32);
        white-space: nowrap;
        letter-spacing: 0.01em;
    }

    .professional-table thead th:last-child {
        border-right: none;
    }

    .professional-table tbody td {
        padding: 13px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.28);
        border-right: 1px solid rgba(255, 255, 255, 0.22);
        white-space: nowrap;
        vertical-align: middle;
        background: rgba(255, 255, 255, 0.07);
    }

    .professional-table tbody td:last-child {
        border-right: none;
    }

    .professional-table tbody tr:nth-child(even) td {
        background: rgba(255, 255, 255, 0.12);
    }

    .professional-table tbody tr:hover td {
        background: rgba(255, 255, 255, 0.18);
    }

    .professional-table tbody tr:last-child td {
        border-bottom: none;
    }

    .professional-table tbody td:first-child {
        font-weight: 550;
    }

    .professional-table td:nth-child(n+3),
    .professional-table th:nth-child(n+3) {
        text-align: right;
    }

    /* Keep the first two columns readable and left aligned. */
    .professional-table td:nth-child(1),
    .professional-table td:nth-child(2),
    .professional-table th:nth-child(1),
    .professional-table th:nth-child(2) {
        text-align: left;
    }

    /* Streamlit tabs: restrained, professional appearance. */
    div[data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.40);
    }

    button[data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        padding: 10px 16px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #163d68;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load fielding scores CSV
fielding_df = pd.read_csv("Fielding_Scores.csv")
fielding_score_map = fielding_df.set_index("Player Name")["Predicted_Fielding_Score"].to_dict()

def select_most_reliable_batters(final_df, roles_df, innings_df):
    positions = list(range(1, 8))
    max_inns = innings_df['Inns'].max()
    innings_summary = innings_df.groupby(["Player Name", "Position"])["Inns"].sum().reset_index()
    total_inns_per_player = innings_df.groupby("Player Name")['Inns'].sum().to_dict()

    position_rankings = {}
    player_position_scores = defaultdict(dict)

    for pos in positions:
        pos_df = final_df[final_df["Position"] == pos].copy()
        valid_roles = ["Batsman", "WK-Batsman", "Batting Allrounder"]
        pos_df = pos_df[pos_df["Role"].isin(valid_roles)]

        pos_inns_map = innings_summary[innings_summary["Position"] == pos].set_index("Player Name")["Inns"].to_dict()
        std_score = pos_df["Predicted_Score"].std()
        std_inns = pd.Series(pos_inns_map).std()
        total_var = std_score + std_inns + 1e-8
        alpha = std_score / total_var
        beta = std_inns / total_var

        pos_df["Inns_at_Pos"] = pos_df["Player Name"].map(pos_inns_map).fillna(0)
        pos_df["Max_Inns_Pos"] = max(pos_inns_map.values()) if pos_inns_map else 1
        pos_df["Normalized_Inns"] = pos_df["Inns_at_Pos"] / pos_df["Max_Inns_Pos"]
        pos_df["Effective_Score"] = pos_df["Predicted_Score"] * (alpha + beta * pos_df["Normalized_Inns"])
        pos_df["Fielding_Score"] = pos_df["Player Name"].map(fielding_score_map).fillna(0)

        pos_df = pos_df.sort_values(by=["Effective_Score", "Fielding_Score"], ascending=False)
        pos_df = pos_df.drop_duplicates("Player Name")

        position_rankings[pos] = []
        for rank_idx, (_, player) in enumerate(pos_df.iterrows(), start=1):
            player_name = player["Player Name"]
            score = player["Effective_Score"]
            player_data = player.to_dict()
            player_data["Rank"] = rank_idx  # 

            position_rankings[pos].append((player_name, score, player_data))
            player_position_scores[player_name][pos] = {
                'score': score,
                'fielding_score': player_data["Fielding_Score"],
                'data': player_data
            }

    class OptimalAssignment:
        def __init__(self):
            self.assignments = {}
            self.player_positions = {}

        def assign(self, position, player_name):
            if player_name in self.player_positions:
                old_pos = self.player_positions[player_name]
                if old_pos in self.assignments:
                    del self.assignments[old_pos]
            if position in self.assignments:
                old_player = self.assignments[position]
                if old_player in self.player_positions:
                    del self.player_positions[old_player]
            self.assignments[position] = player_name
            self.player_positions[player_name] = position

        def is_player_assigned(self, player_name):
            return player_name in self.player_positions

        def get_player_at_position(self, position):
            return self.assignments.get(position)

        def remove_player(self, player_name):
            if player_name in self.player_positions:
                position = self.player_positions[player_name]
                del self.assignments[position]
                del self.player_positions[player_name]
                return position
            return None

        def get_assigned_positions(self):
            return set(self.assignments.keys())

        def get_assigned_players(self):
            return set(self.player_positions.keys())

    def find_optimal_assignment():
        assignment = OptimalAssignment()

        for pos in positions:
            best_available = None
            best_score = -1
            best_fielding = -1
            for player_name, score, player_data in position_rankings[pos]:
                if not assignment.is_player_assigned(player_name):
                    fielding = fielding_score_map.get(player_name, 0)
                    if (abs(score - best_score) < 0.001):
                        if fielding > best_fielding:
                            best_available = player_name
                            best_score = score
                            best_fielding = fielding
                    elif score > best_score:
                        best_available = player_name
                        best_score = score
                        best_fielding = fielding
            if best_available:
                assignment.assign(pos, best_available)

        improved = True
        max_iterations = 10
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for pos in positions:
                current_player = assignment.get_player_at_position(pos)
                if not current_player:
                    continue
                current_score = player_position_scores[current_player][pos]['score']
                current_fielding = player_position_scores[current_player][pos]['fielding_score']
                for player_name, score, player_data in position_rankings[pos]:
                    new_fielding = fielding_score_map.get(player_name, 0)
                    if not assignment.is_player_assigned(player_name):
                        if (
                            score > current_score or
                            (abs(score - current_score) < 0.01 and new_fielding > current_fielding)
                        ):
                            assignment.assign(pos, player_name)
                            improved = True
                            break
                    else:
                        other_pos = assignment.player_positions[player_name]
                        other_current_score = player_position_scores[player_name][other_pos]['score']
                        current_at_other_score = player_position_scores[current_player].get(other_pos, {}).get('score', 0)
                        if current_at_other_score == 0:
                            continue
                        current_total = current_score + other_current_score
                        new_total = score + current_at_other_score
                        if new_total > current_total:
                            assignment.assign(pos, player_name)
                            assignment.assign(other_pos, current_player)
                            improved = True
                            break
                if improved:
                    break

        for pos in positions:
            if pos not in assignment.assignments:
                for player_name, score, player_data in position_rankings[pos]:
                    if not assignment.is_player_assigned(player_name):
                        assignment.assign(pos, player_name)
                        break

        return assignment

    final_assignment = find_optimal_assignment()

    final_selections = []
    total_score = 0
    for pos in positions:
        player_name = final_assignment.get_player_at_position(pos)
        if player_name and pos in player_position_scores[player_name]:
            player_data = player_position_scores[player_name][pos]['data']
            final_selections.append(player_data)
            total_score += player_position_scores[player_name][pos]['score']

    if final_selections:
        best_7_df = pd.DataFrame(final_selections)
        best_7_df = best_7_df.sort_values("Position").reset_index(drop=True)

        st.subheader("Position-Optimal Batting Lineup")
        st.write(f"**Total Lineup Score: {total_score:.4f}**")

        display_professional_table(
            best_7_df[[
                "Player Name", "Role", "Position", "Rank",
                "Predicted_Score", "Effective_Score", "Fielding_Score"
            ]],
            number_formats={
                "Position": "{:.0f}",
                "Rank": "{:.0f}",
                "Predicted_Score": "{:.4f}",
                "Effective_Score": "{:.4f}",
                "Fielding_Score": "{:.4f}",
            },
        )

        tab1, tab2 = st.tabs([
            "Position Performance Analysis",
            "Assignment Process"
        ])

        with tab1:
            performance_rows = []

            for pos in positions:
                player_name = final_assignment.get_player_at_position(pos)
                if not player_name:
                    continue

                best_for_pos = position_rankings[pos][0][0] if position_rankings[pos] else None
                is_optimal_for_position = player_name == best_for_pos

                player_rank = next(
                    (i + 1 for i, (p_name, _, _) in enumerate(position_rankings[pos])
                     if p_name == player_name),
                    None
                )

                score = player_position_scores[player_name][pos]["score"]

                performance_rows.append({
                    "Position": pos,
                    "Selected Player": player_name,
                    "Selection Status": "Optimal for Position" if is_optimal_for_position else f"Rank {player_rank} — Global Assignment",
                    "Score": score,
                })

            performance_df = pd.DataFrame(performance_rows)
            display_professional_table(
                performance_df,
                number_formats={
                    "Position": "{:.0f}",
                    "Score": "{:.4f}",
                },
            )

        with tab2:
            assignment_rows = []

            for pos in positions:
                assigned_player = final_assignment.get_player_at_position(pos)

                for rank, (player_name, score, player_data) in enumerate(
                    position_rankings[pos], start=1
                ):
                    if player_name == assigned_player:
                        status = "Selected"
                    elif not final_assignment.is_player_assigned(player_name):
                        status = "Available"
                    else:
                        status = "Assigned to Another Position"

                    assignment_rows.append({
                        "Position": pos,
                        "Rank": rank,
                        "Player Name": player_name,
                        "Role": player_data.get("Role", ""),
                        "Score": score,
                        "Status": status,
                    })

            assignment_df = pd.DataFrame(assignment_rows)
            display_professional_table(
                assignment_df,
                number_formats={
                    "Position": "{:.0f}",
                    "Rank": "{:.0f}",
                    "Score": "{:.4f}",
                },
                height=420,
            )

        return best_7_df, position_rankings
    else:
        st.error("Could not create optimal lineup assignments.")
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

def select_dynamic_reliable_batters(final_df, roles_df, innings_df, match_context, fielding_df):

    positions = list(range(1, 8))
    combo = match_context["Team_Combo"]

    # Normalize unavailable-player names once so whitespace/casing differences
    # cannot accidentally allow an unavailable player into the final lineup.
    unavailable = {
        str(name).strip()
        for name in match_context.get("Unavailable", [])
        if pd.notna(name) and str(name).strip()
    }

    max_batters = min(combo["Batters"] + combo["Batting_AR"], 7)
    required_bats = combo["Batters"]
    required_ars = combo["Batting_AR"]
    required_wk = 1

    fielding_score_map = fielding_df.set_index("Player Name")["Predicted_Fielding_Score"].to_dict()
    innings_summary = innings_df.groupby(["Player Name", "Position"])["Inns"].sum().reset_index()

    position_rankings = {}
    player_position_scores = defaultdict(dict)

    for pos in positions:
        pos_df = final_df[final_df["Position"] == pos].copy()
        valid_roles = ["Batsman", "WK-Batsman", "Batting Allrounder"]
        pos_df = pos_df[pos_df["Role"].isin(valid_roles)]

        # Remove unavailable players before scoring, ranking, and optimization.
        if unavailable and not pos_df.empty:
            pos_df = pos_df[
                ~pos_df["Player Name"].astype(str).str.strip().isin(unavailable)
            ]

        pos_inns_map = innings_summary[innings_summary["Position"] == pos].set_index("Player Name")["Inns"].to_dict()
        std_score = pos_df["Predicted_Score"].std()
        std_inns = pd.Series(pos_inns_map).std()
        total_var = std_score + std_inns + 1e-8
        alpha = std_score / total_var
        beta = std_inns / total_var

        pos_df["Inns_at_Pos"] = pos_df["Player Name"].map(pos_inns_map).fillna(0)
        pos_df["Max_Inns_Pos"] = max(pos_inns_map.values()) if pos_inns_map else 1
        pos_df["Normalized_Inns"] = pos_df["Inns_at_Pos"] / pos_df["Max_Inns_Pos"]
        pos_df["Effective_Score"] = pos_df["Predicted_Score"] * (alpha + beta * pos_df["Normalized_Inns"])
        pos_df["Fielding_Score"] = pos_df["Player Name"].map(fielding_score_map).fillna(0)

        pos_df = pos_df.sort_values(by=["Effective_Score", "Fielding_Score"], ascending=False)
        pos_df = pos_df.drop_duplicates("Player Name")

        position_rankings[pos] = []
        for rank_idx, (_, player) in enumerate(pos_df.iterrows(), start=1):
            player_name = player["Player Name"]
            score = player["Effective_Score"]
            player_data = player.to_dict()
            player_data["Rank"] = rank_idx

            position_rankings[pos].append((player_name, score, player_data))
            player_position_scores[player_name][pos] = {
                'score': score,
                'fielding_score': player_data["Fielding_Score"],
                'data': player_data
            }

    # ------------------------------------------------------------------
    # Global batting assignment with the existing score methodology.
    #
    # WK-Batsman counts as a Batter and simultaneously satisfies the
    # minimum-WK requirement. It does not consume an additional slot.
    # ------------------------------------------------------------------
    required_batting_players = required_bats + required_ars

    if required_batting_players > len(positions):
        st.warning(
            f"The requested batting composition requires {required_batting_players} "
            f"players, but only {len(positions)} batting positions are available."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    candidates = []
    for player_name, position_dict in player_position_scores.items():
        for pos, info in position_dict.items():
            data = info["data"]
            candidates.append({
                "player": player_name,
                "position": pos,
                "role": data["Role"],
                "score": float(info["score"]),
                "fielding": float(info["fielding_score"]),
                "data": data,
            })

    if not candidates:
        st.warning("No eligible batting player-position combinations are available.")
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    position_to_indices = defaultdict(list)
    player_to_indices = defaultdict(list)
    batter_indices = []
    ar_indices = []
    wk_indices = []

    for i, candidate in enumerate(candidates):
        role = str(candidate["role"]).strip()

        position_to_indices[candidate["position"]].append(i)
        player_to_indices[candidate["player"]].append(i)

        if role in ["Batsman", "WK-Batsman"]:
            batter_indices.append(i)
        elif role == "Batting Allrounder":
            ar_indices.append(i)

        if role == "WK-Batsman":
            wk_indices.append(i)

    # A minimum of one WK is required, but that WK is already part of
    # the requested Batter count.
    if not wk_indices:
        st.warning(
            "Unable to satisfy the minimum WK requirement because "
            "no eligible WK-Batsman is available."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    if len({candidates[i]["player"] for i in batter_indices}) < required_bats:
        st.warning(
            f"Unable to satisfy the requested Batter count ({required_bats})."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    if len({candidates[i]["player"] for i in ar_indices}) < required_ars:
        st.warning(
            f"Unable to satisfy the requested Batting Allrounder count ({required_ars})."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    n = len(candidates)
    # Keep Effective_Score as the optimization objective.
    objective = [-c["score"] for c in candidates]

    row_specs = []

    # One player per batting position.
    for indices in position_to_indices.values():
        row_specs.append((indices, 0.0, 1.0))

    # A player can occupy only one batting position.
    for indices in player_to_indices.values():
        row_specs.append((indices, 0.0, 1.0))

    # Exact Batter count. WK-Batsmen are included here.
    row_specs.append(
        (batter_indices, float(required_bats), float(required_bats))
    )

    # Exact Batting Allrounder count.
    row_specs.append(
        (ar_indices, float(required_ars), float(required_ars))
    )

    # At least one WK, without adding an extra batting slot.
    row_specs.append(
        (wk_indices, 1.0, float(len(wk_indices)))
    )

    # Exact total batting players.
    row_specs.append(
        (
            list(range(n)),
            float(required_batting_players),
            float(required_batting_players),
        )
    )

    A = lil_matrix((len(row_specs), n), dtype=float)
    lower_bounds = []
    upper_bounds = []

    for r, (indices, lb, ub) in enumerate(row_specs):
        for i in indices:
            A[r, i] = 1.0
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    result = milp(
        c=objective,
        integrality=[1] * n,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            A.tocsr(),
            lower_bounds,
            upper_bounds,
        ),
        options={"time_limit": 10},
    )

    if not result.success or result.x is None:
        st.warning(
            "The requested batting composition could not be assigned "
            "without violating the player, position, role, or WK constraints."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    selected_players = [
        candidates[i]["data"]
        for i, value in enumerate(result.x)
        if value > 0.5
    ]

    if not selected_players:
        st.warning("Could not create the requested batting lineup.")
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Score"]
        )

    # Rebuild the final assignment from the optimizer result.
    # The optimizer returns a plain dictionary here, so do not treat it as
    # the old RoleAwareAssignment object.
    final_assignment = {
        int(row["Position"]): row["Player Name"]
        for row in selected_players
    }

    final_selections = []
    total_score = 0
    for pos in sorted(final_assignment.keys()):
        player_name = final_assignment.get(pos)
        if player_name and pos in player_position_scores[player_name]:
            player_data = player_position_scores[player_name][pos]['data']
            final_selections.append(player_data)
            total_score += player_position_scores[player_name][pos]['score']

    if final_selections:
        best_df = pd.DataFrame(final_selections).sort_values("Position").reset_index(drop=True)

        # Final integrity check: reject any lineup that violates the
        # requested composition rather than silently returning it.
        role_series = best_df["Role"].astype(str).str.strip()
        actual_batters = int(role_series.isin(["Batsman", "WK-Batsman"]).sum())
        actual_ars = int((role_series == "Batting Allrounder").sum())
        actual_wks = int((role_series == "WK-Batsman").sum())

        validation_ok = (
            len(best_df) == required_batting_players
            and actual_batters == required_bats
            and actual_ars == required_ars
            and actual_wks >= required_wk
            and best_df["Player Name"].nunique() == len(best_df)
            and best_df["Position"].nunique() == len(best_df)
            and not best_df["Player Name"].astype(str).str.strip().isin(unavailable).any()
        )

        if not validation_ok:
            st.error(
                "Batting assignment validation failed. The lineup was not accepted."
            )
            return pd.DataFrame(
                columns=["Player Name", "Role", "Position", "Predicted_Score"]
            )
        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Role-Constrained Position-Optimal Batting Lineup")
        st.write(f"**Total Lineup Score: {total_score:.4f}**")
        display_professional_table(
            best_df[["Player Name", "Role", "Position", "Predicted_Score"]],
            number_formats={
                "Position": "{:.0f}",
                "Predicted_Score": "{:.4f}",
            },
        )

        return best_df
    else:
        st.error("Could not create optimal lineup assignments.")
        return pd.DataFrame()

def select_dynamic_bowlers_assignment(bowlers_df, fielding_df, innings_df, match_context, used_positions, used_players):
    """Select bowlers using a joint role- and position-constrained optimization.

    The existing innings-aware Total_Score calculation is preserved. The change is
    only in the final assignment stage: instead of selecting greedily position by
    position, the function jointly optimizes player-position assignments while
    enforcing the exact bowling-role composition requested in Team_Combo.
    """
    combo = match_context["Team_Combo"]
    unavailable = {
        str(name).strip()
        for name in match_context.get("Unavailable", [])
        if pd.notna(name) and str(name).strip()
    }
    used_players = {
        str(name).strip()
        for name in (used_players or set())
        if pd.notna(name) and str(name).strip()
    }

    # Exact bowling-role requirements from the user's selected team composition.
    required_counts = {
        "Pacer": combo.get("Pacer_Pure", 0),
        "Spinner": combo.get("Spinner_Pure", 0),
        "Bowling Allrounder (Spinner)": combo.get("Spinner_Bowling_AR", 0),
    }
    total_required = sum(required_counts.values())
    max_positions = sorted(set(range(1, 13)) - set(used_positions))

    if total_required == 0:
        return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    # Mapping fielding score.
    fielding_map = fielding_df.set_index("Player Name")["Predicted_Fielding_Score"].to_dict()
    innings_summary = (
        innings_df.groupby(["Player Name", "Position"])["Inns"]
        .sum()
        .reset_index()
    )

    bowlers_df = bowlers_df.copy()
    bowlers_df["Fielding_Score"] = (
        bowlers_df["Player Name"].map(fielding_map).fillna(0)
    )

    # Filter out unavailable players, already-selected batting players, and roles
    # that are not part of the requested bowling composition.
    player_name_normalized = bowlers_df["Player Name"].astype(str).str.strip()
    bowlers_df = bowlers_df[
        (~player_name_normalized.isin(unavailable))
        & (~player_name_normalized.isin(used_players))
        & (bowlers_df["Role"].astype(str).str.strip().isin(required_counts.keys()))
        & (bowlers_df["Position"].isin(max_positions))
    ].copy()

    if bowlers_df.empty:
        st.warning("No eligible bowlers are available for the requested role composition.")
        return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    # ------------------------------------------------------------------
    # Preserve the existing innings-aware reliability scoring.
    # ------------------------------------------------------------------
    player_position_scores = defaultdict(dict)

    for pos in max_positions:
        pos_df = bowlers_df[bowlers_df["Position"] == pos].copy()
        if pos_df.empty:
            continue

        pos_inns_map = (
            innings_summary[innings_summary["Position"] == pos]
            .set_index("Player Name")["Inns"]
            .to_dict()
        )

        std_score = pos_df["Predicted_Bowl_Score"].std()
        std_inns = pd.Series(pos_inns_map).std()

        # Avoid NaN weights when a position has only one available value.
        std_score = 0.0 if pd.isna(std_score) else float(std_score)
        std_inns = 0.0 if pd.isna(std_inns) else float(std_inns)

        total_var = std_score + std_inns + 1e-8
        alpha = std_score / total_var
        beta = std_inns / total_var

        pos_df["Inns_at_Pos"] = pos_df["Player Name"].map(pos_inns_map).fillna(0)
        max_inns_pos = max(pos_inns_map.values()) if pos_inns_map else 1
        max_inns_pos = max(float(max_inns_pos), 1.0)
        pos_df["Max_Inns_Pos"] = max_inns_pos
        pos_df["Normalized_Inns"] = pos_df["Inns_at_Pos"] / max_inns_pos

        pos_df["Effective_Score"] = (
            pos_df["Predicted_Bowl_Score"]
            * (alpha + beta * pos_df["Normalized_Inns"])
        )
        pos_df["Total_Score"] = (
            pos_df["Effective_Score"] + pos_df["Fielding_Score"]
        )

        # If duplicate rows exist for the same player-position-role combination,
        # retain the highest-scoring one.
        pos_df = pos_df.sort_values(
            by=["Total_Score", "Fielding_Score", "Predicted_Bowl_Score"],
            ascending=False,
        ).drop_duplicates(subset=["Player Name", "Role"], keep="first")

        for _, row in pos_df.iterrows():
            name = row["Player Name"]
            role = row["Role"]
            score = float(row["Total_Score"])
            fielding = float(row["Fielding_Score"])

            player_position_scores[name][pos] = {
                "score": score,
                "fielding_score": fielding,
                "data": row.to_dict(),
            }

    # ------------------------------------------------------------------
    # Build one binary decision variable for every feasible
    # (player, position) assignment.
    # ------------------------------------------------------------------
    candidates = []
    for player_name, position_dict in player_position_scores.items():
        for pos, info in position_dict.items():
            data = info["data"]
            candidates.append({
                "player": player_name,
                "position": pos,
                "role": data["Role"],
                "score": info["score"],
                "fielding": info["fielding_score"],
                "data": data,
            })

    if not candidates:
        st.warning("No feasible player-position combinations are available.")
        return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    # Normalize role strings to the exact keys used in required_counts.
    role_lookup = {
        role.strip().lower(): role for role in required_counts
    }
    for candidate in candidates:
        candidate["role_key"] = role_lookup.get(
            str(candidate["role"]).strip().lower()
        )

    candidates = [c for c in candidates if c["role_key"] is not None]

    # Quick feasibility checks before invoking the optimizer.
    available_positions = set(c["position"] for c in candidates)
    if len(available_positions) < total_required:
        st.warning(
            "Unable to create the requested bowling composition because "
            "there are not enough available positions."
        )
        return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    for role, required in required_counts.items():
        if required <= 0:
            continue
        role_players = {
            c["player"] for c in candidates if c["role_key"] == role
        }
        if len(role_players) < required:
            st.warning(
                f"Unable to satisfy the requested {role} count ({required}) "
                f"with the available eligible players."
            )
            return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    n = len(candidates)
    objective = [-c["score"] for c in candidates]  # milp minimizes

    constraints = []
    lower_bounds = []
    upper_bounds = []

    # Constraint 1: at most one player per available position.
    position_to_indices = defaultdict(list)
    for i, c in enumerate(candidates):
        position_to_indices[c["position"]].append(i)

    # Constraint 2: at most one position per player.
    player_to_indices = defaultdict(list)
    for i, c in enumerate(candidates):
        player_to_indices[c["player"]].append(i)

    # Constraint 3: exact number of players for each required role.
    role_to_indices = defaultdict(list)
    for i, c in enumerate(candidates):
        role_to_indices[c["role_key"]].append(i)

    # Constraint 4: exact total number of bowlers.
    row_specs = []

    for indices in position_to_indices.values():
        row_specs.append((indices, 0.0, 1.0))

    for indices in player_to_indices.values():
        row_specs.append((indices, 0.0, 1.0))

    for role, required in required_counts.items():
        row_specs.append((role_to_indices.get(role, []), float(required), float(required)))

    row_specs.append((list(range(n)), float(total_required), float(total_required)))

    A = lil_matrix((len(row_specs), n), dtype=float)
    for r, (indices, lb, ub) in enumerate(row_specs):
        for i in indices:
            A[r, i] = 1.0
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    result = milp(
        c=objective,
        integrality=[1] * n,
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            A.tocsr(),
            lower_bounds,
            upper_bounds,
        ),
        options={"time_limit": 10},
    )

    if not result.success or result.x is None:
        st.warning(
            "The requested bowling role composition could not be assigned "
            "to the available positions without violating the constraints."
        )
        return pd.DataFrame(columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"])

    # ------------------------------------------------------------------
    # Extract the globally optimized assignment.
    # ------------------------------------------------------------------
    selected_bowlers = []
    for i, value in enumerate(result.x):
        if value > 0.5:
            selected_bowlers.append(candidates[i]["data"])

    final_bowlers_df = pd.DataFrame(selected_bowlers).reset_index(drop=True)
    final_bowlers_df = final_bowlers_df.sort_values(
        by="Position"
    ).reset_index(drop=True)

    # Final integrity check only; the existing bowling MILP and scoring
    # objective remain unchanged.
    role_series = final_bowlers_df["Role"].astype(str).str.strip()
    actual_counts = {
        "Pacer": int((role_series == "Pacer").sum()),
        "Spinner": int((role_series == "Spinner").sum()),
        "Bowling Allrounder (Spinner)": int(
            (role_series == "Bowling Allrounder (Spinner)").sum()
        ),
    }

    validation_ok = (
        len(final_bowlers_df) == total_required
        and all(
            actual_counts[role] == required
            for role, required in required_counts.items()
        )
        and final_bowlers_df["Player Name"].nunique() == len(final_bowlers_df)
        and final_bowlers_df["Position"].nunique() == len(final_bowlers_df)
        and not final_bowlers_df["Player Name"].astype(str).str.strip().isin(unavailable).any()
        and not final_bowlers_df["Player Name"].astype(str).str.strip().isin(used_players).any()
        and not final_bowlers_df["Position"].isin(set(used_positions)).any()
    )

    if not validation_ok:
        st.error(
            "Bowling assignment validation failed. The lineup was not accepted."
        )
        return pd.DataFrame(
            columns=["Player Name", "Role", "Position", "Predicted_Bowl_Score"]
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Role-Constrained Optimal Bowling Selection")

    display_professional_table(
        final_bowlers_df[
            ["Player Name", "Role", "Position", "Predicted_Bowl_Score"]
        ].rename(columns={
            "Predicted_Bowl_Score": "Predicted_Score"
        }),
        number_formats={
            "Position": "{:.0f}",
            "Predicted_Score": "{:.4f}",
        },
    )

    tab1, tab2 = st.tabs([
        "Position Performance Analysis",
        "Assignment Process"
    ])

    selected_by_position = {
        int(row["Position"]): row["Player Name"]
        for _, row in final_bowlers_df.iterrows()
    }

    with tab1:
        performance_rows = []

        for pos in max_positions:
            pos_candidates = []

            for player_name, position_dict in player_position_scores.items():
                if pos in position_dict:
                    info = position_dict[pos]
                    data = info["data"]
                    pos_candidates.append((
                        player_name,
                        float(info["score"]),
                        data.get("Role", "")
                    ))

            if not pos_candidates:
                continue

            pos_candidates.sort(key=lambda x: x[1], reverse=True)
            selected_player = selected_by_position.get(pos)

            if selected_player:
                selected_score = float(
                    player_position_scores[selected_player][pos]["score"]
                )
                selected_rank = next(
                    (i + 1 for i, candidate in enumerate(pos_candidates)
                     if candidate[0] == selected_player),
                    None
                )

                status = (
                    "Optimal for Position"
                    if selected_rank == 1
                    else f"Rank {selected_rank} — Global Assignment"
                )

                performance_rows.append({
                    "Position": pos,
                    "Selected Player": selected_player,
                    "Role": player_position_scores[selected_player][pos]["data"].get("Role", ""),
                    # "Selection Status": status,
                    "Total Score": selected_score,
                })

        performance_df = pd.DataFrame(performance_rows)
        display_professional_table(
            performance_df,
            number_formats={
                "Position": "{:.0f}",
                "Total Score": "{:.4f}",
            },
        )

    with tab2:
        assignment_rows = []

        for pos in max_positions:
            pos_candidates = []

            for player_name, position_dict in player_position_scores.items():
                if pos in position_dict:
                    info = position_dict[pos]
                    data = info["data"]
                    pos_candidates.append((
                        player_name,
                        float(info["score"]),
                        data.get("Role", ""),
                        data
                    ))

            pos_candidates.sort(key=lambda x: x[1], reverse=True)
            assigned_player = selected_by_position.get(pos)

            for rank, (player_name, score, role, data) in enumerate(
                pos_candidates, start=1
            ):
                if player_name == assigned_player:
                    status = "Selected"
                elif player_name in selected_by_position.values():
                    status = "Assigned to Another Position"
                else:
                    status = "Available"

                assignment_rows.append({
                    "Position": pos,
                    "Rank": rank,
                    "Player Name": player_name,
                    "Role": role,
                    "Predicted Bowl Score": data.get("Predicted_Bowl_Score"),
                    "Effective Score": data.get("Effective_Score"),
                    "Fielding Score": data.get("Fielding_Score"),
                    "Total Score": score,
                    "Status": status,
                })

        assignment_df = pd.DataFrame(assignment_rows)
        display_professional_table(
            assignment_df,
            number_formats={
                "Position": "{:.0f}",
                "Rank": "{:.0f}",
                "Predicted Bowl Score": "{:.4f}",
                "Effective Score": "{:.4f}",
                "Fielding Score": "{:.4f}",
                "Total Score": "{:.4f}",
            },
            height=500,
        )

    return final_bowlers_df

