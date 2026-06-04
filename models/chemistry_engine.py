import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv("data/final_merged_dataset.csv")

embeddings = np.load("models/latent_embeddings.npy")

similarity_matrix = cosine_similarity(embeddings)


def get_position_fit(pos1, pos2):
    pos1 = str(pos1)
    pos2 = str(pos2)

    if pos1 == pos2:
        return 1.0

    midfield = ["MF", "DM", "CM", "AM"]
    attack = ["FW", "ST", "LW", "RW"]
    defense = ["DF", "CB", "LB", "RB"]

    if any(x in pos1 for x in midfield) and any(x in pos2 for x in midfield):
        return 0.9

    if any(x in pos1 for x in attack) and any(x in pos2 for x in attack):
        return 0.9

    if any(x in pos1 for x in defense) and any(x in pos2 for x in defense):
        return 0.9

    if (
        any(x in pos1 for x in midfield)
        and any(x in pos2 for x in attack)
    ) or (
        any(x in pos2 for x in midfield)
        and any(x in pos1 for x in attack)
    ):
        return 0.8

    return 0.5


def get_complementarity(player1, player2):
    creative_cols = [
        "Vision",
        "Short Passing",
        "Long Passing"
    ]

    defensive_cols = [
        "Defending",
        "Interceptions"
    ]

    attacking_cols = [
        "Finishing",
        "Positioning"
    ]

    score = 0

    score += abs(
        player1[creative_cols].mean()
        - player2[creative_cols].mean()
    )

    score += abs(
        player1[defensive_cols].mean()
        - player2[defensive_cols].mean()
    )

    score += abs(
        player1[attacking_cols].mean()
        - player2[attacking_cols].mean()
    )

    score = score / 300

    return min(score, 1)


def get_chemistry_players(player_name, top_n=5):

    matches = df[
        df["Player"].str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(matches) == 0:
        print("Player not found.")
        return

    player_index = matches.index[0]
    matched_name = df.iloc[player_index]["Player"]

    print(f"\nMatched Player: {matched_name}")

    player_pos = df.iloc[player_index]["Pos"]
    player_row = df.iloc[player_index]

    chemistry_scores = []

    for idx in range(len(df)):

        if idx == player_index:
            continue

        similarity = similarity_matrix[player_index][idx]

        position_fit = get_position_fit(
            player_pos,
            df.iloc[idx]["Pos"]
        )

        complementarity = get_complementarity(
            player_row,
            df.iloc[idx]
        )

        chemistry = (
            0.5 * similarity
            + 0.3 * position_fit
            + 0.2 * complementarity
        )

        chemistry_scores.append((idx, chemistry))

    chemistry_scores = sorted(
        chemistry_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_players = chemistry_scores[:top_n]

    print(f"\nTop chemistry partners for {matched_name}:\n")

    for idx, score in top_players:
        print(
            f"{df.iloc[idx]['Player']} -> Chemistry: {score:.3f}"
        )


player_input = input("Enter player name: ")

get_chemistry_players(player_input)