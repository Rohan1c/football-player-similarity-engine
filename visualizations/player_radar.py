import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/final_merged_dataset.csv"
)

players = [

    "Pedri",

    "Rodri",

    "Valverde"

]

features = [

    "Pace",

    "Shooting",

    "Passing",

    "Dribbling",

    "Defending",

    "Physicality"

]

angles = np.linspace(

    0,

    2*np.pi,

    len(features),

    endpoint=False

).tolist()

angles += angles[:1]

fig = plt.figure(
    figsize=(10,10)
)

ax = plt.subplot(
    111,
    polar=True
)

for player_name in players:

    player = df[
        df["Player"]
        .str.contains(
            player_name,
            case=False,
            na=False
        )
    ]

    if len(player)==0:
        continue

    player = player.iloc[0]

    values = [

        player[f]

        for f in features

    ]

    values += values[:1]

    ax.plot(

        angles,

        values,

        linewidth=2,

        label=player["Player"]

    )

    ax.fill(

        angles,

        values,

        alpha=0.1

    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(
    features
)

plt.title(

    "FootballIQ Midfielder Archetype Comparison",

    fontsize=16,

    fontweight="bold"

)

plt.legend(
    loc="upper right"
)

plt.tight_layout()

plt.show()