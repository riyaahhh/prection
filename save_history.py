# =========================================================
# PASTE THIS AT THE END OF YOUR COLAB NOTEBOOK
# (after Step 8, once onion_df / potato_df already exist)
# Exports the recent history the live API needs to build
# lag + rolling features for predictions.
# =========================================================

import os

os.makedirs("data", exist_ok=True)

onion_df.reset_index().to_csv("data/onion_history.csv", index=False)
potato_df.reset_index().to_csv("data/potato_history.csv", index=False)

print("Saved onion_history.csv and potato_history.csv to the data/ folder.")
print("Download these from Colab's file panel along with your .pkl model files.")
