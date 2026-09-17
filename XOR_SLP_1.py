import pandas as pd
import numpy as np
from pathlib import Path

# Import Excel file
workbook_path = Path(__file__).resolve().parent / "Logic_Gates_With_Step_Activation (1).xlsx"
df = pd.read_excel(workbook_path)

# Take only the header
header = df.columns

print("Header:")
print(header.tolist())


# --------------------------------
# XOR Gate
# --------------------------------

learning_rate = 0.5

w1 = 0
w2 = 0
b = 0

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = np.array([0, 1, 1, 0])

epochs = 5
iteration = 1

# Print header
print("\nEpoch\tIter\tx1\tx2\tTarget(y)\tw1_in\tw2_in\tb_in\tz\tPrediction\tError(e)\tw1_out\tw2_out\tb_out")



for epoch in range(1, epochs + 1):

    for i in range(len(X)):

        x1 = X[i, 0]
        x2 = X[i, 1]
        target = Y[i]

        # Input weights
        w1_in = w1
        w2_in = w2
        b_in = b

        # Calculate z
        z = w1_in * x1 + w2_in * x2 + b_in

        # Step activation
        if z < 0:
            prediction = 1
        else:
            prediction = 0

        # Error
        error = target - prediction

        # Weight update
        w1_out = w1_in + learning_rate * error * x1
        w2_out = w2_in + learning_rate * error * x2
        b_out = b_in + learning_rate * error

        # Print row
        print(
            f"{epoch}\t{iteration}\t{x1}\t{x2}\t{target}\t"
            f"{w1_in:.1f}\t{w2_in:.1f}\t{b_in:.1f}\t{z:.1f}\t"
            f"{prediction}\t{error}\t{w1_out:.1f}\t"
            f"{w2_out:.1f}\t{b_out:.1f}"
        )

        # Update weights
        w1 = w1_out
        w2 = w2_out
        b = b_out

        iteration += 1