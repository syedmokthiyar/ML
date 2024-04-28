# ML

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to calculate MMAPE
def calculate_mmape(actual, predicted):
    n = len(actual)
    mmape = 0

    for i in range(n):
        if actual[i] != 0:
            mmape += abs((actual[i] - predicted[i]) / actual[i] * (1 + (actual[i] - predicted[i]) / actual[i]))
        else:
            mmape += abs(actual[i] - predicted[i])

    mmape /= n
    return mmape

# Generate example data
np.random.seed(0)
X = np.random.rand(100, 1)  # Example feature (random)
y = np.random.randint(1, 1000, size=(100,))  # Example target (random number of deaths)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MMAPE score
mmape_score = calculate_mmape(y_test, y_pred)
print("MMAPE Score:", mmape_score)

```


# Output

![Screenshot 2024-04-28 133251](https://github.com/syedmokthiyar/ML/assets/118787294/857cc321-4da8-4622-ad2e-f8bfb0aaacb5)




