from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample training using the digits dataset
data = load_digits()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
