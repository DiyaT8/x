# Import necessary libraries
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# Data
weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 
           'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 
           'Overcast', 'Rainy']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
        'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# Create LabelEncoder
le = preprocessing.LabelEncoder()

# Convert string labels into numbers
weather_encoded = le.fit_transform(weather)
temp_encoded = le.fit_transform(temp)
label = le.fit_transform(play)

# Print encoded values
print("Weather Encoded:", weather_encoded)
print("Temp Encoded:", temp_encoded)
print("Play Encoded:", label)

# Combine weather and temp into a single list of tuples
features = list(zip(weather_encoded, temp_encoded))

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features, label)

# Predict Output for a given input
predicted = model.predict([[0, 2]])  # 0: Overcast, 2: Mild
print("Predicted Value:", predicted)
