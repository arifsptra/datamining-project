import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from web_functions import load_data, proses_data

dh, x, y = load_data()
x_train, x_test, y_train, y_test = proses_data(x, y)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Create and train the KNN model
knnModel = KNeighborsClassifier(n_neighbors=3)
knnModel.fit(x_train, y_train)

# Save the trained model to a file
with open('knn_model.sav', 'wb') as file:
    pickle.dump(knnModel, file)