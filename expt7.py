#Import the modules you need. scikit-learn is a popular library formachine learning
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#Create arrays that resemble two variables in a dataset.
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
plt.scatter(x, y)
plt.show()
#Turn the data into a set of points:
data = list(zip(x, y))
print(data)
#Decide on the number of clusters you want the algorithm to find.
#This is often based on your understanding of the data or through techniques like the elbow method.
kmeans = KMeans(n_clusters=2)
#Fit the Model:Train the K-Means model on your data.
kmeans.fit(data)
# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)
#plot the different clusters assigned to the data:
plt.scatter(x, y, c=kmeans.labels_)
plt.show()

