import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 

style.use('ggplot')

class KMeans:
	def __init__(self, k =3, tol = 0.0001, max_iter = 500):
		self.k = k
		self.tolerance = tol
		self.max_iterations= max_iter

	def fit(self, data):

		self.centroids = {}

		
		for i in range(self.k):
			self.centroids[i] = data[i]

		
		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []

			
			for features in data:
				distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
				classi= distances.index(min(distances))
				self.classes[classi].append(features)

			old = dict(self.centroids)

			
			for classi in self.classes:
				self.centroids[classi] = np.average(self.classes[classi], axis = 0)

			isOptimal = True

			for centroid in self.centroids:

				original_centroid = old[centroid]
				curr = self.centroids[centroid]

				if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
					isOptimal = False

			
			if isOptimal:
				break

	def pred(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classi= distances.index(min(distances))
		return classi

def main():
	
	df = pd.read_csv("Humanities_kmeans.csv")
	df = df[['one', 'two']]
	dataset = df.astype(float).values.tolist()

	X = df.values
	km = KMeans(3)
	km.fit(X)

	
	colors = 10*["r", "g", "c", "b", "k"]

	for centroid in km.centroids:
		plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s = 130, marker = "x")

	for classi in km.classes:
		color = colors[classi]
		for features in km.classes[classi]:
			plt.scatter(features[0], features[1], color = color,s = 30)
	
	plt.show()

if __name__ == "__main__":
 main()
