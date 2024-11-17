# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False 

import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
plt.rcParams['savefig.dpi'] = 100 
plt.rcParams['figure.dpi'] = 100 

from sklearn import metrics
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.decomposition import PCA, KernelPCA
from sklearn import random_projection
from sklearn import preprocessing
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import IsolationForest

from itertools import cycle
import glob
import pickle
import os
from typing import Optional

#去掉科学计数法
pd.set_option('display.expand_frame_repr', False)

class MSK:
	def __init__(
		self,
		data: pd.DataFrame,
		standardized_method: list = ['minmax', 0, 1],
		reduced_n_dim: Optional[int] = None,
		reduced_method: str = 'PCA'
	) -> None:
		"""Initialize MSK clustering class
		
		Args:
			data: Input DataFrame
			standardized_method: Standardization method and params
			reduced_n_dim: Number of dimensions to reduce to
			reduced_method: Dimension reduction method
		"""
		self._validate_inputs(data, standardized_method, reduced_method)
		
		self.data = pd.DataFrame(data)
		self.standardized_method = standardized_method
		self.reduced_n_dim = reduced_n_dim
		self.method = reduced_method
		self.cont_rate_list = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

	def _validate_inputs(self, data, standardized_method, reduced_method):
		"""Validate input parameters"""
		if not isinstance(data, (pd.DataFrame, np.ndarray)):
			raise TypeError("Data must be DataFrame or ndarray")
			
		valid_methods = ['PCA', 'FeatureAgglomeration', 'GaussianRandomProjection', 'SparseRandomProjection']
		if reduced_method not in valid_methods:
			raise ValueError(f"Method must be one of {valid_methods}")

	def data_preprocessing(self) -> None:
		"""Data preprocessing including cleaning and standardization"""
		try:
			# Remove rows with any null values
			self.data = self.data.dropna()
			
			if self.standardized_method[0] == 'minmax':
				self._minmax_scale()
			elif self.standardized_method[0] == 'zscore': 
				self._zscore_scale()
			else:
				raise ValueError("Invalid standardization method")
				
		except Exception as e:
			raise RuntimeError(f"Error in preprocessing: {str(e)}")

	def _minmax_scale(self):
		"""Min-Max scaling"""
		self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
		self.data = self.data * (self.standardized_method[2] - self.standardized_method[1]) + self.standardized_method[1]

	def _zscore_scale(self):
		"""Z-score scaling"""
		self.data = (self.data - self.data.mean()) / self.data.std()

	def dimension_reduction(self, cont_rate=0.99):
		"""Dimension reduction
		
		Args:
			cont_rate: Contribution rates of each components used for determining how
			              many components are principle
		Explanation
			- firstly, determine "N" components contribute over 'cont_rate'	
			- then use specific method make data dimension reduced to "N" 
		"""

		if not self.reduced_n_dim:

			pca = PCA(n_components=self.data.shape[1])
			pca.fit(self.data)
			pca_score = pca.explained_variance_ratio_

			for i in range(len(pca_score)):
				if sum(pca_score[:i+1]) >= cont_rate:
					n_components = i + 1
					break

			print("  ----  The top-" + str(n_components) + " component(s) contribute(s) " + str(cont_rate*100) + "% ")

		else:
			n_components = self.reduced_n_dim

		print("  ----  Choose top-" + str(n_components) + " component(s) as principle component(s) ")

		if self.method == 'PCA':
			print("  ----  utilize 'PCA' dimensionality reduction method ")
			self.data = PCA(n_components=n_components).fit_transform(self.data)
			
		elif self.method == 'FeatureAgglomeration':
			print("  ----  utilize 'FeatureAgglomeration' dimensionality reduction method ")
			self.data = FeatureAgglomeration(n_clusters=n_components).fit_transform(self.data)

		elif self.method == 'GaussianRandomProjection':
			print("  ----  utilize 'GaussianRandomProjection' dimensionality reduction method ")
			self.data = random_projection.GaussianRandomProjection(n_components=n_components).fit_transform(self.data)

		elif self.method == 'SparseRandomProjection':
			print("  ----  utilize 'SparseRandomProjection' dimensionality reduction method ")
			self.data = random_projection.SparseRandomProjection(n_components=n_components).fit_transform(self.data)

		self.data = pd.DataFrame(self.data)
		

	def mean_shift(self, isPlot=False):
		"""MeanShift clustering
		
		Args:
			isPlot: Plot or not
		"""

		print(" [INFO] starting MeanShift combind method ")

		self.data_preprocessing()
		print(" [INFO] data pre-processing done ")

		ind = 0
		while self.data.shape[1] > self.data.shape[0]:

			self.dimension_reduction(cont_rate=self.cont_rate_list[ind])

			if self.data.shape[1] > self.data.shape[0]:
				print("  ----  reduced data dimension larger then data amount, alter another contribution rate ")
				ind += 1

			else:
				print("  ----  reduced data dimension smaller then data amount ")
				break

		print(" [INFO] dimensionality reduction done ")

		X = np.array(self.data)
		bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10000, random_state=42, n_jobs=2) 
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		ms.fit(X)
		output_label = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = np.unique(output_label)
		n_clusters_ = len(labels_unique)

		print(" [INFO] end ")

		data_cluster_dict = {}
		for cls in labels_unique:
			data_cluster_dict.update({cls:[]})
		for i in range(output_label.shape[0]):
			data_cluster_dict[output_label[i]].append(self.data.index.tolist()[i])

		if isPlot and X.shape[1] == 1:
			x = []
			for i in range(len(data_cluster_dict)):
				x.extend(data_cluster_dict[i])

			self.data = self.data.reindex(x)
			self.data = self.data.reset_index(drop=True)

			j = 0
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')
			plot_color = list('bgrcmykbgrcmykbgrcmykbgrcmyk')
			for i in range(len(data_cluster_dict)):
				cluster_center = cluster_centers[i]
				plt.plot(self.data.loc[j:j+len(data_cluster_dict[i])-1], color=plot_color[i], marker=plot_shape[i], \
					linestyle='', linewidth=2.0)
				tmp_list = list([kk for kk in range(j,j+len(data_cluster_dict[i])-1)])
				if len(tmp_list) == 0:
					plt.plot(self.data.loc[j:j+len(data_cluster_dict[i])-1],'o', markerfacecolor=plot_color[i],markeredgecolor='k', markersize=14)    
				else:
					plt.plot(np.mean(list([kk for kk in range(j,j+len(data_cluster_dict[i])-1)])), cluster_center[0],'o', markerfacecolor=plot_color[i], markeredgecolor='k', markersize=14)
				j += len(data_cluster_dict[i])
			plt.show()
		else:
			colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')

			for k, col in zip(list(set(output_label)), colors):
				my_members = output_label == k
				cluster_center = cluster_centers[k]
				plt.plot(X[my_members, 0], X[my_members, 1], col + plot_shape[k])
				plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
						markeredgecolor='k', markersize=14)
			plt.show()
		return output_label, data_cluster_dict

	def kmeans(self, n_cluster_list=list(range(2,10)), isPlot=False):
		"""KMeans clustering
		
		Args:
			n_cluster_list: List of cluster amounts for calculating different silhouette scores
			isPlot: Plot or not
		Explanation
			- if having data clustered into known "N" classes, for example "N=3", then set 'n_cluster_list=[3]'
			- if having no idea how many clusters should be decided, then give it a list with numbers you are
			  interested in, then select "N" with highest SILHOUETTE score 
		"""

		print(" [INFO] starting KMeans combind method ")

		self.data_preprocessing()
		print(" [INFO] data pre-processing done ")

		ind = 0
		while self.data.shape[1] > self.data.shape[0]:

			self.dimension_reduction(cont_rate=self.cont_rate_list[ind])

			if self.data.shape[1] > self.data.shape[0]:
				print("  ----  reduced data dimension larger then data amount, alter another contribution rate ")
				ind += 1

			else:
				print("  ----  reduced data dimension smaller then data amount ")
				break

		print(" [INFO] dimensionality reduction done ")

		np.random.seed(42)
		X = np.array(self.data) 

		if len(n_cluster_list) != 1:
			silhouette_avg = []
			for n in n_cluster_list:
				estimator = KMeans(init='random', n_clusters=n, max_iter=1000, n_init=10)
				cluster_labels = estimator.fit_predict(X)
				silhouette_avg.append(silhouette_score(X, cluster_labels))
				print("  ----  For n_clusters = " + str(n) + ", the average silhouette_score is : " + str(silhouette_avg[-1]) + ".")
			n_samples, n_features = X.shape
			n_clusters = n_cluster_list[silhouette_avg.index(max(silhouette_avg))] #
			print("  ----  Choose n_clusters = " + str(n_clusters) + " with max average silhouette score as final clusters number.")
			kmeans = KMeans(init='random', n_clusters=n_clusters, max_iter=1000, n_init=10)
			output_label = kmeans.fit_predict(X)
			cluster_centers = kmeans.cluster_centers_
		else:
			kmeans = KMeans(init='random', n_clusters=n_cluster_list[0], max_iter=1000, n_init=10)
			output_label = kmeans.fit_predict(X)
			cluster_centers = kmeans.cluster_centers_
            
		data_cluster_dict = {}
		for cls in set(output_label):
			data_cluster_dict.update({cls:[]})
		for i in range(output_label.shape[0]):
			data_cluster_dict[output_label[i]].append(self.data.index.tolist()[i])
            
		if isPlot and X.shape[1] == 1:
			x = []
			for i in range(len(data_cluster_dict)):
				x.extend(data_cluster_dict[i])

			self.data = self.data.reindex(x)
			self.data = self.data.reset_index(drop=True)

			j = 0
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')
			plot_color = list('bgrcmykbgrcmykbgrcmykbgrcmyk')
			for i in range(len(data_cluster_dict)):
				cluster_center = cluster_centers[i]
				plt.plot(self.data.loc[j:j+len(data_cluster_dict[i])-1], color=plot_color[i], marker=plot_shape[i], \
					linestyle='', linewidth=2.0)
				plt.plot(np.mean(list([kk for kk in range(j,j+len(data_cluster_dict[i])-1)])), cluster_center[0],\
                    'o', markerfacecolor=plot_color[i],
						markeredgecolor='k', markersize=14)
				j += len(data_cluster_dict[i])
			plt.show()
		else:
			colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')

			for k, col in zip(list(set(output_label)), colors):
				my_members = output_label == k
				cluster_center = cluster_centers[k]
				plt.plot(X[my_members, 0], X[my_members, 1], col + plot_shape[k])
				plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
						markeredgecolor='k', markersize=14)
			plt.show()
		
		print(" [INFO] end ")

		return output_label, data_cluster_dict

	def combined(self, cluster_num=None, isPlot=False):
		"""Combined clustering
		
		Args:
			cluster_num: Input "int", manually determine number of clusters
			isplot: Plot or not
		Explanation
			- utilize Mean-Shift method to make initial centroids of K-Means
		"""

		print(" [INFO] starting meanshift-kmeans combind method ")

		self.data_preprocessing()
		print(" [INFO] data pre-processing done ")

		ind = 0
		while self.data.shape[1] > self.data.shape[0]:

			self.dimension_reduction(cont_rate=self.cont_rate_list[ind])

			if self.data.shape[1] > self.data.shape[0]:
				print("  ----  reduced data dimension larger then data amount, alter another contribution rate ")
				ind += 1

			else:
				print("  ----  reduced data dimension smaller then data amount ")
				break

		print(" [INFO] dimensionality reduction done ")

		X = np.array(self.data)
		bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10000, random_state=42, n_jobs=2) 
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		ms.fit(X)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)

		dict_1 = []
		for i in labels:
			dict_1.append((i,str(i)))

		np.random.seed(42)
		data = np.array(cluster_centers) 

		if n_clusters_ > 2:
			if not cluster_num:

				range_n_clusters = list(range(2,n_clusters_))
				silhouette_avg = []

				for n in range_n_clusters:
					estimator = KMeans(init='random', n_clusters=n, max_iter=1000, n_init=10)
					cluster_labels = estimator.fit_predict(data)
					silhouette_avg.append(silhouette_score(data, cluster_labels))
					print("  ----  For n_clusters = " + str(n) + ", the average silhouette_score is : " + str(silhouette_avg[-1]) + ".")

				# use K-Means to cluster Mean-Shift centroids
				n_digits = range_n_clusters[silhouette_avg.index(max(silhouette_avg))] #
				print("  ----  Choose n_clusters = " + str(n_digits) + " with max average silhouette score as final clusters number.")
				kmeans = KMeans(init='random', n_clusters=n_digits, max_iter=1000, n_init=10)
				output_label = kmeans.fit_predict(data) 
				cluster_centers = kmeans.cluster_centers_

				# use clustered Mean-Shift centroids as initial centroids of K-Means
				kmeans = KMeans(init=cluster_centers, n_clusters=len(cluster_centers), max_iter=1000)
				output_label = kmeans.fit_predict(self.data) 

			else:
				# use K-Means to cluster Mean-Shift centroids
				kmeans = KMeans(init='random', n_clusters=min(cluster_num,n_clusters_), max_iter=1000, n_init=10)
				output_label = kmeans.fit_predict(data) 
				cluster_centers = kmeans.cluster_centers_

				# use clustered Mean-Shift centroids as initial centroids of K-Means
				kmeans = KMeans(init=cluster_centers, n_clusters=len(cluster_centers), max_iter=1000)
				output_label = kmeans.fit_predict(self.data) 

		else:
			output_label = labels

		print(" [INFO] end ")

		if isPlot and X.shape[1] >= 2:
#			plt.figure(2)
#			plt.clf()
			colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')

			for k, col in zip(list(set(output_label)), colors):
				my_members = output_label == k
				cluster_center = cluster_centers[k]
				plt.plot(X[my_members, 0], X[my_members, 1], col + plot_shape[k])
				plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
						markeredgecolor='k', markersize=14)

			plt.show()

		dict_2 = {}
		for i in range(len(output_label)):
			dict_2.update({str(i):output_label[i]})

		new_labels = []
		for i in range(len(dict_1)):
			new_labels.append(dict_2[dict_1[i][1]])
		new_labels = np.array(new_labels).astype(int)

		labels_unique = np.unique(new_labels)
		n_clusters_ = len(labels_unique)

		new_labels_pd = pd.DataFrame(new_labels,index=self.data.index.tolist(),columns=['labels'])
		data_cluster_dict = {}
		output = pd.DataFrame(index=list(set(new_labels_pd['labels'])),\
							columns=self.data.columns)

		for label in set(new_labels_pd['labels']):
			a = list(new_labels_pd[(new_labels_pd['labels'] == label)].index)
			tmp = self.data.loc[a]
			for col in list(self.data.columns):
				output.loc[label][col] = tmp[col].mean()
				data_cluster_dict.update({label:a})

		if isPlot and X.shape[1] == 1:
			x = []
			for i in range(len(data_cluster_dict)):
				x.extend(data_cluster_dict[i])

			self.data = self.data.reindex(x)
			self.data = self.data.reset_index(drop=True)

			j = 0
			plot_shape = list('.^*o+dp.^*o+dp.^*o+dp^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.^*o+dp.')
			plot_color = list('bgrcmykbgrcmykbgrcmykbgrcmyk')
			for i in range(len(data_cluster_dict)):
				plt.plot(self.data.loc[j:j+len(data_cluster_dict[i])-1], color=plot_color[i], marker=plot_shape[i], \
					linestyle='', linewidth=2.0)
				j += len(data_cluster_dict[i])
			plt.show()

		return output, data_cluster_dict

	def _plot_clusters(self, X: np.ndarray, labels: np.ndarray, 
					  centers: np.ndarray) -> None:
		"""Helper method for cluster visualization"""
		colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
		markers = cycle('.^*o+dp')
		
		for k, col in zip(np.unique(labels), colors):
			mask = labels == k
			plt.plot(X[mask, 0], X[mask, 1], 
					f"{next(col)}{next(markers)}")
			plt.plot(centers[k, 0], centers[k, 1], 'o',
					markerfacecolor=col, markeredgecolor='k', 
					markersize=14)


if __name__ == '__main__':
	centers = [[1, 1], [-1, -1], [1, -1]]
	X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
	                            random_state=0)
	obj = MSK(X,Standardized_method=['minmax',0,1],reduced_n_dim=None,reduced_method='PCA')
	# obj.mean_shift(isPlot=True)
	# obj.kmeans(n_cluster_list=[4],isPlot=True)
	obj.combined(cluster_num=3,isPlot=True)
