import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
pd.options.mode.chained_assignment = None
pd.set_option('max_columns',12)
pd.set_option('max_colwidth',15)

# To-Do List:
# * Add option in prefiltering for doing a prefilter just for that distance,
# meaning we don't actually fill in the column 'min_distance' (just go through
# the pdist array number by number, checking off points when they qualify)
# * Add method for automatically showing a scatterplot with appropriate parameters
# (with a few built-in palettes to choose from if you're lazy)




class KMeans_and_Cluster:
    # structure for breaking a set of points into chunks using k-means, then
    # recombining them by single-link clustering into a smaller number of clusters
    # throughout, CHUNK means one of the original k-means groups, while
    # CLUSTER means one of the intermediate or final larger groupings
    def __init__(self, points, **kwargs):
        # points: a DF in which the first column is the x-coord and second column is y-coord
        # can have other columns (so long as their names don't overlap with ones defined here),
        # they'll just be carried along and ignored
        self.points = points.rename(columns={points.columns[0]:'x', points.columns[1]:'y'})
        self.points['chunk'] = np.nan
        self.points['cluster'] = np.nan
        self.points['min_distance'] = np.nan
        self.points['include'] = True
        self.filter_radius = np.nan
        self.k_chunks = np.nan
        self.num_clusters = np.nan
        self.n = len(self.points)
        self.pdist_array = pdist(np.array(self.points[['x','y']]))
        self.min_distance_calc = False
        if 'prefilter' in kwargs:
            self.prefilter(kwargs['prefilter'])

    def ut_distance(self, ind1, ind2):
        if ind1==ind2:
            return 0
        if ind1 > ind2:
            ind1, ind2 = ind2, ind1
        return self.pdist_array[int((self.n-1)*ind1-ind1*(ind1-1)/2+ind2-ind1-1)]

    def prefilter(self,dist):
        # un-include any points that are more than a distance of dist away
        # from all other points in the set
        # overwrites any previous prefiltering
        # will calculate the min_distances if they haven't been computed already
        self.filter_radius = dist
        self.points['include'] = True
        if not self.min_distance_calc:
            dist_dict = {i:np.inf for i in range(self.n)}
            col_counter = 1
            row_counter = 0
            for d in self.pdist_array:
                if d < dist_dict[col_counter]:
                    dist_dict[col_counter] = d
                if d < dist_dict[row_counter]:
                    dist_dict[row_counter] = d
                if col_counter == self.n-1:
                    col_counter = row_counter+2
                    row_counter += 1
                else:
                    col_counter += 1
            self.points['min_distance'] = [value for (key,value) in sorted(dist_dict.items())]
        self.points.loc[(self.points.min_distance > self.filter_radius), 'include'] = False


    def chunkify(self, k_chunks):
        # do the k-means clustering, then compute pairwise intercluster distances
        # note that this will overwrite any earlier k-chunk attempt
        self.k_chunks = k_chunks
        included_points = self.points.loc[self.points.include]
        ks = KMeans(n_clusters=k_chunks).fit(np.array(included_points[['x','y']])).labels_
        self.points.loc[self.points.include,'chunk'] = ks
        self.points.loc[self.points.include,'cluster'] = ks
        self.chunk_dict = {t:included_points.loc[self.points.chunk==t].index
                           for t in range(k_chunks)}
        self.cluster_distances = dict()
        for j in range(k_chunks):
            for i in range(j):
                i_index, j_index = self.chunk_dict[i], self.chunk_dict[j]
                M = min(self.ut_distance(s,t) for s in i_index for t in j_index)
                self.cluster_distances[i,j] = M
        self.all_clusters = set(range(k_chunks))

    def condense(self, c1, c2):
        # agglomerate clusters c1 and c2, eliminating the higher-numbered one
        if c1 not in self.all_clusters:
            raise ValueError('Cluster %d does not exist or is already agglomerated.' % (c1))
        if c2 not in self.all_clusters:
            raise ValueError('Cluster %d does not exist or is already agglomerated.' % (c2))
        if c1 > c2:
            c1, c2 = c2, c1
        if c1 == c2:
            return
        self.points.cluster.replace(c2,c1, inplace=True)
        self.all_clusters.remove(c2)
        for c in self.all_clusters:
            if c < c1:
                self.cluster_distances[c,c1] = min(self.cluster_distances[c,c1], self.cluster_distances[c,c2])
                del self.cluster_distances[c,c2]
            elif c1 < c < c2:
                self.cluster_distances[c1,c] = min(self.cluster_distances[c1,c], self.cluster_distances[c,c2])
                del self.cluster_distances[c,c2]
            elif c > c2:
                self.cluster_distances[c1,c] = min(self.cluster_distances[c1,c], self.cluster_distances[c2,c])
                del self.cluster_distances[c2,c]
        del self.cluster_distances[c1,c2]

    def clusterify(self, final_clusters):
        # do single-link clustering on the k-chunks until there are n=final_clusters of them
        # return the original (x,y)-coordinates labeled with their clusters
        if final_clusters > self.k_chunks:
            raise ValueError('Must have at least as many chunks as clusters.')
        self.num_clusters = final_clusters
        while len(self.all_clusters) > self.num_clusters:
            pairs_to_use = [key for (key,value) in self.cluster_distances.items()
                            if value==min(self.cluster_distances.values())]
            pairs_to_use.sort(key=lambda pair: (-pair[1],-pair[0]))
            for (c1,c2) in pairs_to_use:
                self.condense(c1,c2)
        return self.points.loc[self.points.include][['x','y','cluster']]

    def chunk_and_clusterify(self, k_chunks, final_clusters):
        self.chunkify(k_chunks)
        self.clusterify(final_clusters)

    def scatter_plot(self, size=0.1, hue='cluster', palette='auto'):
        if hue not in ['cluster','chunk','include']:
            raise ValueError("hue must be 'cluster', 'chunk', or 'include'.")
        if palette == 'auto':
            pass
            #figure out what to do
        included_points = self.points.loc[self.points.include]
        return sns.scatterplot(x=included_points.x, y=included_points.y, hue=included_points[hue], size=size)


