import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
pd.options.mode.chained_assignment = None
pd.set_option('max_columns',6)
pd.set_option('max_colwidth',15)

# To patch:
# * Make scatter_plot able to place a legend in a good way

class KMeans_and_Cluster:
    # structure for breaking a set of points into chunks using k-means, then
    # recombining them by single-link clustering into a smaller number of clusters
    # throughout, CHUNK means one of the original k-means groups, while
    # CLUSTER means one of the intermediate or final larger groupings
    def __init__(self, points, **kwargs):
        # points: an n x 2 numpy array in which the first column is the x-coord and second column is y-coord
        if points.shape[1] != 2:
            raise TypeError('points must be of shape (n,2)')
        self.points = pd.DataFrame(points, columns=['x','y'])
        self.points.reset_index(inplace=True)
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

    def get_points(self, *args, output='dataframe'):
        # returns x and y columns of self.points, with any add'l columns requested
        # via args
        if output not in ['dataframe','arrays']:
            raise ValueError("output must be 'dataframe' or 'arrays'.")
        diff = set(args).difference({'include','chunk','cluster','min_distance'})
        if diff != set():
            raise ValueError(str(diff)+" is/are not available as column(s).")
        if output == 'dataframe':
            return self.points[['x','y']+list(args)]
        else:
            return (np.array(self.points[['x','y']]),
                    *(np.array(self.points[arg]) for arg in args))
    
    def ut_distance(self, ind1, ind2):
        if ind1==ind2:
            return 0
        if ind1 > ind2:
            ind1, ind2 = ind2, ind1
        try:
            return self.pdist_array[int((self.n-1)*ind1-ind1*(ind1-1)/2+ind2-ind1-1)]
        except:
            print(ind1, ind2)

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

    def chunkify(self, k_chunks, output=None, verbose=False, random_state=None):
        # do the k-means clustering, then compute pairwise intercluster distances
        # note that this will overwrite any earlier k-chunk attempt
        if output not in ['dataframe','arrays',None]:
            raise ValueError("output must be 'dataframe', 'arrays', or None.")
        self.k_chunks = k_chunks
        included_points = self.points.loc[self.points.include]
        ks = KMeans(n_clusters=k_chunks, random_state=random_state).fit(np.array(included_points[['x','y']])).labels_
        if verbose:
            print('K-Means clustering complete.')
        self.points.loc[self.points.include,'chunk'] = ks
        self.points.loc[self.points.include,'cluster'] = ks
        self.chunk_dict = {t:included_points.loc[self.points.chunk==t].index
                           for t in range(k_chunks)}
        self.chunk_distances = dict()
        # speedup: do the same trick here as in prefilter (go through pdist once)
        distances_done = 0
        total_distances = int(k_chunks*(k_chunks-1)/2)
        for j in range(k_chunks):
            for i in range(j):
                i_index, j_index = self.chunk_dict[i], self.chunk_dict[j]
                M = min(self.ut_distance(s,t) for s in i_index for t in j_index)
                self.chunk_distances[i,j] = M
                distances_done += 1
                if distances_done % 100 == 0 and verbose:
                    print('Computed %d of %d distances.' % (distances_done, total_distances))
        self.all_clusters = set(range(k_chunks))
        if output == 'dataframe':
            return self.points.loc[self.points.include][['x','y','chunk']]
        elif output == 'arrays':
            return (np.array(self.points.loc[self.points.include][['x','y']]),
                    np.array(self.points.loc[self.points.include]['chunk']))

    def condense(self, c1, c2):
        # agglomerate clusters c1 and c2, eliminating the higher-numbered one
        # not intended to be public, but no real reason why user couldn't call it
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

    def clusterify(self, final_clusters, output='dataframe', verbose=False):
        # do single-link clustering on the k-chunks until there are n=final_clusters of them
        # return the original (x,y)-coordinates labeled with their clusters
        # will overwrite previous clustering, if there was one
        if self.k_chunks == np.nan:
            raise RuntimeError('Must perform chunkify before clusterify.')
        if output not in ['dataframe','arrays',None]:
            raise ValueError("output must be 'dataframe', 'arrays', or None.")
        if final_clusters > self.k_chunks:
            raise ValueError('Must have at least as many chunks as clusters.')
        self.points.cluster = self.points.chunk
        self.all_clusters = set(range(self.k_chunks))
        self.cluster_distances = self.chunk_distances.copy()
        self.num_clusters = final_clusters
        while len(self.all_clusters) > self.num_clusters:
            pairs_to_use = [key for (key,value) in self.cluster_distances.items()
                            if value==min(self.cluster_distances.values())]
            pairs_to_use.sort(key=lambda pair: (-pair[1],-pair[0]))
            for (c1,c2) in pairs_to_use:
                if verbose:
                    print('Condensed clusters %d and %d at distance %.4f' %
                          (int(c1),int(c2),self.cluster_distances[c1,c2]))
                self.condense(c1,c2)
        if output == 'dataframe':
            return self.points.loc[self.points.include][['x','y','cluster']]
        elif output == 'arrays':
            return (np.array(self.points.loc[self.points.include][['x','y']]),
                    np.array(self.points.loc[self.points.include]['cluster']))

    def auto_clusterify(self):
        # will eventually fold this in as an option for clusterify
        # idea is to continue clustering until you're left with "a few" "big" clusters
        # probably based on condensing up through kth percentile of available distances
        # as measured by chunk_distances.values()
        pass

    def chunk_and_clusterify(self, k_chunks, final_clusters):
        self.chunkify(k_chunks)
        self.clusterify(final_clusters)

    def scatter_plot(self, size=0.1, hue='cluster', no_legend=True):
        if hue not in ['cluster','chunk','include']:
            raise ValueError("hue must be 'cluster', 'chunk', or 'include'.")
        if hue != 'include':
            self.points[hue+'_cat'] = self.points[hue].astype('str')+'a'
            included_points = self.points.loc[self.points.include]
            hue = hue+'_cat'
        else:
            included_points = self.points
        if no_legend:
            return sns.scatterplot(x=included_points.x, y=included_points.y,
                               hue=included_points[hue], size=size, legend=None)
        else:
            return sns.scatterplot(x=included_points.x, y=included_points.y,
                               hue=included_points[hue], size=size)

