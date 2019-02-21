About This Repository
======================

This repository is an implementation of the "k-means and cluster" strategy outlined in a few places for clustering data. The idea is to split your data up into a few dozen chunks using k-means clustering, then recombine those chunks into just a few larger clusters using single-linkage clustering. It's especially useful for data that comes in discrete, non-convex blobs--my original use for it was to distinguish New York City GPS coordinates as being in or out of Manhattan. I couldn't find an implementation in the ```sklearn.cluster``` or ```scipy``` packages, so I made my own (which is not to say that one doesn't exist there already; in fact I'd be a little surprised if it didn't).

Contents
---------

Ideally, this repository contains the following:

* **kmcluster.py**, which contains all the necessary code.
* **Demo.ipynb**, which demonstrates all the useful features.
* **documentation** of some kind.

Right now, there is no detailed documentation (outside of some explanatory comments in the code), but the Demo should make the code usable.
