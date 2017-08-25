#
import numpy
import scipy
import scipy.optimize
import sklearn
import sklearn.neighbors
import pylab as plt
import itertools
import random
#
#import operator
import functools
import multiprocessing as mpp
#
import numpy
import math
import pylab as plt
import datetime as dtm
import time
#

#import pandas

colors_ = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


# write a script to get one nn cluster... in fact, let's go ahead and start with a class structure,
# which will help to avoid becoming overly-procedural down the road.
#
# so i'm trying to make this smart and modular, but maybe it's just not. maybe we need to just be brute and
# a bit more programmatically old-school...
#
# and on that note, for this to be fast and scalable to larger dimensions, the best approach is to keep one
# list of nodes and manage clusters as lists of indices referencing that list. we'll start with simple lists
# or arrays and look at pandas DF later..
#
class NN_cluster_finder(object):
	#
	#def __init__(self, node_points, nn_seeds=None, n_neighbors=2, nn_radius=1.0, nn_dist_threshold_factor=.8):
	def __init__(self, node_points, nn_seeds=None, n_neighbors=2, nn_radius=None, dist_cutoff_nn_factor=None,  dist_cutoff=None, nn_algorithm='auto', n_cpu=1):
		# ... and we'll move parameters into the call signature as we go...
		n_cpu = (n_cpu or mpp.cpu_count())	# this can also be n_cpu=-1 to get all processors in sklearn.neighbors.NearestNeighbors class.
		nn_algorithm = (nn_algorithm or 'auto')
		working_nodes = []
		sk_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=max(2, n_neighbors), radius=nn_radius, algorithm=nn_algorithm, n_jobs=n_cpu)
		#
		# now, we need some tuning parameters. let's learn about our NN pairs. what is the distribution
		# of NN distances, etc. we will use this to determine when a cluster ends... and other stuff.
		sk_nbrs.fit(node_points)
		nn_distances, nn_indices = sk_nbrs.kneighbors(node_points, return_distance=True)

		nn_mean_distances = [nn_mean(rw) for rw in nn_distances]	
		#
		# distance cutoff bits:
		if not dist_cutoff is None and dist_cutoff<0:
			dist_cutoff = sorted(nn_mean_distances)[dist_cutoff]
		else:
			#
			dist_cutoff = (dist_cutoff or  dist_cutoff_nn_factor*sorted(nn_mean_distances)[min(int(dist_cutoff_nn_factor*len(nn_distances)), len(nn_mean_distances)-1)])
		print('initializing with dist_cutoff = {}'.format(dist_cutoff))
		#
		#nn_radius = (nn_radius or 5.*sorted(nn_mean_distances)[int(.9*len(nn_mean_distances))])
		nn_radius = (nn_radius or 5*max(nn_mean_distances))
		#dist_cutoff = .25
		#
		# this will require some more delicate handling. on the one hand, we can give the 'find_clusters'
		# angorithm an arbitrary set of seeds; on the other hand, we want to use unique seeds from the 
		# nodes data set, which is an emergent condition...
		#
		# note: seeds should be actual point vertices, and for cases where we find all clusters,
		# we might just forego this in the initialization.
		nn_seeds = (nn_seeds or numpy.copy(node_points))
		#
		self.__dict__.update(locals())
		#
	#
	def get_nn_to_cluster(self, kernel_points, node_points=None, sk_nbrs=None, return_dists=True, do_train=False):
		# return the single point closest to a point in a cluster.
		# 'clusters' will, at least initially, be collections of node indices.
		# this is meant to be a simple wrapper and (for now) assumes by default that sk_nbrs is trained.
		#
		node_points = (node_points or self.node_points)
		sk_nbrs = (sk_nbrs or self.sk_nbrs)
		#if do_train: sk_nbrs.fit(node_points)
		if do_train:
			# train the sk_nbrs nn-finder. see note above: if n_nn > n_samples, fix it:
			if len(node_points)<sk_nbrs.n_neighbors:
				# this will break; we need more samples than neighbors, so let's just downgrade (this should just occur for corner cases).
				# so fix it or break it? let's fix it here; we should try 'break' options from the calling side.
				#
				sk_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=len(node_points), radius=self.nn_radius, algorithm=self.nn_algorithm)
			sk_nbrs.fit(node_points)
		#
		# we might want to move this
		#self.sk_nbrs.fit(node_points)
		#nns = self.sk_nbrs.kneighbors(kernel_points, return_distance=True)
		dists, nn_indices = sk_nbrs.kneighbors(numpy.atleast_2d(kernel_points), return_distance=True)
		#
		# nns ~ [[d1, d2,], [] ...], [[j1, j2,...', []...]] of NNs between kernel_points and points in node_points.
		#
		# this part is a little bit tricky. we might operate a multiple NN algorithm; we might be looking at
		# kernel_points that are a subset of node_points, etc., so we need to be smart and flexible.
		#
		# get index of closest nn pair. we could sort, but we only need one element, so it's faster (i think...)
		# to just spin through the dists. the main question is: is it faster to sort or to manage the copy/recalc
		# required if we skip the sort. i think it's actually faster to sort with an enumerate...
		# we can bench this in a later version.
		#min_dist_index = 0
		#min_dist = numpy.mean([x for x in dists[0] if x!=0])   # use the mean NN distance, exclude self-distance=0
		#for k,dd in dists:
		#	this_dist = 
		#	if numpy.mean([x for x in dd if x!=0])<min_dist:
		#		min_dist_index=k
		#		min_dist = 
		#print('dists: ', dists)
		#j_nn, dx_nn = sorted(enumerate(dists),key=lambda rw: numpy.mean([x for x in rw[1] if x!=0.]))[0]
		j_nn, dx_nn = sorted(enumerate(dists),key=lambda rw: numpy.mean(nn_mean(rw[1])))[0]
		# this is the index and distances of the point from node_points that is closest to a point in
		# kernel_points.
		
		#
		# note: for optimization this return stmt. can be wrapped into the sort() stmt. above.
		if return_dists:
			return {'index': nn_indices[j_nn], 'dists': dx_nn}
		else:
			return nn_indices[j_nn]
		#
	#
	def get_nn_cluster(self, kernel_points, node_points=None, dist_cutoff=None, copy_nodes=True):
		node_points = (node_points or self.node_points)
		dist_cutoff = (dist_cutoff or self.dist_cutoff)
		#
		#if copy_nodes:
		#	self.working_nodes = node_points.copy()
		#else:
		#	self.working_nodes = node_points
		#working_nodes = self.working_nodes
		#
		# TODO: i think the cluster initialization bit can be worked into the regular loop...
		#
		if copy_nodes:
			working_nodes = node_points.copy()
		else:
			working_nodes = node_points
		self.working_nodes = working_nodes
		#
		nns = self.get_nn_to_cluster(kernel_points = kernel_points, node_points=working_nodes,
									 return_dists=True, do_train=True)
		dx = nn_mean(nns['dists'])
		#
		# cluster needs to be a list of vertices, not indices of vertices, because the indices
		# are going to change. otherwise, we need to impose a more sophisticated indexing system
		# an probably rewrite the NN finder all together.
		#
		#cluster=[working_nodes[j] for j in set(nns['index']) if not working_nodes[j] in numpy.atleast_2d(kernel_points)]
		cluster=[working_nodes[j] for j in set(nns['index']) ]
		#
		#if len(cluster)==0: return cluster
		# now, remove those elements from working_nodes. doing this reliably and efficiently is a bit
		# tricky. we probably want to use an indexed structure, but then we have to update the index...
		# if we restrict this to 1 element growth, we can just split the table, but that comes with its
		# own problems. we can also do an ordered removal or split/connect where we update the indices
		# with each action. the main idea is to avoid copying the data and/or spinning the whole table.
		# for now, try this, and we'll figure out optimization later:
		# the simlest syntax is:
		# working_nodes = [rw for j,rw in enumerate(working_nodes) if not j in nns['index']]
		# ... but i think this will be faster... but maybe not.
		# "pop" nns['index'] from working_nodes
		working_nodes=pop_list(working_nodes, nns['index'])
		#
		#print('dx/<dx>: {}/{}'.format(dx, dist_cutoff))
		#
		while dx<=dist_cutoff and len(working_nodes)>0:
			#if len(cluster)==0: continue
			#print('** ', cluster)
			nns = self.get_nn_to_cluster(kernel_points = cluster, node_points=working_nodes,
										return_dists=True, do_train=True)
			#print('nns: ', nns)
			# nn_mean() gets the mean value of a set of numbers (presumably nn values) but excludes zero-valued entries. the assumption is that this is a
			#  standard return for a nn algorithm given a seed value included in the training set.
			dx = nn_mean(nns['dists'])
			#
			#cluster += [working_nodes[j] for j in set(nns['index']) if not (working_nodes[j] in numpy.atleast_2d(cluster)
			#															   or working_nodes[j] in numpy.atleast_2d(kernel_points))]
			cluster += [working_nodes[j] for j in set(nns['index']) if not (working_nodes[j] in numpy.atleast_2d(cluster) )]
			# get new kernel points, then trim up working_nodes:
			working_nodes = pop_list(working_nodes, nns['index'])
			self.working_nodes = working_nodes
			#
			#print('dx: ', dx, nns)
			if dx>dist_cutoff:
				#print('breaking...')
				break
			#
			
		#
		return cluster
	#
	#def get_self_nn_clusters(self, node_points=None, dist_cutoff=None, dist_cutoff_nn_factor=None, n_neighbors=None, do_copy=True, min_clust_size=None):
	def get_self_nn_clusters(self, node_points=None, dist_cutoff=None, n_neighbors=None, do_copy=True, min_clust_size=None):
		#
		# TODO: work on the dist_cutoff_factor bit. really, we need 2 parameters: dist_cutoff_fraction and dist_cutoof factor, and our cutoof is like:
		# dist_cutoff_factor*nn_distances[int(len*dist_cutoff_fraction)]
		# ... and maybe keep it simple here. we can add complex default handlers to the class __init__, but i think it gets confusing if we repeat it here.
		# we're better off considering the class object disposable. note also, given the newer, simpler model for this algorithm, it might be better to just
		# pull it out of the class object and go procedural with it -- especially if we need to mpp it later.
		#
		# so the idea is to guess than n*max_nn_dist is a new cluster. but we don't want the call signature or the parameter handling to get too complicated.
		# this, of course, can also be handled offline (estimate the cutoff distance first; then call this function).
		#
		node_points = (node_points or self.node_points)
		n_neighbors = (n_neighbors or self.n_neighbors)
		#
		# make a copy/reff to working data (nodes):
		if do_copy:
			working_nodes = node_points.copy()
		else:
			working_nodes = node_points
		#
		# first, get all NN sequences; sort them by closest:
		# get a nn initializer. don't use the self.one because we might set that to nn=1; we need nn>1 here.
		sk_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=max(2, self.n_neighbors), radius=self.nn_radius, n_jobs=self.n_cpu, algorithm=self.nn_algorithm)
		sk_nbrs.fit(node_points)
		#
		seeds = sorted(list(zip(*sk_nbrs.kneighbors(numpy.atleast_2d(working_nodes), return_distance=True))), key=lambda rw: nn_mean(rw[0]))
		seed_indices = seeds[0]
		#
		#dist_cutoff_nn_factor = (dist_cutoff_nn_factor or self.dist_cutoff_nn_factor)
		#if dist_cutoff is None and not dist_cutoff_nn_factor is None:
		#	# try to guess a good distance cutoff factor:
		#	#dist_index = sorted([nn_mean(rw[0]) for rw in seeds])
		#	# dist_cutoff = dist_cutoff_nn_factor*dist_index[int(.9*len(dist_index))]
		#	dist_cutoff = dist_cutoff_nn_factor*sorted([nn_mean(rw[0]) for rw in seeds])[int(.9*len(seeds))]
		#
		dist_cutoff = (dist_cutoff or self.dist_cutoff)
		#
		# first cluster is tightest nn pair:
		seed_indices = seeds[0][1]
		#
		#print(seeds[0], seed_indices)
		# remove these indices; put them into the first cluster:
		clusters = [[working_nodes[j] for j in seed_indices]]
		#print('clustsers: ', clusters)
		#
		# now, we have our first cluster. remove the seed indices from working nodes. make an index of the nn_indices to sort working_nodes, so we can pull them off one at a time
		# as we make the cluster.
		# TODO: look up the behavior and optimization of pop(), for all the obvious stuff.
		#
		working_nodes = [working_nodes[rw[1][0]] for rw in seeds if not rw[1][0] in seed_indices]
		#
		# now, make a new neighbors handler with the proper n_neighbors (=1 for pure operations...):
		sk_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors, radius=self.nn_radius, n_jobs=self.n_cpu, algorithm=self.nn_algorithm)
		#
		kk=0
		while len(working_nodes)>0:
			kk+=1
			# assign each node to a cluster...
			#
			# first, train the sk_nbrs handler on the revised nodes:
			sk_nbrs.set_params(n_neighbors=min(len(working_nodes), n_neighbors))
			#
			sk_nbrs.fit(working_nodes)
			#
			nns = sorted(list(zip(*sk_nbrs.kneighbors(numpy.atleast_2d(clusters[-1]), return_distance=True))), key=lambda rw: nn_mean(rw[0]))
			#
			dx = nn_mean(nns[0][0])
			if dx>dist_cutoff:
				#
				#if (not min_clust_size is None) and len(clusters[-1])<min_clust_size: clusters = clusters[:-1]
				clusters += [[working_nodes.pop(0)]]
			else:
				#
				clusters[-1] += [working_nodes[x] for x in nns[0][1]]
				working_nodes = pop_list(working_nodes,nns[0][1])
			
		return clusters
		
	#
		#
	#def nn_mean(self, X, strip_zero=True):
	#	# note: in production, move this function to the module level.
	#	return numpy.mean([x for x in X if not x==0.])

def nn_mean(X, strip_zero=True):
	# note: in production, move this function to the module level.
	return numpy.mean([x for x in numpy.atleast_2d(X) if not x==0.])

def get_Spectral_Clustering(XY, n_clusters=5, sc_affinity='nearest_neighbors', **kwargs):
	'''
	# wrapper around an sklearn SpectralClustering() cluster finder.
	'''
	#
	sc = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity=sc_affinity,**kwargs)
	lbls = sc.fit_predict(XY)
	return clusters_from_labels(XY,lbls)
def get_sklearn_clustering(XY, clust_class=None, **kwargs):
	# general wrapper for sklearn cluster classes. 
	#
	clust_class = (clust_class or sklearn.cluster.SpectralClustering)
	#print('cc: ', clust_class)
	#
	sc=clust_class(**kwargs)
	lbls = sc.fit_predict(XY)
	return clusters_from_labels(XY,lbls)

def get_Spectral_Clustering_analysis(XY, n_clusters=5, fignum=None, lsq_metric_baseline_exponent=0.5, sc_affinity='nearest_neighbors', return_keys=['clusters', 'stacked', 'lsq'], hi=.9, lo=.1,**kwargs):
	'''
	# TODO: this should probably be modularized a bit. this is a sort of all-in-one and should be broken up for a sw module.
	# NOTE: SpectralDecomposition can return different clusters every time you run it in its default config. to get the same set of clusters, 
	# use the random_state parameter... though i'm not sure exatly how. i think you need some sort of random_state() instance; just setting it to an integer values
	# doesn't quite cut it.
	#
	# @sc_affinity: affinity= ‘nearest_neighbors’, ‘precomputed’, ‘rbf’
	# a wrapper around sklearn.clusters.SpectralClustering(). return one or more of (by specifying return_keys): clusters, 
	#"stacked" clusters (mean-subtracted) and a least-squares based 'goodness' metric
	# @lsq_metric_baseline_exponent: exponent for least-squares metric. the idea is to include the baseline over which the lsq is taken, aka:
	#   goodness = chi_sqr/( (dx**2 + dy**2)**lsq_metric_baseline_exponent )
	'''
	#n_clusters=8
	#XY = XY.copy()
	return_keys = (return_keys or ['clusters', 'stacked', 'lsq'])
	f_lin = lambda x,a,b: a + b*x
	#
	#sc = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity=sc_affinity)
	#lbls = sc.fit_predict(XY)
	##
	## consolidate clusters (later on, there may be a way to better optimize this):
	#clusters = cluster_tools.clusters_from_labels(XY,lbls)
	clusters = get_Spectral_Clustering(XY=XY, n_clusters=n_clusters, sc_affinity=sc_affinity)
	#
	XY_stacked = stack_clusters(clusters)
	#lsq = list(optimizers.lsq_XY(XY_stacked))
	lsq = list(numpy.linalg.lstsq([[1.]+rw[:-1] for rw in XY_stacked], [rw[-1] for rw in XY_stacked]))
	#
	# TODO: new metric idea: take PCA; maximize the ratio of the eigenvalues. what does that look like in higher dimensions? minimize the combined
	#   length of all but the largest e-val (aka, minimize lengths of orthogonal axes)?
	# add a metric (chi-sqr/baseline):
	# simple 2D version:
	#dx = max(XY_stacked, key=lambda rw: rw[0])[0]-min(XY_stacked, key=lambda rw: rw[0])[0]
	#dy = max(XY_stacked, key=lambda rw: rw[1])[1]-min(XY_stacked, key=lambda rw: rw[1])[1]
	# more generally:
	dx_var = 0.
	for k in range(len(XY_stacked[0])):
		xx = sorted([rw[k] for rw in XY_stacked])
		dx_var += (xx[int(min(1., hi*len(xx)))] - xx[int(lo*len(xx))])**2.
	dx_var = dx_var**lsq_metric_baseline_exponent
	#dx_var = numpy.sqrt(dx**2. + dy**2.)
	#
	#lsq.append([x/numpy.sqrt(dx**2 + dy**2) for x in lsq[1]])
	lsq.append([x/dx_var for x in lsq[1]])
	#
	if not fignum is None:
		fg=plt.figure(fignum)
		ax1 = plt.gca()
		ax1.set_title('SpectralClustering (n_clusters={})'.format(n_clusters))
		ax1.scatter(*zip(*XY_stacked), marker='.')
		X = numpy.array([min(XY_stacked,key=lambda rw:rw[0])[0], max(XY_stacked,key=lambda rw:rw[0])[0]])
		#
		#print(X)
		ax1.plot(X, f_lin(X,*lsq[0]), color='r', lw=2., ls='-', label='lsq: {}'.format(lsq[0]))
		ax1.legend(loc=0)
	#
	#return lsq
	return_vals = {'clusters': clusters, 'stacked':XY_stacked, 'lsq':lsq}
	#
	return {key:return_vals[key] for key in return_keys}

def clusters_from_labels(nodes, labels, min_clust_size=None, as_dict=False):
	# ... and we might also add options for min number of clusters, or we can just do that offline.
	#
	if as_dict:
		my_clusters = {j:[] for j in set(labels)}
	else: 
		my_clusters = [[] for _ in list(set(labels))]  # could also be a dict with the index stated explicitly.
	#
	for k_index, xy in zip(labels, nodes):
		my_clusters[k_index] += [xy]
	#
	# and (maybe) spin off small clusters:
	size_threshold = max(0, (min_clust_size or 0) )
	if as_dict:
		my_clusters = {key:val for key,val in my_clusters.items() if len(val)>size_threshold}
	else:
		my_clusters = [clust for clust in my_clusters if len(clust)>size_threshold]
	#
	return my_clusters
	#

def pop_list(X,pops):
	new_nodes = []
	# clean it up a bit...
	pops = sorted(list(set(pops)))
	j=0
	for j_c in sorted(pops + [len(X)]):
		#print('j_c: {}'.format(j_c))
		new_nodes += X[j:j_c]
		j=j_c + 1
	return new_nodes
#
def nn_mean(X, strip_zero=True):
	# note: in production, move this function to the module level.
	if numpy.all(numpy.array(X)==0.):
		return 0.
	else:
		return numpy.mean([x for x in X if (not x==0. and strip_zero)])
#
def get_flatspots(YY,f_len=10, hi=.2, lo=0., return_full_set=False):
	# YY: array (iterable) like [[x0,y0,z0,...], [x1,y1,z1,...], ...]
	# hi,lo: high/low stdev percentiles; typically, lo=0., unless stdev=0 is suspicious.
	# @return_full_set{binary}: return a full length (well, N-1) list like YY, otherwise
	#  just return rows that are discontinuous. if you're putting these data back together
	#  into a time-series (or some other excluded axis), set this to True; excluded rows
	#  will be set to [[None, None,...]]. eventually, maybe we can alternatively return with
	#  an index column, so [[x,y,z], ...] ---> [[j,x,y,z], ...] and we can reassemble time-series.
	#
	# spin through YY; XYY can have multiple columns. 
	# for now, just make a little working copy of YY. handle
	# [y,y,..] vs [[y0,y1,y2], [y0,y1, y2], ... ] formats:
	#YY_working = [numpy.atleast_1d(rw) for rw in YY]
	#
	# probability distribution scores:
	flat_spots = []
	for col in zip(*numpy.atleast_2d(YY)):
		N = len(col)-1  # length of stdev vector.
		#
		# get stdevs and an index. we'll sort by x[1] to get P(x), then resort by x[0] (the index)
		# and add a sorted index to get probabilities.
		# [[index, stdev]]
		stdevs = [[j, numpy.std(col[max(0,j-f_len):j])] for j in range(2,len(col)+1)]
		#print('**: ', col[0:10])
		# now, add a sorted-index (or divide to get a probability): [[P,j_0, stdev], ...]
		#stdevs = [[(k+1)/N, j, s] for k, (j,s) in enumerate(sorted(stdevs, key=lambda rw:rw[1]))]
		stdevs = [[j, s] for k, (j,s) in enumerate(sorted(stdevs, key=lambda rw:rw[1]))]
		#
		# if we calc the probability for each row, we can skip this step, but that is a bit more
		# memory intensive; it's prbably better to forego the (k+1)/N column and grab the hi/lo values.
		# note, also maintain the shorthand that stdev is the last column; everything bofore it is
		# an index.
		#print('**', stdevs)
		this_hi = stdevs[min(len(stdevs)-1, int(hi*len(stdevs)))][-1]
		this_lo = stdevs[int(lo*len(stdevs))][-1]
		#
		#print('hi-lo: ', this_hi, this_lo)
		# ... and re-sort to original sequence:
		stdevs.sort(key=lambda rw: rw[-2])
		#
		flat_spots += [[x if (stdevs[k][-1]>=this_lo and stdevs[k][-1]<=this_hi) 
						else None for k,x in enumerate(col[1:])]]
	#
	# now, None out all the rows that contain one or more None (so far, we can have mixed rows)
	flat_spots = numpy.array(flat_spots).transpose()
	if return_full_set:
		for j,rw in enumerate(flat_spots):
			# this throws an exception warning, so we might need to keep an eye on it...
			if None in rw:
				for k,x in rw: flat_spots[j][k]=None
			flat_spots[j]=tuple(flat_spots[j])

		return flat_spots
	else:
		#print('returning subset...')
		return [tuple(rw) for rw in flat_spots if not None in tuple(rw)]
#
def get_flatspots_single_sequence(Y, f_len=25, hi=.2, lo=0., return_full_set=False):
	# simple function to return "flat-spots", or sub-sequences with low local stdev over a running
	# f_len measurements.
	# @return_full_set==True: return the whole sequence with None substituted for not-flat spots.
	# @return_full_set==False: return subset(s) with indices.
	#
	# if Y is 2D, assume XY format, so [[x,y, whatever], ...],
	# re-stitch together the time column upon return:
	#print('*** **: ', numpy.atleast_2d(Y)[0])
	do_time_stitch=False
	if len(numpy.atleast_1d(Y[0]))>1:
		stdevs = sfp.running_stdev([rw[1] for rw in Y])
		do_time_stitch=True
	else:
		stdevs = sfp.running_stdev(Y, f_len)
	#
	sig_lo, sig_hi = sorted(stdevs)[int(lo*len(stdevs))::int(len(stdevs)*(hi-lo))][0:2]
	#
	if not do_time_stitch:
		return [[j,x,sig] for j, (x,sig) in enumerate(zip(Y,stdevs)) if sig>=sig_lo and sig<=sig_hi]
	else:
		return [[Y[j][0],x[1],sig] for j, (x,sig) in enumerate(zip(Y,stdevs)) if sig>=sig_lo and sig<=sig_hi]

def get_n_contiguous_sequences(XY, n_clusts=2, col_index=1, do_flatten=False, f_len=25, hi=.2, lo=0., just_clusters=True):
	''''''
	# break up a sequence into n_clusts clusters or sub-sequences.
	# get n_clusts "clusters" for now, this is limited to using get_contiguous_sequences(),
	# but we can generalize if we so desire...
	# there is probalby a smart way to converge on the right solution... or a right solution
	# (this problem is not well defined)
	# presumably, however, we'll pretty much always get more clusters when we reduce dx0, so 
	# if we can find a reasonable (Newtonian?) approach to sample dx0, we should do well.
	# also note that we won't always be able to find n_clusts clusters; consider a peridoc sequence...
	#
	# so, let's use the distribution of dy-s. start with the median value; if we have too many clusters,
	# make dx0 bigger, otherwise, make it smaller, each time by median steps.
	#
	# @do_flatten: flatten the sequence? aka, select "flatspots" based on local running standard deviation,
	# aka get_Flatspots_single_sequence(). if True, then use the subsequent parameters.
	#
	# TODO: what's the best way to get the max/min dx0 that gives n_clusts clusters?
	''''''
	#
	if do_flatten:
		XY = get_flatspots_single_sequence(Y=XY, f_len=f_len, hi=hi, lo=lo, return_full_set=False)
	#
	dYs = [abs(rw[col_index]-XY[j][col_index]) for j,rw in enumerate(XY[1:])]
	dx_index = sorted([[j,dy] for j,dy in enumerate(dYs)], key=lambda rw: rw[1])
	#dx_index = sorted([[j,rw[col_index]-XY[j][col_index]] for j,rw in enumerate(XY[1:])], key=lambda rw:rw[1])
	N_dx = len(dx_index)
	n_clusters = N_dx	# current count.
	#
	#
	# indices: j0 is our current index; j_upper, j_lower are the upper/lower bounds.
	# note on // ingeger division: float(x1)//{x2} will give float(int(x1/x2)), which will
	# chuck an error as an index.
	j_0 = int(N_dx//2)
	j_lower = 0
	j_upper = N_dx-1
	dx0 = dx_index[j_0][1]
	clusts = []
	#
	#
	#print('now loop...')
	while n_clusters != n_clusts and dx0>dx_index[0][1]:
		# while we still have too many clusters and before we reach the end of our sequence (iterated through
		# the whole set)...
		#
		dx0 = dx_index[j_0][1]
		clusts = get_contiguous_sequences(XY, col_index=col_index, dx0=dx0)
		n_clusters = len(clusts)
		#
		#print('*** {} / {} / {} :: {}'.format(j_lower, j_0, j_upper, dx0))
		#
		if n_clusters>=n_clusts:
			#
			# too many clusters; make dx0 bigger and exclude the (presumed) 'smaller' domain:
			j_lower = j_0
			j_0 = int(((j_0 + j_upper)/2))
		elif n_clusters < n_clusts:
			# > vs >= : do we want to dial it down until we get the min dx for separation? if so, 
			# we'll also have to adjust our loop conditions.
			#
			# not enough clusters; make dx0 smaller and try again:
			j_upper = j_0
			j_0 = int(((j_0 + j_lower)/2))
		#
	#
	#print('len**: ', len(clusts))
	if just_clusters:
		return clusts
	else:
		return {'clusters':clusts, 'dx0':dx0, 'j_0': j_0}

def get_contiguous_sequences(XY, col_index=1, dx0=None):
	# return sub-sequences of contigous elements (contiguous means dx<=dx0)
	# assume XY are sorted by X...
	# ... but just based on dy. eventually, this will be a Bayes discriminator...
	# TODO: sort out a proper Bayes discriminator
	# TODO: generalize this for multi-variate input; sort out the linear algebra form of this.
	#	   at least, we nominally want to separate in x and y, if not also by some hidden dimension.
	#
	#if dy0 is None: dy0 = dy_factor*numpy.std([rw[1] for rw in XY])
	if dx0 is None: dx0 = 25
	#print('dy0: ', dy0)
	#
	sub_sequences = [[XY[0]]]
	for j,rw in enumerate(XY[1:]):
		# if not contiguous, add a new row.
		#print('rw: ', rw)
		if abs(rw[col_index]-XY[j][col_index])>dx0:
			#print('ss_len: ', len(sub_sequences[-1]))
			sub_sequences += [[]]
		sub_sequences[-1]+=[rw]
	return sub_sequences

	
#
def stack_clusters(clusters, min_clust_size=1):
	#
	XY_reduced = []
	for clust in clusters:
		#
		if len(clust)<min_clust_size: continue
		#
		x0s = [numpy.mean(x) for x in zip(*clust)]
		XY_reduced += [[x-x0s[j] for j,x in enumerate(rw)] for rw in clust]
	XY_reduced.sort(key = lambda rw: rw[0])
	#
	return XY_reduced
#
###################
def nn_cluster_demo(N_nn=2):
	# this works in the notebook but maybe not here...
	# make some data:
	# to start with, make clusters with exactly the same slope; shift them around a bit
	#
	n_nn = int(N_nn)
	R_x  = random.Random()
	R_dy = random.Random()
	sig_y = .2
	#
	b = 1.5
	clusters_input = []
	# [ [x0, x_max, b, y0, N], ... ]
	prms = [[2., 5.,b, 1., 100], [2.,4., b, 3.,50], [4.,6.,b,2.,75], [4.5, 5.5,b,-1.5,100]]
	#
	for x0, x1,b,y0,N in prms:
		#pp = {key:val for key,val in zip(['x0', 'x1','b','y0','N'], prms)}
		#
		dx = x1-x0
		X = sorted([x0 + R_x.random()*dx for j in range(N)])
		Y = [(-sig_y + 2.*R_dy.random()*sig_y) + y0 + b*x for x in X]
		clusters_input += [list(zip(X,Y))]
	#
	#print(len(clusters_input))
	#for c in clusters_input: print(len(c))
	#
	fg=plt.figure(0, figsize=(14,6))
	plt.clf()
	#ax1 = fg.add_axes([.1,.1,.35,.8])
	#ax2 = fg.add_axes([.5,.1,.35,.8])
	ax1 = fg.add_axes([.1,.1,.25,.8])
	ax2 = fg.add_axes([.4,.1,.25,.8])
	ax3 = fg.add_axes([.7,.1,.25,.8])
	ax1.set_xlabel('$x$', size=16)
	ax1.set_ylabel('$y$', size=16)
	ax1.set_title('Clusters')
	#
	#return None
	#
	XY = []
	for j,c in enumerate(clusters_input):
		#
		ax1.scatter(*zip(*c), marker='.', color=colors_[j%len(colors_)], zorder=5)
		XY += c
		#
	ax1.scatter(*zip(*XY), marker='o', alpha=.3, zorder=1, s=50, color='y')
	#	
	# find clusters, etc.
	CC = NN_cluster_finder(XY, nn_dist_threshold_factor=.5, n_neighbors=N_nn)
	print('cutoff: {}'.format(CC.dist_cutoff))

	S = CC.get_self_nn_clusters(dist_cutoff=CC.dist_cutoff)
	print('len(S): ', len(S))

	XY_reduced = []
	for s in S:
		#
		#if len(s)<3: continue
		x0s = [numpy.mean(x) for x in zip(*s)]
		XY_reduced += [[x-x0s[j] for j,x in enumerate(rw)] for rw in s]
	XY_reduced.sort(key = lambda rw: rw[0])
	#
	lsq1 = numpy.linalg.lstsq([[1.,x] for x,y in XY], [y for x,y in XY])
	lsq2 = numpy.linalg.lstsq([[1.,x] for x,y in XY_reduced], [y for x,y in XY_reduced])
	#
	print('lsq1: ', lsq1)
	print('lsq2: ', lsq2)
	#
	f_lin = lambda x,a,b: a + b*x
	#
	#fg0 = plt.figure(figsize=(12,10))
	#ax0 = plt.gca()
	x_fit = numpy.array([XY[0][0], XY[-1][0]])
	x_rfit = numpy.array([XY_reduced[0][0], XY_reduced[-1][0]])
	#
	for j,s in enumerate(S):
		ax2.scatter(*zip(*s), marker='.', color=colors_[j%len(colors_)], label='clust: {}/{}'.format(j, len(s)))
	ax2.plot(x_fit, f_lin(x_fit,*lsq1[0]), lw=2., ls='-', label='fit: {}'.format(lsq1[0]))
	ax2.legend(loc=0)
	#
	# now, compress:
	#plt.figure()
	ax3.plot(*zip(*XY_reduced), ls='', marker='.')
	ax3.plot(x_rfit, f_lin(x_rfit,*lsq2[0]), ls='-', lw=2., marker='', label='fit: {}'.format(lsq2[0]))
	ax3.legend(loc=0)

	#
	return S

#
def running_stdev(data_in, med_len=25):
	# running standard deviation.
	# ... we can simplify this syntax.
	#n=len(data_in)
	return numpy.array([numpy.std(data_in[int(max(j+1-med_len,0)):j+1]) for j,x in enumerate(data_in)])
#


