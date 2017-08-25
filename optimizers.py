'''
# veriipy.optimizers.py:
# optimization, fitting, and other related algorithms.
#
# i think the thing to do, eventually, is to integrate these functions and classes (this module) into
# a more modular, generalized, portable "optimizers" library. these are really "auto_optimizers"; we can 
# put these together with ROC tools and other optimization algorithms.
#
'''
import matplotlib as mpl
import pylab as plt
import numpy
#import matplotlib.dates as mpd
import os
import json
import multiprocessing as mpp
import time
import math
import operator
import functools

from numba import jit
#
#
def lsq_XY(XY):
	'''
	# a simple least-squares wrapper. assume XY is like [[x,y], ...], or more generally, [[x0,x1,x2,...,y], ...]
	'''
	return numpy.linalg.lstsq([[1.]+rw[:-1] for rw in XY], [rw[-1] for rw in XY])
#
def auto_optimize_cols(A,Bs,col_names=None, min_len_factor=.5, step_size=1, chi_expon=1.5, n_cpu=None):
	# a procedural wrapper to use Auto_Optimizer and return a
	return None
	

class Auto_Optimizer(object):
	# would be benefit from interiting mpp.Pool(), or just contain it, and maybe manually map some of the functions?
	# we could build this like we built Global_ETAS, where we have a base Optimizer class, variations of that, and then an MPP handler that inherits from Optimizer
	# and tranparelty handles the mpp parts. or, we just create a single class that recognizes the spp/mpp implementations.
	#
	# earlier versions of this parallelized by distributing multiple optimization jobs to processors. we'll actually break down each optimizatio job.
	# so we'll parallelize by splitting over the fit-range of each optimization job,
	# and we'll distirubt the start-points so that each processor gets approximately the same load, aka:
	#  - step_size_mpp = step_size*n_cpu
	#  - start_indices = [j for j in range(n_cpu)]
	#
	# note: as per pickling and piping overhead, for ~1000 rows, the spp solution is much faster than full mpp, so we should put in some sort of logic to guess the best
	# cpu-count (of course, if it's not mpp.cpu_count(), then it's only a second or two, so who cares).
	#
	def __init__(self, A,B, col_name='mycol', min_len_factor=.5, step_size=1, chi_expon=1.5, n_cpu=None, auto_run=True):
		#
		self.__dict__.update(locals())
		#self.A=A
		#self.B=B		# result vector(s), transposed, so b1, b1, b3 --> B = [[b10, b20, b30], [b11,b21,b31], ...]
		#
		n_cpu=(n_cpu or mpp.cpu_count())
		if n_cpu<1: n_cpu = mpp.cpu_count()-n_cpu
		#
		if len(A)<1500: n_cpu=1
		#
		#
		if auto_run:
			self.best_prams = self.optimize()
		#
	#
	def optimize(self, A=None, B=None, n_cpu=None):
		n_cpu=(n_cpu or self.n_cpu)
		n_cpu=(n_cpu or mpp.cpu_count())
		#
		if A is None: A=self.A
		if B is None: B=self.B
		#
		#print('n_cpu: ', n_cpu)
		if n_cpu==1:
			self.best_prams = self.auto_optimize_array(A,B)
		else:
			self.best_prams = self.auto_optimize_mpp_handler(A,B, n_cpu)
		#
		return self.best_prams
		#
	#
	def auto_optimize_mpp_handler(self,A,B, n_cpu=None):
		if A is None: A=self.A
		if B is None: B=self.B
		n_cpu = (n_cpu or self.n_cpu)
		#
		mpp_step_size = self.step_size*n_cpu
		#
		P = mpp.Pool(n_cpu)
		resultses = [P.apply_async(self.auto_optimize_array, args=(A,B), kwds={'start_index':j, 'step_size':mpp_step_size}) for j in range(n_cpu)]
		P.close()
		P.join()
		#
		best_fits = [res.get() for res in resultses]
		#print('best_fits: ', best_fits)
		best_fit = sorted(best_fits, key=lambda rw: (None if len(rw[2])==0 else numpy.sqrt(numpy.mean(rw[2]))/max(1, (len(A)-rw[0]))**self.chi_expon))[0]
		#
		return best_fit
	#
	# auto_optimize_array can fundamentally be a procedural function, outside this class. we'll just write a connecter to it here.
	def auto_optimize_array(self, A=None,B=None, min_len_factor=None, step_size=None, start_index=0, chi_expon=None,fignum=None, do_clf=True, ax=None, show_all_lsqs=True, do_aggregate=True):
		#
		# @A: matrix of model coefficients (aka, T,zeta, etc. series).
		# @B: matrix (transposed vectors) of outputs, so:
		#  A \dot p' = b, or B = [b0,b1,b2...]^T, and A \dot P' = B, and p are the parameters for one vector, P are the parameters for multiple vectors.
		# using more than one output vector b amounts to solving for a set of training parameters from multiple sources.
		# this approach, however, might be problematic. the residual is minimized for each vector b_j, but the starting point (and therefore the selected set of fits)
		# is chosen by the combined minimum chi-square, so it is probably more rigorous to simply fit each column completely independently.
		#
		if A is None: A = self.A
		if B is None: B = self.B
		if min_len_factor == None: min_len_factor=self.min_len_factor
		if step_size==None: step_size=self.step_size
		chi_expon = (chi_expon or self.chi_expon)
		#
		return auto_optimize_array(A=A,B=B,min_len_factor=min_len_factor,step_size=step_size,start_index=start_index,chi_expon=chi_expon, fignum=fignum, do_clf=do_clf, ax=ax,show_all_lsqs=show_all_lsqs,do_aggregate=do_aggregate)
		'''
		# walk through starting positions 0,1,... and to a least-squares fit. pick the subset that gives the best fit.
		# this is largely ripped out from get_optimal_training_sets(), and in fact the matrix part of get_optimal_training_sets() should be ripped out and reference this function.
		#
		# self-optimize Ab' = b. for now, assume our concern is the leading parts of A,b, so we'll start with the full sequence and walk down the starting point until we get an optimal chi-square.
		# for now, force b --> vector.
		#
		#	
		len_A = len(A)
		len_rw = len(A[0])
		#
		#print('lens: ', len(A), len(b))
		#
		max_starting_index = max(1, int(min_len_factor*len(A)))
		#
		#lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], b[j:])[0:2]) for j in range(int(len(A)-1-len_rw))]
		# this works, but only shows the subset. do we want to see the full distribution of lsqs?
		if show_all_lsqs:
			#lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, int(len(A)-1-len_rw), step_size)]
			lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, min(max_starting_index, int(len(A)-1-len_rw)), step_size)]
		else:
			lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, min(max_starting_index, int(len(A)-1-len_rw)), step_size)]
		#
		#print('len(lsqs): ', len(lsqs), start_index, step_size)
		# sometimes we get a weird outcome, so trap cases where lsqs() returns junk.
		#chis = [[x[0],(None if len(x[2])==0 else numpy.sqrt(x[2][0])/max(1, (len(A)-x[0]))**chi_expon)] for x in lsqs] 	# aka, reduced_chi-square/sqrt(n_dof) like quantity.
		chis = [[k, x[0],(None if len(x[2])==0 else numpy.sqrt(numpy.mean(x[2]))/max(1, (len(A)-x[0]))**chi_expon)] for k,x in enumerate(lsqs)] 	# aka, reduced_chi-square/sqrt(n_dof) like quantity.
		#
		#print('chis: ', chis[0:5])
		
		# now, get the optimal chi:
		chis.sort(key=lambda rw: rw[-1])
		k_chi, j_chi, chi_0 = chis[0]		# for spp (start_index=0, step_size=1), k_chi==j_chi... right?
		#
		if fignum!=None:
			plt.figure(fignum)
			if do_clf: plt.clf()
			if ax==None: ax=plt.gca()
			#
			clr = colors_[len(plt.gca().lines)%len(colors_)]
			ax.plot(*zip(*[(x,y) for x,y in chis]), ls='-', color=clr, lw=2.5)
			ax.plot([j_chi], [chi_0], marker='o', color=clr)
		#
		# do we want to necessarily aggregate here? why not. we can change it later, or make it optional.
		# lsqs returns like [[a0,a1,a2,...], [b0,b1,b2,...], ]
		# if we pass only a single b vector, this is still true, but each parameter is an l=1 vector, so the operation does not change.
		# same idea with the second value/array returned by lsqs.
		#
		if do_aggregate:
			#
			return [lsqs[k_chi][0], numpy.array([numpy.mean(X) for X in lsqs[k_chi][1]]), [numpy.mean(lsqs[k_chi][2])]]
		else:
			#
			return lsqs[k_chi]
		#
		'''
#
class Synthetic_Distribution(list):
    # build it from a numpy array? maybe...
    # Class for a synthetic distribution (aka, an array of XY data that we'll treat like some sort of function
    # or probability distribution)
    # one of the main things we'll do is find roots, extrema, etc.
    #
    def __init__(self, XY):
        ''''''
        # load XY. if XY is 1D or len(XY_j)==1, add an index as an X component. later we might decide this
        # is not convenient and let a 1D array be 1D... or we can handle that in code.
        ''''''
        #
        #if not numpy.atleast_1d(XY[0])
        XY = [numpy.atleast_1d(rw) for rw in XY]
        #
        # assume everything is the same width...
        # ... and maybe just check for a list type and length instead of this more convoluted approach.
        if len(XY[0])<2:
            XY = [[j,x[0]] for j,x in enumerate(XY)]
        #
        super(Synthetic_Distribution,self).__init__(XY)
        self.sort(key=lambda rw: rw[0])
    #
    #
    @property
    def x_min(self):
        return self[0][0]
    @property
    def x_max(self):
        return self[-1][0]
    @property
    def y_min(self):
        return min([y for x,y in self])
    @property
    def y_max(self):
        return max([y for x,y in self])
    #
    @property
    def segments(self, x_min=None, x_max=None):
        #x_min = (x_min or self.x_min)
        #x_max = (x_max or self.x_max)
        #
        return [[self[j],xy] for j,xy in enumerate(self[1:])]
    @property
    def Y(self):
        return numpy.array([y for x,y in self])
    @property
    def X(self):
        return numpy.array([x for x,y in self])
    #
    def get_f_of_x(self,x, interp_hi=True, interp_lo=True):
    	# find f(x) discretely. find the intermediate values of x; then interpolate (just like get_roots(), but easier. assume only one value?
    	# @truncate_hi/lo: if x is outside the distribution's domain, return the end values if True.
    	#
    	# are we outside the domain?
    	if (interp_hi and x>=self.x_max): return self[-1][1]
    	if (interp_lo and x<=self.x_min): return self[0][1]
    	y = None
    	#
    	for (x1,y1), (x2,y2) in self.segments:
    		#
    		# trap for exact match,
    		# we can handle this in the general case with <=
    		#if x==x1: return y1
    		#if x==x2: return y2
    		#
    		# else...
    		if (x1-x)*(x2-x)<=0.:
    			# x == {x1 or x2} or ....
    			# one diff. is negative, one is positive, so x is in between.
    			# we'll be interpolating...
    			dx = x2-x1
    			dy = y2-y1
    			#
    			# TODO: make this more robust to handle multi-valued scenario (aka, vertical line)?
    			if not dy==0.:
    				y = y1 + (dy/dx)*(x-x1)
    			else:
    				y = numpy.mean([y1,y2])
    			#
    			break
    		#
    	#
    	return y
    #
    def get_roots_discrete(self, y0=0., return_slope=False):
        '''
        # find the roots at y=y0 for our self-distribution. the zero will fall between segments, so we want
        # segments where y1>y0 and y2<y0, or vice versa.
        #
        # Also, return the local slope at the root value. this might get rifined as we go, to get a weighted mean over the current and
        # adjacent segments. for now, just use the local slope.
        '''
        #
        roots = []
        for (x1,y1), (x2,y2) in self.segments:
            if ((y1-y0)*(y2-y0)) <= 0.:
                # either y0==y1 or y0==y2 or one's + and the other's -, so it's a root segment.
                #print('*** rooting ***')
                dy = y2-y1
                dx = x2-x1
                dy_dx = dy/dx
                #
                if not dy==0:
                    x = (y0-y1)*dy_dx + x1
                else:
                    x = .5*(x1+x2)
                #
                if return_slope:
                	roots += [[x,dy_dx]]
                else:
	                roots += [x]
        #
        return roots
    def get_roots(self,*args,**kwargs):
        return self.get_roots_discrete(*args, **kwargs)
    #
    '''
    def get_roots_brute(self, y, x_min=None, x_max=None):
        x_min = (x_min or self.x_min)
        x_max = (x_max or self.x_max)
        #
        # the idea of this is to pass a function and some simple prams and try a brute force approach to
        # root finding. however, we are shifting towards discrete analyses, so we might not end up doing
        # much with this.
        #
        pass
    '''
    #
    def get_roots_fsolve_2(self):
        # use scipy.optimize.fsolve(). this uses (i think) an Newtonian type iteration over the input function
        # f to find roots. you do need to know a thing or two about the function; this script provides an 
        # an example of how to find the roots of a Poisson distribution with arbitrary k.
        #
        r1=scipy.optimize.fsolve(lambda x: f_poisson(x,k,chi,y), x0)[0]
        #
        # get max:
        xy_max = sorted(zip(X,Y), key=lambda rw:rw[1])[-1]
        print('xy_max: ', xy_max)
        # here's a good trick to get roots to converge. we're assuming two roots (or that we want the first 2)
        # for the second root, start looking at r2_0 = max_val + r1
        # this, however, is probably not much faster, if at all, (except that some of the calculation might
        # be done in the compiledlibrary) than scanning the whole distribution for intercept, especially if we
        # can pre-index the distribution (namely because of the max() function). it also assumes a well behaved
        # global-maximum behavior.
        #
        r2=scipy.optimize.fsolve(lambda x: f_poisson(x,k,chi,y), r1+xy_max[0])[0]

        r=[r1,r2]
        
#
def auto_optimize_multiple_arrays(A=None,bs=None, min_len_factor=.5, step_size=1, start_index=0, chi_expon=1.5,fignum=None, do_clf=True, ax=None, show_all_lsqs=True, do_aggregate=True,n_cpu=None, verbose=0):
	'''
	# (this appears to work.. but needs more work. see **prms and params sent in mpp mode as well.).
	# auto_optimize; parallelize by each column. we see in Auto_Optimize() that for relatively small sets (certainly len(A)<=1000 or so), the mpp overhead is more costly than the benefits.
	# so for many operations, it will be faster to just parallelize by full sequence, so there's (presumably) less piping.
	#
	# A: matrix of model coefficients
	# bs: array of output vectors. note this is different than some of the ther auto_optimize_array() inputs; for this, we're going to force b -> a vector, and forego
	# simultaneous fits. Note that, with a little bit of code, we can force this condition by simply measuring and correcting (as needed) the dimension of the b input.
	#   SO, A --> a fitting matrix, [[a0,c0, d0,...], [a1,c1, d1, ...], ...], bs -> [b1, b2, b3...] ], where b_j are vectors of observables (aka, MOScap measurement seq.).
	'''
	print('running auto-optimize for multiple vectors.')
	original_inputs = {key:val for key,val in locals().items() if not key in ['n_cpu']}
	if n_cpu==1:
		prms = {key:val for key,val in original_inputs.items() if key not in ['bs']}
		prms['n_cpu']=1
		#
		# note: we might need touse this weird format for b/B, basically to facilitate arrays of multiple b vectors... or we rewrite the optimize function.
		#return [auto_optimize_array(B=[[x] for x in b], **XX) for b in bs]
		return [auto_optimize_array(A=A,B=b, **prms) for b in bs]
	#
	n_cpu = (n_cpu or mpp.cpu_count())
	P = mpp.Pool(n_cpu)
	#
	A = numpy.array(A)
	bs=numpy.array(bs)
	#
	if verbose>0:print('mpp auto-optimizing: ', A.shape, bs.shape)
	#
	#resultses = [P.apply_async(auto_optimize_array, args=(A,b), kwds={'start_index':start_index, 'step_size':step_size}) for b in bs]
	opt_params = {'start_index':start_index, 'step_size':step_size, 'min_len_factor':min_len_factor, 'chi_expon':chi_expon, 'do_aggregate':do_aggregate}
	#
	# do these necessarioly come back in the correct order? i think they do (or backwards) because they are loaded into their respective arrays in order.
	# in other words, we can label them from the calling side, as opposed to inforcing a data structure like:
	# resultses = {col_name:P.apply_async(auto_optimize_array, args=(A,b), kwds=opt_params) for col_name,b in bs.items()]
	#  ... and then b_f = {key:res.get() for key,res in resultes.items()} ... or we can do it as a 2D list.
	#
	if verbose>0: print('auto-fitting. len(bs): {}, {}'.format(len(list(bs)), numpy.shape(bs)))
	#
	resultses = [P.apply_async(auto_optimize_array, args=(A,b), kwds=opt_params) for b in bs]
	#
	P.close()
	P.join()
	#
	best_fits = [res.get() for res in resultses]
	#print('best_fits: ', best_fits)
	#best_fit = sorted(best_fits, key=lambda rw: (None if len(rw[2])==0 else numpy.sqrt(numpy.mean(rw[2]))/max(1, (len(A)-rw[0]))**chi_expon))[0]
	#
	return best_fits	
#
def auto_optimize_mpp_handler(self,A,B, step_size=1, n_cpu=None):
	'''
	# an mpp handler to auto-fit a single sequence. split up the job by splitting up the starting points. for short sequences, the overhead seems to overwhelm the
	# benefit of mpp, and it is probably better to use either a spp function or auto_optimize_multiple_arrays(). nominally, this handler is adept for optimizing one or two
	# really long sequences.
	'''
	#if A is None: A=self.A
	#if B is None: B=self.B
	#n_cpu = (n_cpu or self.n_cpu)
	#
	n_cpu = (n_cpu or mpp.cpu_count())
	#
	mpp_step_size = step_size*n_cpu
	#
	P = mpp.Pool(n_cpu)
	resultses = [P.apply_async(auto_optimize_array, args=(A,B), kwds={'start_index':j, 'step_size':mpp_step_size}) for j in range(n_cpu)]
	P.close()
	P.join()
	#
	best_fits = [res.get() for res in resultses]
	#print('best_fits: ', best_fits)
	best_fit = sorted(best_fits, key=lambda rw: (None if len(rw[2])==0 else numpy.sqrt(numpy.mean(rw[2]))/max(1, (len(A)-rw[0]))**self.chi_expon))[0]
	#
	return best_fit
#	
def auto_optimize_array(A=None,B=None, min_len_factor=.5, step_size=1, start_index=0, chi_expon=2,fignum=None, do_clf=True, ax=None, show_all_lsqs=True, do_aggregate=True):
	'''
	# self-optimize Ab' = b. for now, assume our concern is the leading parts of A,b, so we'll start with the full sequence and walk down the starting point until we get an optimal chi-square.
	# for now, force b --> vector.
	# note: this can be used as a SPP worker for an MPP handler like auto_optimize_multiple_arrays(). for smaller arrays, this approach is much faster than MPP models that break
	# up each data set for MPP handling. for 2 or 3 (ana, n<n_cpu) very large data sets (don't know what that means yet), the latter approach is probably optimal, and i think it is coded
	# up somewhere.
	#
	# @A: matrix of model coefficients (aka, T,zeta, etc. series).
	# @B: matrix (transposed vectors) of outputs, so:
	#  A \dot p' = b, or B = [b0,b1,b2...]^T, and A \dot P' = B, and p are the parameters for one vector, P are the parameters for multiple vectors.
	# using more than one output vector b amounts to solving for a set of training parameters from multiple sources.
	# this approach, however, might be problematic. the residual is minimized for each vector b_j, but the starting point (and therefore the selected set of fits)
	# is chosen by the combined minimum chi-square, so it is probably more rigorous to simply fit each column completely independently.
	#
	# walk through starting positions 0,1,... and to a least-squares fit. pick the subset that gives the best fit.
	# this is largely ripped out from get_optimal_training_sets(), and in fact the matrix part of get_optimal_training_sets() should be ripped out and reference this function.
	#
	'''
	#
	#	
	len_A = len(A)
	len_rw = len(A[0])
	#
	#print('lens: ', len(A), len(b))
	#
	max_starting_index = max(1, int(min_len_factor*len(A)))
	#
	#lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], b[j:])[0:2]) for j in range(int(len(A)-1-len_rw))]
	# this works, but only shows the subset. do we want to see the full distribution of lsqs?
	if show_all_lsqs:
		#lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, int(len(A)-1-len_rw), step_size)]
		lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, min(max_starting_index, int(len(A)-1-len_rw)), step_size)]
	else:
		lsqs = [[j] + list(numpy.linalg.lstsq(A[j:], B[j:])[0:2]) for j in range(start_index, min(max_starting_index, int(len(A)-1-len_rw)), step_size)]
	#
	#print('len(lsqs): ', len(lsqs), start_index, step_size)
	# sometimes we get a weird outcome, so trap cases where lsqs() returns junk.
	# lsqs are like [[start_index, params, chi_sqrs], ...]
	# where chi_sqrs are an array (in general) because we might pass a matrix, not a simple vector, to the lstsq() operation, and so we average them (mean(x[2])  below)
	# for our cost-function.
	#
	#chis = [[x[0],(None if len(x[2])==0 else numpy.sqrt(x[2][0])/max(1, (len(A)-x[0]))**chi_expon)] for x in lsqs] 	# aka, reduced_chi-square/sqrt(n_dof) like quantity.
	chis = [[k, x[0],(None if len(x[2])==0 else numpy.sqrt(numpy.mean(x[2]))/max(1, (len(A)-x[0]))**chi_expon)] for k,x in enumerate(lsqs)] 	# aka, reduced_chi-square/sqrt(n_dof) like quantity.
	#
	#print('chis: ', chis[0:5])
	
	# now, get the optimal chi:
	# (index, value) of best fit chi
	chis.sort(key=lambda rw: rw[-1])
	k_chi, j_chi, chi_0 = chis[0]		# for spp (start_index=0, step_size=1), k_chi==j_chi... right?
	#
	if fignum!=None:
		plt.figure(fignum)
		if do_clf: plt.clf()
		if ax==None: ax=plt.gca()
		#
		clr = colors_[len(plt.gca().lines)%len(colors_)]
		ax.plot(*zip(*[(x,y) for x,y in chis]), ls='-', color=clr, lw=2.5)
		ax.plot([j_chi], [chi_0], marker='o', color=clr)
	#
	# do we want to necessarily aggregate here? why not. we can change it later, or make it optional.
	# lsqs returns like [[a0,a1,a2,...], [b0,b1,b2,...], ]
	# if we pass only a single b vector, this is still true, but each parameter is an l=1 vector, so the operation does not change.
	# same idea with the second value/array returned by lsqs.
	#
	# return will be [[best_chi_indes, best_params, chi_sqr_ish_val],... ]
	# "chi_sqr_ish" because it is actually the residual, i think squared, and then elevated to our input power -- which we use to emphasize sequence length, etc.
	# so, fo external operations to estimate parameter variability, etc., we need to eithe re-calculate error or take out this chi-power.
	if do_aggregate:
		#
		return [lsqs[k_chi][0], numpy.array([numpy.mean(X) for X in lsqs[k_chi][1]]), [numpy.mean(lsqs[k_chi][2])]]
	else:
		#
		return lsqs[k_chi]
		#
#
#
def get_stdev_training_set(input_data, N_sample=10, lo=0., hi=.8, fignum=None, verbose=False):
    #''''''
    ## get the subset of input_data with low running stdevs. Specifically:
    ## 1) take running stdevs over N_sample for each col; keep the whole sequence, so stdev(X[min(0,j-N),j)
    ## 2) get the CDF for each stdev column in input_data, 
    ## 3) then keep the rows of input_data where all columns fall between the lo/hi stdev probability range.
    # Note: this "and" type filter could end up significantly reducing a training set
    # @verbose: False: returns just a reduced set, with the excluded elements excluded.
    #           True: returns a dict containing "keepers", "rejects", which are what they sound like; for each set
    #               excluded rows are replaced with a vector of [None,..] values.
    #
    # assume: input_data is passed  in row, column format: [[x0,y0,z0], [x1,y1,z1], ...,[xn, yn, zn] ]
    # eventually, we'll try to account for any standard iterable.
    # assume: if input_data is a dict type, all rows have the same length 
    # it's effectivly an indexed nxm list(ish) type)
    #
    # TODO: we want to make this support multiple iterables, including indexed inputs like dicts, recarrays, etc.
    # however, this presents potential challenges returning data in the correct sequence, as the same type, etc.
    # for now, let's restrict this to an array-of-rows input. we'll sort out dict, etc. support later.
    #''''''
    #
    #
    #if isinstance(input_data, dict):
    #    #cols, vals = zip(*input_data.items())
    #    cols_vals = input_data
    #if not isinstance(input_data, dict):
    #    # this might need a list() wrapper or something.
    #    cols_vals = {j:x for j,x in enumerate(zip(*input_data))}
    #
    # set up a little reduce() function. reduce() used to be part of the core Python library, but
    # it has been deemed unworth, so now we can either import it specially (from functools or something like that)
    # or write our own...
    #
    if verbose: fignum = (fignum or 0)
    #
    # get running standard deviations:
    stdevs = {j:[numpy.std(vals[max(0, k+1-N_sample):k+1]) for k,x in enumerate(vals)] 
              for j, vals in enumerate(zip(*input_data))}
    #
    # diagnostic:
    if not fignum is None:
        plt.figure(fignum)
        plt.clf()
        ax=plt.gca()
        #
        #ax.set_yscale('log')
        for col,val in sorted(stdevs.items(),key=lambda x:x[0]):
            plt.plot(val, marker='.', ls='-', label=col)
        plt.legend(loc=0, numpoints=1)
    #
    # now, get hi-low thresholds for each column.
    thresholds = {}
    for col,vals in stdevs.items():
        # sort values; get hi/lo threshold values.
        # note: we want to support different types of iterables, so use Python's sort() function.
        # for now, let's just be a little bit sloppy with interpolation, small sample sizes,etc.
        x = sorted(vals)
        thresholds[col] = {'hi': x[min(int(len(x)*hi), int(len(x)-1))], 'lo':vals[int(len(x)*lo)]}
    #
    #stdevs = list(zip(*[stdevs[j] for j in range(len(stdevs))]))
    #print('thresholds: ',thresholds)
    # now, spin through the input data; keep only rows where all values fall within stdev thresholds.
    # return as recarray of floats (int type folks will have to sort it out on the other side);
    # this will make the list/dict type input fairly transparent.
    #
    # now, spin through the input data; return only rows where all elements fall within the hi/lo range:
    #
    if not verbose:
        # note: stdevs is a dict, effectively transposed to input_data
        #
        return[rw for j,rw in enumerate(input_data) 
            if functools.reduce(operator.and_, [(stdevs[k][j]>=thresholds[k]['lo'] 
                                                 and stdevs[k][j]<=thresholds[k]['hi']) for k,x in enumerate(rw)])]
    else:
        #keepers = [rw for j,rw in enumerate(input_data) 
        #    if and_reduce([(stdevs[k][j]>=thresholds[k]['lo'] and stdevs[k][j]<=thresholds[k]['hi']) for k,x in enumerate(rw)])]
        keepers = [[a if functools.reduce(operator.and_,[(stdevs[k][j]>=thresholds[k]['lo'] and stdevs[k][j]<=thresholds[k]['hi']) for k,x in enumerate(rw)])
                    else None for a in rw]  for j,rw in enumerate(input_data) ]
             
        rejects = [[a if not functools.reduce(operator.and_,[(stdevs[k][j]>=thresholds[k]['lo'] and stdevs[k][j]<=thresholds[k]['hi']) for k,x in enumerate(rw)])
                    else None for a in rw]  for j,rw in enumerate(input_data) ]
        #
        #rejects = [rw for j,rw in enumerate(input_data) 
        #    if not and_reduce([(stdevs[k][j]>=thresholds[k]['lo'] and stdevs[k][j]<=thresholds[k]['hi']) for k,x in enumerate(rw)])]
        #
        #return {'keepers': keepers, 'rejects':rejects, 'reduced': [rw for rw in keepers if not None in rw]}
        #
        return {'keepers': keepers, 'rejects':rejects}
   
#
def optimizer_test(data_file=os.path.join( os.path.split(os.path.abspath(__file__))[0],'data/test_optimization_data.json'), n_cpu=None):
	# define some tests. some reasonable unit tests would also be to modify the b column values (aka, *=1.5 or something) and see that 1) the outputs change,
	# and 2) if we modify the single column and the 2-of-same column, we get the same outputs.
	# ... and other mods too...
	#
	AB = json.load(open(data_file,'r'))
	#
	print('load an optimizer object, then run some unit diagnostics.')
	print('All fitting solutions should produce the same output. eventually, we will build the check-sum for this into the unit test itself.')
	opter = Auto_Optimizer(AB['A'],numpy.array(AB['B']), n_cpu=1, auto_run=False)
	#
	print('run a simple, spp optimization')
	best_prams_spp1 = opter.optimize(n_cpu=1)
	print('best prams, spp1.: ', best_prams_spp1)
	#
	# (and a little bit of a different calling signature, but amounting to the same thing)
	# fit two of the same column:
	print('fit to copy of target columns. output should be the same as spp1')
	opter = Auto_Optimizer(AB['A'],numpy.array([[b[0],b[0]] for b in AB['B']]), n_cpu=1)
	print('best prams2: ', opter.best_prams)
	#
	mpp_pramses=[]
	#
	for k in range(1,mpp.cpu_count()+1):
		# now, try with multiple processes:
		print('run a simple, mpp[{}] optimization'.format(k))
		t0=time.time()
		mpp_pramses += [opter.optimize(n_cpu=k)]
		print('best prams[dt={}], mpp{}.: {}'.format(time.time()-t0, k, mpp_pramses[-1]))
	#
	print('******')
	xx = opter.auto_optimize_array(start_index=0,step_size=1)
	print('xx: ', xx)
	xx = opter.auto_optimize_array(start_index=1,step_size=2)
	print('xx: ', xx)
	xx = opter.auto_optimize_array(start_index=0,step_size=2)
	print('xx: ', xx)
	#
#
def jit_fact_test(N,n):
	# script to test @jit compiler.
	obj = Jit_tester()
	obj.jit_fact_test(5,5)
	print('*******************\n\n')
	obj.jit_fact_test(N,n)
	#
#
@jit
def n_fact_jit(x):
	# calc x! with a for-loop
	# this is just a test, so for a fraction, just take the int(x) and chuck a warning.
	if x%1!=0:
		print('WARNING: non-integer value submitted: %f. calculating, instead, %d!' % (x,int(x)))
		x=int(x)
	#
	x=float(x)
	#xx=1
	#for j in range(1,x+1): xx*=j
	for j in range(1,int(x)): x*=j
	#
	return x
class Jit_tester(object):
	# can we make numbas.jit work?
	def jit_fact_test(self,N,n=100):
		#
		# test jit with a factorial test.
		t0=time.time()
		for j in range(n): x=self.n_fact_jit(N)
		print('sJIT: N! = {}'.format(x))
		t1=time.time()
		print('finished: {}/{}'.format(t1-t0, math.log((t1-t0))))
		#
		# test jit with a factorial test.
		t0=time.time()
		for j in range(n): x=n_fact_jit(N)
		print('xJIT: N! = {}'.format(x))
		t1=time.time()
		print('finished: {}/{}'.format(t1-t0, math.log((t1-t0))))
		#
		t0=time.time()
		for j in range(n): x=self.n_fact(N)
		print('pypy: N! = {}'.format(x))
		t1=time.time()
		print('finished: {}/{}'.format(t1-t0, math.log((t1-t0))))
		#
		t0=time.time()
		for j in range(n): x=math.factorial(N)
		print('math: N! = {}'.format(x))
		t1=time.time()
		print('finished: {}/{}'.format(t1-t0, math.log((t1-t0))))
	#
	@jit
	def n_fact_jit(self,x):
		# calc x! with a for-loop
		# this is just a test, so for a fraction, just take the int(x) and chuck a warning.
		if x%1!=0:
			print('WARNING: non-integer value submitted: %f. calculating, instead, %d!' % (x,int(x)))
			x=int(x)
		#
		x=float(x)
		#xx=1
		#for j in range(1,x+1): xx*=j
		for j in range(1,int(x)): x*=j
		#
		return x
	#
	def n_fact(self,x):
		# calc x! with a for-loop
		# this is just a test, so for a fraction, just take the int(x) and chuck a warning.
		if x%1!=0:
			print('WARNING: non-integer value submitted: %f. calculating, instead, %d!' % (x,int(x)))
			x=int(x)
		#
		x=float(x)
		#xx=1
		#for j in range(1,x+1): xx*=j
		for j in range(1,int(x)): x*=j
		#
		return x
	
	

	
