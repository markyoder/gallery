
#
import numpy
import math
import matplotlib.dates as mpd
import datetime as dtm
import pytz
#
tz_utc = pytz.timezone('UTC')
#
# will this need to be a class object?
#
def datetime_handler(date_in, tz_out=None, r_type='datetime'):
	# tz_out=tz_utc
	# for now, return a datetime.datetime. later, we might code in some options.
	#
	if isinstance(date_in, dtm.datetime):
		r_val = date_in
	#
	# date-to-datetime is surprisingly annoying. here's an ok way to do it:
	elif isinstance(date_in, dtm.date):
		r_val = dtm.datetime.combine(date_in, dtm.datetime.min.time())
	#
	#
	elif isinstance(date_in, str):
		# string object.
		# matplotlib.dates does a pretty good job of handling date strings. we can give it just about anything but 
		# a european format (dd-mm-yy) and get a consistent output. note that it will interpret an impossible american
		# format as a european format, aka for 2016 may 10 11:25:10.77, we get consistent results for YYYY/mm/dd, dd/mm/YYYY
		# but somethign different for mm/dd/YYYY, since 10/5 and 5/10 are permissible.
		# for 15 May 2016, we get consistent output for YYYY/mm/dd, mm/dd/YYYY, dd/mm/YYYY, since month=15 cannot exist.
		# this is kinda awesome, but could result in some tricky exceptions if we're not careful.
		#
		# anyway, use mpd to convert a datestring to a number:
		#f_date = mpd.datestr2num(date_in)
		#print('executing string conversion.')
		r_val = mpd.num2date(mpd.datestr2num(date_in), tz=tz_out)
	#
	# numpy.datetime64...
	# these types are a HUGE pain in my ... neck. for some reason, they keep changing. they appear to be based on an array object, so once upon
	# a time, list(numpy.datetime64) --> datetime.datetime object. but no longer. there's a x.tostring() function but it returns some sort of
	# nonsense; tolist() gives a big integer...
	# there is an astype() function, but it's tricky. 
	# astype(dtm.datetime) (for now) returns astype(int); astype(float) correctly returns a float value -- give a google to see what these are
	# precisely; i think they're some multiple of a unix timestamp. x.astype(str) returns x.__str__(), or equivalently str(x)...
	# so i think it's a pretty good bet for now.
	# i think we want to try to figure out which functions are deliberate datetime64 members and which are inherited from its various subclasses.
	#
	elif isinstance(date_in, numpy.datetime64):
		# if x.astype() gets screwy, a better approach might be to use the x.__str__() method or just str(x).
		#return mpd.num2date(mpd.datestr2num(str(date_in)), tz=tz_out)
		r_val = mpd.num2date(mpd.datestr2num(date_in.astype(str)), tz=tz_out)
	#
	if r_type=='datetime':
		return r_val
	else:
		return str(rval)
	#

def guess_type(x, int_to_float=False):
	# guess the data type. for now, assume relatively well constrained types: int, float, date, string. initial input will be string.
	# if not string or bytestring, then just return.
	# note: returns the variable re-cast as guessed type.
	# note: this won't work well for complex strings.
	#x = x0
	#if x[0]=='-': x = x[1:]
	#
	# first, let't take a guess that this is a good way to catch byte-strings:
	if hasattr(x, 'decode'): x=x.decode()
	if not isinstance(x, str): return x
	if x.strip() == None: return x		# best way to handle '' or ' '?
	#
	if x.count('/')==2 or x.count('-') in (2,3):
		# it's probably a date-time. we might also look for ':' and some other bits.
		# i think i found a single command for this somewhere, but i don't recall...
		# this might also return a numpy.datetime instead of a datetime.datetime, so keep an eye on it.
		#
		try:
			#return mpd.num2date(mpd.datestr2num(x))
			return datetime_handler(x)
		except:
			print("puked on guessing format for: %s" % str(x))
			return str(x)
		#
	elif x.count('.')==1 and (str.isnumeric(x.replace('.','')) or (x[0]=='-' and str.isnumeric(x[1:].replace('.','')))):
		# probably a float:
		return float(x) 
	elif (str.isnumeric(x) or (x[0]=='-' and str.isnumeric(x[1:]))):
		if int_to_float:
			return float(x)
		else:
			return int(x)
	else:
		return x

def datetime_handler_unit_tests():
	#
	# TODO: figure out how to get the local timezone so we can properly evaluate whether or not the timezone
	# conversions work. right now, i'm just making them work from CDT.
	#
	# lay out some scenarios and the expected equality to dtm0 ([[date_entry, equals_dtm0], ...])
	dtm0 = dtm.datetime(2016,5,10,11,25,10,int(.77*1e6))
	f_dtm0 = mpd.date2num(dtm0)
	# string format variations:
	dtstrs = [['2016-05-10 11:25:10.77', True], ['2016/05/10 11:25:10.77', True], ['2016-05-10 11:25:10.77', True],
			  ['2016/05/10 11:25:10.77', True], ['5-10-2016  11:25:10.77', True], ['5/10/2016  11:25:10.77', True],
			  ['05-10-2016  11:25:10.77', True]]
	# now, timezone information and variations:
	dtstrs += [['5-10-2016  11:25:10.77+00:00', True], ['5-10-2016  11:25:10.77+01:00', False],
			  ['5-10-2016  11:25:10.77+02:00', False], ['5-10-2016  7:25:10.77-04:00', True]]
	#
	# some datetime objects, including with tz variability.
	# ... but these are tricky because, apparently, US/Eastern timezone is not -5, but -4:56
	# and Central is not -6, but -5:51... which totally makes sense.
	# ... but some versions of pytz appear to use the integer values, so this will be a screwy test to diagnose.
	dtstrs += [[dtm.datetime(2016,5,10,11,25,10,int(.77*1e6)), True], 
			   [dtm.datetime(2016,5,10,11,25,10,int(.77*1e6), pytz.timezone('UTC')), True],
			   [dtm.datetime(2016,5,10,6,29,10,int(.77*1e6), pytz.timezone('US/Eastern')), True],
			   [dtm.datetime(2016,5,10,5,34,10,int(.77*1e6), pytz.timezone('US/Central')), True],
			   [dtm.datetime(2016,5,10,11,25,10,int(.77*1e6), pytz.timezone('US/Central')), False]
			  ]
	#
	# ... and how 'bout those stupid numpy.datetime64 objects. ugh...
	# note some syntax: these are rec-array types, so they have an expected length. month, day, etc. entries must
	# be len=2, so "-05", not "-5". also note that when we give it a time-zone, it will return something in the local
	# time but with the timezone info included.
	# the
	dtstrs += [[numpy.datetime64('2016-05-10 11:25:10.77'), False], 
			  [numpy.datetime64('2016-05-10 11:25:10.77+04:00'), False],
			  [numpy.datetime64('2016-05-10 11:25:10.77-03:00'), False], 
			  [numpy.datetime64('2016-05-10 11:25:10.77-00:00'), True]
			  ]
	#
	failed_tests = []
	#
	print("target datetime: {}".format(dtm0))
	for j, (dt, t_f) in  enumerate(dtstrs):
		dtm_processed = datetime_handler(dt, tz_out=tz_utc)
		#
		# we want to check equality, but we can get small differences in microseconds.
		# we can't subtract (not)-naive types, so convert to numbers...
		#delta_t = f_dtm0-mpd.date2num(dtm_processed)
		is_equal = mpd.date2num(dtm_processed)==f_dtm0
		did_pass = (is_equal==t_f)
		if not did_pass: failed_tests += [[j,dt,t_f]]
		print('({}) dt_{}[{}::({})]: {} :: {}'.format('pass' if did_pass else 'fail', j, is_equal, t_f, dt, dtm_processed))
		#
	#
	n_failed = len(failed_tests)
	n_total  = len(dtstrs)
	print('\n\n{}% ({} of {}) of tests passed.'.format(100.*(n_total-n_failed)/n_total, (n_total-n_failed), n_total))
	
	if len(failed_tests)>0:
		print("[{}/{}] failed tests: ".format(n_failed, n_total))
		print('note: there apper to be some inconsistencies in pytz (timezones) data, for example some versions keep dt=-4:56, not dt=-5:0 for US/Eastern, etc.')
		for rw in failed_tests:
			print('{}'.format(rw))
	else:
		print('All tests passed!')
#

   	
		
		
