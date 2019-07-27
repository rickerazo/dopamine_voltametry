# dopamine_voltammetry.py
#################### Code developed in Python 3.6.8 to analyze voltametry data 
# Ricardo Erazo
# Johnatan Borland
# Georgia State University
# Neuroscience Institute
# Open Source Code
# Summer 2019
################################	Intro:
	# goal 1:  	input -> chunk of data: detect peaks (timepoint), amplitude, duration 		implemented
	# 			output -> back into the excel file 											TBD
	# goal 2:		input -> chunk of data: automatically read time-tags and compute 		implemented
	# 					current mean 
	#				output -> back into excel file 											TBD
#	data input format requirements: excel files columns should not have blank spaces where
#		there isn't data. If any blanks exist in raw data (nA), fill the blanks with a
# 		reasonable aproximation based on the previous and/or subsequent timepoints.
#		There should be a behavioral tag for each timepoint of interest that will be analyzed
#		in a separate spreadsheet

################################### Import Libraries
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
## graphix stuff
font = {'weight' : 'bold',
        'size'   : 50}
plt.rc('font', **font)
lw = 3
mk = 10
############################# FUNCTIONS ######################################################

def timebins(t_ini,t_end):
	q0=np.nonzero(time<=t_end)
	q1=np.nonzero(time[q0]>=t_ini)
	t = time[q1]
	R = np.array(I[q1])
	N = np.array(I_norm[q1])
	return t,R, N

def peak_analysis(qk1,tm1,R1):
	peak_eval_threshold = 0.1	#threshold for discriminating begin and end of peak
	for i in range(0,len(qk1)):
		peak = qk1[i]
		Ipeak = R1[peak]

		# current peak amplitude:
		variability_1 = np.std(R1[peak-ev_tol:peak])/np.sqrt(ev_tol)
		baseline_1 = np.mean(R1[peak-ev_tol:peak])
		peak_amplitude = (Ipeak-baseline_1)/variability_1	

		# ref1 = int(tm1[peak])
		tbin_before = tm1[peak-ev_tol:peak]
		tbin_before = tbin_before[::-1]
		tbin_after = tm1[peak:peak+ev_tol]

		# plt.plot([tm1[peak-ev_tol],tm1[peak] ],[baseline_1,baseline_1])
		Ibin_before = R1[peak-ev_tol:peak] # select timepoints before peak
		Ibin_before = Ibin_before [::-1] # reverse order of points: counting back from peak time
		Ibin_after = R1[peak:peak+ev_tol] # select timepoints after peak
		tbin_after = tm1[peak:peak+ev_tol]
		counter=0

		print(Ipeak)
		print(peak-ev_tol)
		print(R1[peak-ev_tol:peak])
		while counter < 10:
			if Ipeak-Ibin_before[counter] > peak_eval_threshold:
				# start_peak = Ibin_before[counter]
				start_peak = tbin_before[counter]
				# print(start_peak)
				# mean_I_before = np.mean(tbin_before[0:])
				# print(counter)
				counter=10
			counter=counter+1

		counter = 0

		while counter < 10:
			if Ipeak-Ibin_after[counter] > peak_eval_threshold:
				# end_peak = Ibin_after[counter]
				end_peak = tbin_after[counter]
				# print(end_peak)
				counter=10
			counter=counter+1

		peak_duration = end_peak-start_peak
		# print(peak_duration)

		mean1 = R1[peak]

			#area under curve:
		#triangle formula, when the spike has a triangular shape
		area_1= peak_duration*peak_amplitude/2
		print('timestamp='+str(tm1[peak]))
		print('Amplitude '+str(peak_amplitude))
		print('Duration '+ str(peak_duration)+'\n')
	return tm1[peak],peak_amplitude,peak_duration

def averages(I,t_ini,t_end):
	ave1 = np.mean(I[t_ini:t_end])
	return ave1

def timelist_(tm2,lb1):
	conter = 0
	LB = []
	nonsoc = []
	soc = []
	agg = []

	for i in range(0,len(lb1)):
		tm3 = int(tm2[i])
		lb0 = str(lb1[i])
		counter = 0
		if lb0=="['non social']":
			t1 = tm3
			t2 = int(tm2[i+1])-1
			lb10 = 'Non Social'
			nonsoc.append([t1,t2,lb10])
		
		if lb0=="['social investigation']":
			t1 = tm3
			t2 = int(tm2[i+1])-1
			lb10 = 'Social Investigation'
			soc.append([t1,t2,lb10])
			# print(t1,t2)

		if lb0== "['aggression']":
			t1 = tm3
			t2 = int(tm2[i+1])-1
			lb10 = 'Aggression'
			agg.append([t1,t2,lb10])

		LB.append(lb0)
	return nonsoc,soc,agg

def timetags(x):
	timebins_list = []
	for i in range(0,len(x)):
		x
		tag = x[i]
		timebins_list.append([tag[0],tag[1]])
	return timebins_list

############################## Import Raw DATA ################################################
# social session: tonic
t1 = 3096
t2 = 4199
# social session: peaks
t3 = 3080
t4 = 4199

annotations = '19-107_run2B_annotations.xls'
file_name = '19-107_run2B_export.xls' # path to file + file name
sheet =  0# sheet name or sheet number or list of sheet numbers and names
# data_cols = "C:E"#columns from excel spreadsheet that will be used.
ev_tol = 20 # timebin length for evaluation 

#import raw data: -> parameter usecol'D' should be 
time = np.array(pd.read_excel(io=file_name, sheet_name=sheet,usecols="D"))
time = time.flatten()
temp1 = pd.read_excel(io=file_name, sheet_name=sheet,usecols="E")
temp1 = temp1+1-1
I =temp1.values
# import annotations
tm2 = pd.read_excel(io=annotations, sheet_name=sheet,usecols="D")
tm2 = tm2.values
lb1 = pd.read_excel(io=annotations, sheet_name=sheet,usecols="F",converters={'Annotation':str})
lb2 = lb1.values
# Normalization technique: substract a smoothed curve of the data as a running mean from the raw signal 
I_norm = I.flatten() - savgol_filter(I.flatten(),51,1) # normalized data -> raw data - filtered data
# peak threshold algorithm 
peak_threshold = np.sqrt(np.mean(np.square(I_norm))) * 0.2
#plot raw data
plt.figure(1,figsize=(60,25))
plt.plot(time[0:len(I)],I,'.-',markersize=mk,linewidth=lw,label='raw data')
plt.plot(time[0:len(I)],savgol_filter(I.flatten(),51,1),linewidth=lw,label='filtered data "running mean" ')
plt.plot(time[0:len(I_norm)],I_norm,'.-',markersize=mk,linewidth=lw,label='normalized data')
plt.ylabel('nA')
plt.xlabel('seconds')
plt.title('graphical summary of analysis')


##################################################### Analyze data ###############################
tm1,R1,N1 = timebins(t3,t4)
R1=R1.flatten()

## peaks
Analyze_peaks = False
if Analyze_peaks == True:
	
	qk1,p0 = find_peaks(R1,prominence=peak_threshold)
	
	plt.plot(tm1,R1,'.-',markersize=mk,linewidth=lw,label='timebin at hand')
	plt.plot(tm1[qk1],R1[qk1],'*',markersize=mk,linewidth=lw,color='red')
	
	timestamp,amplitude,duration = peak_analysis(qk1,tm1,R1)
	plt.savefig('peaks')

### tonic
Analyze_tonic = True
if Analyze_tonic == True:
	tm1,R1,N1 = timebins(t1,t2)
	n,s,a=timelist_(tm2,lb2)

	print('Non Social')
	timebins_list = timetags(n)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(I,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(av1)
		plt.axis([t3,t4,-1,6.2]) 
		plt.savefig('tonic')

	print('Social Investigation')
	timebins_list = timetags(s)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(I,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(av1)
		plt.axis([t3,t4,-1,6.2]) 
		plt.savefig('tonic')


	print('Aggression')
	timebins_list = timetags(a)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(I,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(av1)
		plt.axis([t3,t4,-1,6.2]) 
		plt.savefig('tonic')

plt.axis([t3,t4,-1,6.2]) 
plt.legend()
plt.ion()
plt.show()

########################################## Save output to excel file