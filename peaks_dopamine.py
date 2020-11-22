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
	# 			output -> into the text file 												implemented
	# goal 2:		input -> chunk of data: automatically read time-tags and compute 		implemented
	# 					current mean.
	#				output -> into text file 												implemented
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

import os
## graphix stuff
font = {'weight' : 'bold',
        'size'   : 50}
plt.rc('font', **font)
lw = 3
mk = 20
############################# FUNCTIONS ######################################################

def timebins(t_ini,t_end):
	q0=np.nonzero(time<=t_end)
	q1=np.nonzero(time[q0]>=t_ini)
	t = time[q1]
	R = np.array(I[q1])
	N = np.array(I_norm[q1])
	return t,R, N

def peak_analysis(qk1,tm1,R1,data_acquisition_rate,timebin_baseline):
	peak_eval_threshold = peak_threshold	#threshold for discriminating begin and end of peak
	out1=[]
	out2=[]
	out3=[]
	ev_tol = int(timebin_baseline/2 / data_acquisition_rate)
	for i in range(0,len(qk1)):
		peak = qk1[i]
		# print(ev_tol)
		if peak > ev_tol:
			Ipeak = R1[peak]
			# current peak amplitude:
			variability_1 = np.std(R1[peak-ev_tol:peak])/np.sqrt(ev_tol)
			baseline_1 = np.mean(R1[peak-ev_tol:peak])
			peak_amplitude = (Ipeak-baseline_1)/variability_1	
			# print(variability_1)
			# print(baseline_1)
			# print(peak_amplitude)
			tbin_before = tm1[peak-ev_tol:peak]
			tbin_before = tbin_before[::-1]
			tbin_after = tm1[peak:peak+ev_tol]
		
			Ibin_before = R1[peak-ev_tol:peak] # select timepoints before peak
			Ibin_before = Ibin_before [::-1] # reverse order of points: counting back from peak time
			Ibin_after = R1[peak:peak+ev_tol] # select timepoints after peak
			tbin_after = tm1[peak:peak+ev_tol]
			counter=0
			count2 = 0
			while counter < 10:
				if Ipeak-Ibin_before[count2] > peak_eval_threshold:
					start_peak = tbin_before[count2]
					counter=10
				count2 = count2+1
				counter=counter+data_acquisition_rate*1
			counter = 0
			count2 = 0
			# print(Ibin_after)
			while counter < 10:
				if Ipeak-Ibin_after[count2] > peak_eval_threshold:
					# print(count2)
					end_peak = tbin_after[count2]
					counter=10
				counter=counter+data_acquisition_rate*1
				count2 = count2+1
				# print(counter)

			peak_duration = end_peak-start_peak

			mean1 = R1[peak]

			area_1= peak_duration*peak_amplitude/2
			print('timestamp = '+str(tm1[peak]))
			print('Amplitude = '+str(peak_amplitude))
			print('Duration = '+ str(peak_duration)+'\n')
			
			out1.append(tm1[peak])
			out2.append(peak_amplitude)
			out3.append(peak_duration)

	return out1,out2,out3

def averages(tm1,I,t_ini,t_end):
	av1 = []
	for i in range(0,len(tm1)):
		t = float(tm1[i])
		r1 = float(I[i])
		if t >=t_ini and t<=t_end:
			av1.append(r1)
	# print(av1)
	ave1 = np.mean(av1)
	return ave1

def timelist_(tm2,lb1,t_ini,t_end,timestep):
	nonsoc = []
	soc = []
	agg = []
	groom=[]
	for i in range(0,len(lb1)):
		tm3 = float(tm2[i])
		lb0 = str(lb1[i])
		if tm3<=t_end and tm3>=t_ini:
			# print(tm3)
			if lb0=="['non social']":
				t1 = tm3
				t2 = float(tm2[i+1]-timestep)
				lb10 = 'Non Social'
				nonsoc.append([t1,t2,lb10])
				# print('t1')
				# print(t1)
				# print('t2')
				# print(float(tm2[i+1]))
				# print('diff')
				# print(np.diff(tm2))
			if lb0=="['social investigation']":
				t1 = tm3
				t2 = float(tm2[i+1]-timestep)
				lb10 = 'Social Investigation'
				soc.append([t1,t2,lb10])
				# print(t1,t2)

			if lb0== "['aggression']" or lb0== "['attacks']":
				t1 = tm3
				t2 = float(tm2[i+1]-timestep)
				lb10 = 'Aggression'
				agg.append([t1,t2,lb10])

			if lb0== "['grooming']":
				t1 = tm3
				t2 = float(tm2[i+1]-timestep)
				lb10 = 'Aggression'
				groom.append([t1,t2,lb10])

	return nonsoc,soc,agg, groom

def timetags(x):
	timebins_list = []
	for i in range(0,len(x)):
		x
		tag = x[i]
		timebins_list.append([tag[0],tag[1]])
	return timebins_list

############################## Import Raw DATA ################################################
# social session: tonic
t1 = 1367
t2 = 4356
# social session: peaks
t3 = 2300
t4 = 2600
data_acquisition_rate = 0.2
annotations = '19-107 run2.0 annotations.xls'
file_name = '19-107_run2B_export.xls' # path to file + file name
sheet =  0# sheet name or sheet number or list of sheet numbers and names
# data_cols = "C:E"#columns from excel spreadsheet that will be used.
ev_tol = 20 # timebin length for evaluation : unit seconds -> defines time before any given peak to evaluate and compute a running baseline.

fileID = file_name[0:13]
#import raw data: -> parameter usecol'D' should be 
time = np.array(pd.read_excel(io=file_name, sheet_name=sheet,usecols="D",skiprows=[0,1,2,3,4,5,6,7]))
time = time.flatten()
temp1 = pd.read_excel(io=file_name, sheet_name=sheet,usecols="E",skiprows=[0,1,2,3,4,5,6,7])
I =temp1.values

# import annotations
tm2 = pd.read_excel(io=annotations, sheet_name=sheet,usecols="D",skiprows=[0,1,2,3,4,5,6,7])
#
tm2 = tm2.values
# tm2.flatten()
lb1 = pd.read_excel(io=annotations, sheet_name=sheet,usecols="F",converters={'Annotation':str},skiprows=[0,1,2,3,4,5,6,7])
#Mac 
# lb1 = lb1[6:len(lb1)]
lb2 = lb1.values

# Normalization technique: substract a smoothed curve of the data as a running mean from the raw signal 
I_norm = I.flatten() - savgol_filter(I.flatten(),51,1) # normalized data -> raw data - filtered data
# peak threshold algorithm
peak_threshold = np.sqrt(np.mean(np.square(I_norm))) * 0.4
#plot raw data
plt.figure(1,figsize=(60,25))
plt.plot(time[0:len(I)],I,'.-',markersize=mk,linewidth=lw,label='raw data')
plt.plot(time[0:len(I)],savgol_filter(I.flatten(),51,1),linewidth=lw,label='filtered data "running mean" ')
plt.plot(time[0:len(I_norm)],I_norm,'.-',markersize=mk,linewidth=lw,label='normalized data')
plt.ylabel('nA')
plt.xlabel('seconds')
plt.title('graphical summary of analysis')

plt.ylim([-0.1,0.5])

##################################################### Analyze data ###############################
tm1,R1,N1 = timebins(t3,t4)
R1=R1.flatten()
N1=N1.flatten()
peak_threshold = np.sqrt(np.mean(np.square(N1))) * 0.4
## peaks
Analyze_peaks = True

if Analyze_peaks == True:
	outfile_p = open(fileID+'_peaks.txt','w')
	outfile_p.write('t stamp(s)'+'	'+'amplitude(nA)'+'	'+'duration(s)'+'\n')
	
	qk1,p0 = find_peaks(R1,prominence=peak_threshold)
	
	plt.plot(tm1,R1,'.-',markersize=mk,linewidth=lw,label='timebin at hand')
	plt.plot(tm1[qk1],R1[qk1],'*',markersize=mk,linewidth=lw,color='k')
	
	timestamp,amplitude,duration = peak_analysis(qk1,tm1,R1,data_acquisition_rate,ev_tol)

	for i in range(0,len(timestamp)):
		outfile_p.write(str(timestamp[i])+'	'+str(amplitude[i])+'	'+str(duration[i])+'\n')
	outfile_p.close()
	plt.axis([t3,t4,-0.5,1])
	plt.legend()
	plt.savefig('peaks')

	os.makedirs('signal_repo', exist_ok=True)

	np.save('signal_repo/time2',tm1)
	np.save('signal_repo/R2',R1)



### tonic
Analyze_tonic = False

aggregate_nonsocial=[]
aggregate_social=[]
aggregate_aggression=[]
aggregate_grooming=[]

if Analyze_tonic == True:
	outfile_t = open(fileID+'_tonic.txt','w')
	outfile_t.write('timestamp (ini, end)'+'	'+'y1'+'\n')
	
	tm1,R1,N1 = timebins(t1,t2)

	n,s,a,g=timelist_(tm2,lb2,t1,t2,data_acquisition_rate)

	print('Non Social')
	outfile_t.write('Non Social'+'\n')
	timebins_list = timetags(n)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(tm1,R1,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(ti[0],ti[1],av1)
		outfile_t.write(str(ti[0])+'		')
		outfile_t.write(str(ti[1])+'		')
		outfile_t.write(str(av1)+'\n')

		aggregate_nonsocial.append(ti[1]-ti[0])
	outfile_t.write('Total='+str(sum(aggregate_nonsocial)))
	outfile_t.write('\n')
	outfile_t.write('\n')


	print('Social Investigation')
	outfile_t.write('Social'+'\n')
	timebins_list = timetags(s)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(tm1,R1,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(ti[0],ti[1],av1)
		outfile_t.write(str(ti[0])+'		')
		outfile_t.write(str(ti[1])+'		')
		outfile_t.write(str(av1)+'\n')		

		aggregate_social.append(ti[1]-ti[0])

	outfile_t.write('Total='+str(sum(aggregate_social)))
	outfile_t.write('\n')
	outfile_t.write('\n')

	print('Aggression')
	outfile_t.write('Aggression'+'\n')
	timebins_list = timetags(a)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(tm1,R1,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(ti[0],ti[1],av1)
		outfile_t.write(str(ti[0])+'		')
		outfile_t.write(str(ti[1])+'		')
		outfile_t.write(str(av1)+'\n')		


		aggregate_aggression.append(ti[1]-ti[0])

	outfile_t.write('Total='+str(sum(aggregate_aggression)))
	outfile_t.write('\n')
	outfile_t.write('\n')


	print('Grooming')
	outfile_t.write('Grooming'+'\n')
	timebins_list = timetags(g)

	p1,p2= np.shape(timebins_list)
	for i in range(0,p1):
		ti = timebins_list[i]
		av1 = averages(tm1,R1,ti[0],ti[1])
		plt.plot([ti[0],ti[1]],[av1,av1],linewidth=lw*2,color='k')#,label='behavior_tag_x')
		print(ti[0],ti[1],av1)
		outfile_t.write(str(ti[0])+'		')
		outfile_t.write(str(ti[1])+'		')
		outfile_t.write(str(av1)+'\n')		


		aggregate_grooming.append(ti[1]-ti[0])

	outfile_t.write('Total='+str(sum(aggregate_grooming)))
	# outfile_t.write('\n')
	# outfile_t.write('\n')


	plt.axis([t1,t2,-1,6.2]) 
	plt.legend()
	plt.savefig('tonic')
	outfile_t.close()

plt.legend()
plt.ion()
plt.show()
