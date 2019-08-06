# dopamine_voltammetry.py
### Code developed in Python 3.6.8 to analyze voltametry data 
# Authors:
### Ricardo Erazo
### Johnatan Borland
## Georgia State University
## Neuroscience Institute
Open Source Code
Summer 2019
## Intro:
1. goal 1:  	input -> chunk of data: detect peaks (timepoint), amplitude, duration 		implemented
	       			output -> into the text file 												                      implemented
2. goal 2:		input -> chunk of data: automatically read time-tags and compute 		      implemented
	   					current mean.
	      			output -> into text file 												                          implemented
## data input format requirements:
excel files columns should not have blank spaces where there isn't data. If any blanks exist in raw data (nA), fill the blanks with a reasonable aproximation based on the previous and/or subsequent timepoints. There should be a behavioral tag for each timepoint of interest that will be analyzed in a separate spreadsheet.

Code tested and compatible across OSX and Linux Anaconda distributions. OK!
