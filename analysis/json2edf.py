#!/usr/bin/python3

import pyedflib

import datetime
import numpy as np
import json
import sys

def parse_edf_datetime(edf_date, edf_time):
	return datetime.datetime.strptime(f"{edf_date} {edf_time}", "%d.%m.%Y %H.%M.%S")

def process_header(writer, header):
	for (i, label) in enumerate(header["labels"]):
		writer.setLabel(i, label)
	for (i, freq) in enumerate(header["sampleFrequency"]):
		writer.setSamplefrequency(i, freq)
	for (i, prefilter) in enumerate(header["prefilter"]):
		writer.setPrefilter(i, prefilter);
	for (i, dimension) in enumerate(header["physicalDimension"]):
		writer.setPhysicalDimension(i, dimension)
	writer.setStartdatetime( parse_edf_datetime(header["startDate"], header["startTime"]) )
	writer.setPatientCode(header["patientCode"])


assert len(sys.argv) >= 3

json_path = sys.argv[1]

if len(sys.argv) >= 3: edf_path = sys.argv[2]
else:                  pass

with open(json_path, "r") as r: json_dict = json.load(r)

num_channels = len(json_dict["header"]["labels"])
writer = pyedflib.EdfWriter(edf_path, num_channels)
process_header(writer, json_dict["header"])

data = np.array( json_dict["data"] ).transpose()
writer.writeSamples(data)

writer.close()
