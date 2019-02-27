#!/usr/bin/python3

import json
import argparse
import sys


def convert(jsonPath, signalOut=None, eventOut=None):
	if not(signalOut or eventOut): return

	with open(jsonPath, "r") as r:
		jsonData = json.load(r)

	signalNames = jsonData["header"]["labels"]
	#FIXME: Cheap hack until I separate signal and event labels
	eventNames = signalNames[-2:]
	signalNames = signalNames[:-2] 

	startTimestamp = jsonData["header"]["timestamp"]

	s = open(signalOut, "w") if signalOut else None
	e = open(eventOut, "w") if eventOut else None

	if s: s.write( ",".join(["latency"]+signalNames) )
	if e: e.write( ",".join(["latency", "event"]) )

	for record in jsonData["data"]:
		latency = str(record["timestamp"] - startTimestamp)
		if "data" in record:
			if s:
				data = [str(val) for val in record["data"]]
				s.write("\n" + ",".join( [latency] + data ) )
		elif e:
			eventName = record["eventName"]
			eventValue = record[eventName]
			e.write("\n" + ",".join( [latency] + [f'{eventName}:{eventValue}'] ) )
	if s: s.close()
	if e: e.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Convert an EEG json file to CSV file(s)")
	parser.add_argument("--data", default="data.json", help="Source data file")
	parser.add_argument("--signal", default="signal.csv", help="Output file for signal data")
	parser.add_argument("--event", default="event.csv", help="Output file for event data")
	args = parser.parse_args()

	convert(args.data, args.signal, args.event)
