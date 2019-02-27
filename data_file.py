import datetime
import json

class DataFile(object):
	def __init__(self, pathPrefix, header, log=False):
		self.log = log
		"""
		Whether to log messages to stdout
		"""
		self.header = header
		"""
		Schema for the data
		"""

		path = f"{pathPrefix}_{_dateStr()}.json"
		self.out = self._openOutfile(path, self.header)
		"""
		Output file
		"""

	def write(self, packet):
		self.out.write(",")
		self.out.write( json.dumps(packet) )

	def close(self):
		"""
		"""
		self.out.write("]") #End list
		self.out.close()
		if self.log: print(f"Closed outfile {self.out.name}")

	def _openOutfile(self, filename, header):
		if self.log: print(f"Opening outfile {filename}")
		f = open(filename, "w")
		f.write("[") #Begin root object
		f.write(json.dumps(header))
		return f

def _dateStr():
	return datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S")
