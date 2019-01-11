import json
import asyncio
import websockets
import netifaces
import datetime

lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))


class Server(object):
	"""
	A single-threaded server for receiving data packets
	"""

	def __init__(self):
		self.curr_file = None
		self.header_provided = None
		self.data_records = 0

	def receive_packet(self, packet):
		"""
		Description

		:param dict(str, obj) packet: The header, data, etc., packet
		"""
		if packet["type"] == "header":
			self._reset_globals()
			self.header_provided = True

			self.curr_file = Server._begin_file()
			self.curr_file.write('\"header\":{}'.format( json.dumps(packet["body"]) ))
			self.curr_file.write(', \"data\":[')
			
		elif packet["type"] == "eof":
			self._reset_globals()
		else:
			if not self.header_provided:
				print("Header has not been provided--rejecting data record")
				return

			if self.data_records > 0: self.curr_file.write(",")
			self.curr_file.write( json.dumps(packet["body"])[1:-1] )	
			self.data_records += 1


	#PRIVATE
	def _reset_globals(self):
		if self.curr_file: Server._end_file(self.curr_file)
		self.curr_file = None
		self.header_provided = False
		self.data_records = 0
	@staticmethod
	def _begin_file():
		curr_file = Server._new_outfile()
		curr_file.write("{") #Begin root object
		return curr_file
	@staticmethod
	def _new_outfile():
		filename = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S") + ".json"
		print(f"Opening outfile {filename}")
		return open(filename, "w")
	@staticmethod
	def _end_file(data_file):
		data_file.write("]") #End data
		data_file.write("}") #End root object
		data_file.close()
		print(f"Closed outfile {data_file.name}")
	
		
		
bb_server = Server()
async def print_message(websocket, path):
	global bb_server
	async for message in websocket:
		json_obj = json.loads(message)
		bb_server.receive_packet(json_obj)
			

asyncio.get_event_loop().run_until_complete( websockets.serve(print_message, lan_ip, port) )
asyncio.get_event_loop().run_forever()
