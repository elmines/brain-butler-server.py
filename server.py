import json
import asyncio
import websockets
import netifaces
import datetime

#Local libraries
import eeg
import rl

lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))


class Server(object):
	"""
	A single-threaded server for receiving data packets
	"""

	def __init__(self):
		self.agent = None
		self.curr_file = None
		self.header_provided = None
		self.data_records = 0

	def processData(self, packet):
		if self.data_records > 0: self.curr_file.write(",")
		self.curr_file.write( json.dumps(packet["body"])[1:-1] )
		self.data_records += 1
		if eeg.errp(packet["body"]): return self.agent.act(None)

	def processEvent(self, packet):
		if self.data_records > 0: self.curr_file.write(",")
		self.curr_file.write( json.dumps(packet["body"]) ) #Keep the curly braces
		self.data_records += 1


	def receive_packet(self, packet):
		"""
		Description

		:param dict(str, obj) packet: The header, data, etc., packet

		:rtype: str
		:returns: The action that the RL agent is taking,
			or None if the agent is not taking a formal action
		"""
		if packet["type"] == "header":
			self._reset_globals()
			self.header_provided = True
			self.curr_file = Server._begin_file()
			self.curr_file.write('\"header\":{}'.format( json.dumps(packet["body"]) ))
			self.curr_file.write(', \"data\":[')
			return None
		if packet["type"] == "eof":
			self._reset_globals()
			return None

		if not self.header_provided:
			print("Header has not been provided--rejecting record")
			return None

		if packet["type"] == "data":
			return self.processData(packet)
		if packet["type"] == "event":
			return self.processEvent(packet)

		print(f'Received packet of unknown type {packet["type"]}--rejecting')
		return None


	#PRIVATE
	def _reset_globals(self):
		if self.curr_file: Server._end_file(self.curr_file)
		self.curr_file = None
		self.header_provided = False
		self.data_records = 0
		self.agent = rl.Agent()
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
		reply = bb_server.receive_packet(json_obj)
		if reply:
			reply = {"action": reply};
			print(f"About to send {reply}");
			await websocket.send(json.dumps(reply))


asyncio.get_event_loop().run_until_complete( websockets.serve(print_message, lan_ip, port) )
asyncio.get_event_loop().run_forever()

