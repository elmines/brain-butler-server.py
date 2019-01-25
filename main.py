import netifaces
import websockets
import json
import asyncio

from server import Server


lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))

bb_server = Server()
async def print_message(websocket, path):
	global bb_server
	async for message in websocket:
		json_obj = json.loads(message)
		reply = bb_server.receivePacket(json_obj)
		if reply:
			reply = {"action": reply};
			print(f"About to send {reply}");
			await websocket.send(json.dumps(reply))


asyncio.get_event_loop().run_until_complete( websockets.serve(print_message, lan_ip, port) )
asyncio.get_event_loop().run_forever()
