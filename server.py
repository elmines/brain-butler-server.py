import asyncio
import websockets
import netifaces

lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))

async def print_message(websocket, path):
    async for message in websocket:
        print(message)

asyncio.get_event_loop().run_until_complete(
    websockets.serve(print_message, lan_ip, port))
asyncio.get_event_loop().run_forever()
