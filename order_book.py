"""

	Order Book Data Collection
	==========================

	This script connects to the Binance WebSocket API to receive Level 2 orderbook updates for a specific symbol.
	The script stores the orderbook updates in JSON files at regular intervals.
	The orderbook data is stored in the 'orderbook_data' directory.

	author: jpolec
	data: 2024-04-10
	
"""

import websocket
import json
import time
import os
import logging
import threading

# Configure logging
logging.basicConfig(filename='orderbook.log', level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(message)s')

# Class OrderBookStore --------------------------------------------------------
class OrderBookStore:
	def __init__(self, symbol, depth, storage_path):
		self.symbol = symbol
		self.depth = depth
		self.storage_path = storage_path
		self.orderbook = None
		self.last_update_time = None
		self.max_storage_size = 100  # Maximum number of stored orderbooks
		self.stored_orderbooks = []

	def process_message(self, message):
		"""
  		Update the orderbook with the received message
		"""
		self.orderbook = message
		self.last_update_time = time.time()

		# Log the received message
		logging.info(f"Received orderbook update for {self.symbol}: {message}")

	def store_orderbook(self):
		"""
		Store the current orderbook data in a JSON file.
  		"""
		if self.orderbook:
			timestamp = int(self.last_update_time * 1000)  # Convert to milliseconds
			filename = f"{self.symbol}_{timestamp}.json"
			file_path = os.path.join(self.storage_path, filename)

			with open(file_path, 'w') as file:
				json.dump(self.orderbook, file)

			self.stored_orderbooks.append(file_path)

			# Remove the oldest stored orderbook if the maximum storage size is exceeded
			if len(self.stored_orderbooks) > self.max_storage_size:
				oldest_file = self.stored_orderbooks.pop(0)
				os.remove(oldest_file)

			logging.info(f"Orderbook stored: {file_path}")


def main():

	symbol = 'BTCUSDT'
	depth = 20
	storage_path = 'orderbook_data'

	# Create the storage directory if it doesn't exist
	os.makedirs(storage_path, exist_ok=True)

	orderbook_store = OrderBookStore(symbol, depth, storage_path)

	def on_message(ws, message):
		"""
		Process the received message from the WebSocket connection.
  		"""
		message = json.loads(message)
		orderbook_store.process_message(message)

	def on_error(ws, error):
		logging.error(f"WebSocket error: {error}")

	def on_close(ws, close_status_code, close_msg):
		logging.info("WebSocket connection closed")

	def on_open(ws):
		"""
		Subscribe to the Level 2 orderbook updates for a specific symbol when the WebSocket connection is opened.
  		"""
		logging.info("WebSocket connection opened")
		# Subscribe to the Level 2 orderbook updates for a specific symbol
		ws.send(json.dumps({
			'method': 'SUBSCRIBE',
			'params': [
				f'{symbol.lower()}@depth{depth}'  # Top N levels of depth
			],
			'id': 1
		}))

	def periodic_storage():
		"""
		Periodically store the orderbook data at regular intervals.
  		"""
		while True:
			time.sleep(storage_interval)  # Sleep before the next run
			print(f"Storing orderbook for {orderbook_store.symbol}")
			orderbook_store.store_orderbook()

	socket = "wss://stream.binance.com:9443/ws"
	ws = websocket.WebSocketApp(socket,
								on_open=on_open,
								on_message=on_message,
								on_error=on_error,
								on_close=on_close)

	# Setup threading for periodic storage
	storage_interval = 5  # seconds
	storage_thread = threading.Thread(target=periodic_storage)
	storage_thread.start()

	ws.run_forever(ping_interval=10, ping_timeout=5)
	
if __name__ == "__main__":
	
   main()