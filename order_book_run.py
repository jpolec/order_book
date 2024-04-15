"""

	Order Book Simulation
	=====================
	This module provides a simple simulation of an order book for a financial market.
	It includes classes for representing market events, price levels, and the order book itself.
	The order book can process events to update its state and provide information such as the best bid and ask prices.
	It also includes functions for visualizing the order book using matplotlib.

	The `Event` class represents a market event with a timestamp, sequence number, trade indicator, buy/sell indicator, price, and size.
	The `Level` class represents a price level in the order book with a price and size.
	The `OrderBook` class represents the order book itself and includes methods for processing events, updating the order book, and retrieving information such as the best bid and ask prices.


	author: jpolec
	data: 2024-04-10
	
"""


import json
import os
import sys
import time
from dataclasses import dataclass

from typing import Optional
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import seaborn as sns

import websocket
import logging
import threading

from sortedcontainers import SortedDict

# Dynamically add the grandparent directory to the Python path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Configure logging
logging.basicConfig(filename='orderbook.log', level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(message)s')

# Dev Lib for testing (remove in production)
import _dev.dev

# Event class --------------------------------------------------------
@dataclass
class Event:
	"""
 	A class to represent a market event.
  	"""
	timestamp: int
	seq: int
	is_trade: bool
	is_buy: bool
	price: float
	size: float

	def price_ticks(self, tick_size: float) -> int:
		"""
  		Convert the price to an integer number of ticks.
		"""
		return int(self.price / tick_size)


# Level class --------------------------------------------------------
@dataclass
class Level:
	"""
 	A class to represent a price level in the order book.
 	"""
	price: float
	size: float

	def __post_init__(self):
		"""
  		Ensure that size is positive.
		"""
		#self.size = float(self.size)
  
		# if self.size <= 0:
		# 	raise ValueError("Size must be greater than zero")


# OrderBook class --------------------------------------------------------
class OrderBook:
	"""
 	A class to represent the order book.
  	"""
	def __init__(self, tick_size: float, json_path: str = None):
		"""
  		Initialize the order book with tick size and optional path to a JSON file.
		"""
		self.bids = SortedDict()
		self.asks = SortedDict()
		self.tick_size = tick_size
		self.last_updated = 0
		self.last_sequence = 0
		if json_path:
			self.load_orderbook(json_path)


	def load_orderbook(self, path: str) -> None:
		"""
  		Load order book data from a JSON file.
  		"""
		try:
			with open(path, 'r') as file:
				data = json.load(file)
				for bid in data.get('bids', []):
					self.bids[bid[0]] = Level(price=bid[0], size=bid[1])
				for ask in data.get('asks', []):
					self.asks[ask[0]] = Level(price=ask[0], size=ask[1])
			print(f"Loaded orderbook from {path}")
		except FileNotFoundError:
			print(f"Orderbook file {path} not found.")


	def update(self, data: dict):
		"""
		Update the order book from WebSocket data.
		"""
		# Clear previous data for simplicity in this example
		self.bids.clear()
		self.asks.clear()
		for bid in data['bids']:
			price, size = float(bid[0]), float(bid[1])
			if size > 0:
				self.bids[price] = size
		for ask in data['asks']:
			price, size = float(ask[0]), float(ask[1])
			if size > 0:
				self.asks[price] = size
	
		print("Updated bids:", self.bids)  # Debug statement
		print("Updated asks:", self.asks)  # Debug statement


	def process(self, event: Event) -> None:
		"""
  		Process an event to update the order book.
		"""
		if event.timestamp < self.last_updated or event.seq < self.last_sequence:
			return  # Skip events that are out of order

		price_ticks = event.price_ticks(self.tick_size)

		if event.is_trade:
			self.process_trade(event, price_ticks)
		else:
			self.process_level2(event, price_ticks)

		self.last_updated = event.timestamp
		self.last_sequence = event.seq


	def process_level2(self, event: Event, price_ticks: int) -> None:
		"""
  		Update or remove a price level based on the event.
		"""
		book = self.bids if event.is_buy else self.asks
		if event.size == 0:
			book.pop(price_ticks, None)
		else:
			book[price_ticks] = Level(price=event.price, size=event.size)


	def process_trade(self, event: Event, price_ticks: int) -> None:
		"""
  		Process a trade event by adjusting sizes or removing levels.
		"""
		book = self.bids if not event.is_buy else self.asks
		if price_ticks in book and book[price_ticks].size >= event.size:
			book[price_ticks].size -= event.size
			if book[price_ticks].size == 0:
				del book[price_ticks]


	def get_best_bid(self) -> Optional[Level]:
		"""
  		Return the highest bid from the order book.
		"""
		if self.bids:
			return self.bids.peekitem(-1)[1]
		return None


	def get_best_ask(self) -> Optional[Level]:
		"""
  		Return the lowest ask from the order book.
		"""
		if self.asks:
			return self.asks.peekitem(0)[1]
		return None


	def snapshot(self) -> dict:
		"""
  		Create a snapshot of the current state of the order book.
		"""
		return {
			"bids": [(level.price, level.size) for level in self.bids.values()],
			"asks": [(level.price, level.size) for level in self.asks.values()]
		}


	def midprice(self) -> Optional[float]:
		"""
  		Calculate the midprice between the best bid and ask.
		"""
		best_bid = self.get_best_bid()
		best_ask = self.get_best_ask()
		if best_bid and best_ask:
			return (best_bid.price + best_ask.price) / 2.0
		return None


	def weighted_midprice(self) -> Optional[float]:
		"""
  		Calculate the weighted midprice based on sizes and prices.
		"""
		best_bid = self.get_best_bid()
		best_ask = self.get_best_ask()
		if best_bid and best_ask:
			num = best_bid.size * best_ask.price + best_bid.price * best_ask.size
			den = best_bid.size + best_ask.size
			return num / den
		return None


# RealTimePlotter class --------------------------------------------------------
class RealTimePlotter:
	def __init__(self, order_book):
		self.order_book = order_book
		self.fig, self.ax = plt.subplots()
		self.ani = None
		self.fixed_range_center = None
		self.fixed_range_width = None
		sns.set_style("whitegrid")


	def update_plot(self, i: int) -> None:
		
		self.ax.clear()
		self.ax.set_ylim(0, 5) 
	
		# Check if there is data to plot
		if self.order_book.bids and self.order_book.asks:
			# Get the bids and asks sorted by price
			bid_prices, bid_sizes = zip(*sorted(self.order_book.bids.items(), reverse=True))
			ask_prices, ask_sizes = zip(*sorted(self.order_book.asks.items()))

			# Plot the bids and asks
			self.ax.bar(bid_prices, bid_sizes, width=0.3, color='green', label='Bids')
			self.ax.bar(ask_prices, ask_sizes, width=0.3, color='red', label='Asks')

			# Calculate and display the spread
			spread = ask_prices[0] - bid_prices[0]
			self.ax.annotate(f'Spread: {spread:.2f}', 
							xy=((ask_prices[0] + bid_prices[0]) / 2, 0), 
							xytext=(0, 20), 
							textcoords='offset points', 
							arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), 
							ha='center', va='bottom')

			# Determine the center and width for the fixed range on the first data set
			if self.fixed_range_center is None or self.fixed_range_width is None:
				self.fixed_range_center = (min(bid_prices) + max(ask_prices)) / 2
				self.fixed_range_width = max(ask_prices) - min(bid_prices) + 25

			self.ax.set_xlim(
				self.fixed_range_center - self.fixed_range_width / 2,
				self.fixed_range_center + self.fixed_range_width / 2
			)

			# Set the x-axis to show more numbers (ticks)
			self.ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))

		# Add labels and legend
		self.ax.set_xlabel('Price')
		self.ax.set_ylabel('Size')
		self.ax.legend()

		# Invert the x-axis to have the higher prices on the right
		self.ax.invert_xaxis()


	def update_plot_with_sns(self, i: int) -> None:
	 
		self.ax.clear()
		self.ax.set_ylim(0, 5)  # Fixed y-axis range

		# Only plot if there is data
		if self.order_book.bids and self.order_book.asks:
			bid_prices, bid_sizes = zip(*sorted(self.order_book.bids.items(), reverse=True))
			ask_prices, ask_sizes = zip(*sorted(self.order_book.asks.items()))

			# Plot aesthetics
			bid_color = sns.color_palette("deep")[2]  # Green-like color
			ask_color = sns.color_palette("deep")[3]  # Red-like color
			
			self.ax.bar(bid_prices, bid_sizes, width=0.3, color=bid_color, label='Bids')
			self.ax.bar(ask_prices, ask_sizes, width=0.3, color=ask_color, label='Asks')

			# Add spread annotation
			spread = ask_prices[0] - bid_prices[0]
			# self.ax.annotate(f'Spread: {spread:.2f}', xy=((ask_prices[0] + bid_prices[0]) / 2, max(bid_sizes)), 
			# 				 ha='center', va='bottom', color='blue')

			# Calculate the current required range to display all data
			current_range_center = (min(bid_prices) + max(ask_prices)) / 2
			current_range_width = (max(ask_prices) - min(bid_prices)) * 1.1  # 10% padding

			# Determine fixed range if not set
			if not self.fixed_range_center or not self.fixed_range_width:
				self.fixed_range_center = current_range_center
				self.fixed_range_width = current_range_width

			# Update range if current data is beyond the fixed range
			if current_range_width > self.fixed_range_width or \
				abs(current_range_center - self.fixed_range_center) > self.fixed_range_width / 2:
				self.fixed_range_center = current_range_center
				self.fixed_range_width = max(self.fixed_range_width, current_range_width)

			# Apply fixed range to x-axis
			self.ax.set_xlim(
				self.fixed_range_center - self.fixed_range_width / 2,
				self.fixed_range_center + self.fixed_range_width / 2
			)

			# Add more ticks on x-axis for better granularity
			self.ax.xaxis.set_major_locator(MaxNLocator(nbins=10, steps=[1, 2, 5, 10]))

			# Invert x-axis
			self.ax.invert_xaxis()

		# Labels and Legend
		self.ax.set_xlabel('Price')
		self.ax.set_ylabel('Size')
		self.ax.legend(loc='upper right')


	def start_animation(self):
		self.ani = FuncAnimation(self.fig, self.update_plot_with_sns, interval=500)


	def save_animation(self, filename: str, fps: int = 1):
		if self.ani:
			self.ani.save(filename, writer='ffmpeg', fps=fps)
		else:
			print("Animation not started or no frames to save.")



# WebSocketHandler class --------------------------------------------------------
class WebSocketHandler:
	"""
	A class to handle WebSocket connections for real-time order book updates.
	"""
	def __init__(self, url: str, order_book: 'OrderBook', symbol: str = 'BTCUSDT', depth: int = 20):
		self.ws = websocket.WebSocketApp(url, on_open=self.on_open, on_message=self.on_message)
		self.url = url
		self.order_book = order_book
		self.symbol = symbol.lower()
		self.depth = depth

  
	def on_open(self, ws):
		print("WebSocket connection opened")  # Debug statement
		ws.send(json.dumps({
			'method': 'SUBSCRIBE',
			'params': [f'{self.symbol}@depth{self.depth}'],
			'id': 1
		}))
		print("Subscription message sent")  # Debug statement
		
  
	def on_message(self, ws, message):
		print("Received message:", message)  
		data = json.loads(message)
		if 'bids' in data and 'asks' in data:
			print("Received data:", data)  # Debug statement
			self.order_book.update(data)
		else:
			print("Data does not contain 'bids' and 'asks'")


	def start(self):
		"""
		Start the WebSocket in a separate thread.
		"""
		print("Starting WebSocket connection...")  # Debug statement
		self.ws = websocket.WebSocketApp(self.url,
										on_open=self.on_open,
										on_message=self.on_message,
										on_error=self.on_error,
										on_close=self.on_close)
		threading.Thread(target=self.ws.run_forever).start()

	def on_error(self, ws, error):
		print("WebSocket error:", error)  # Debug statement
		
	
	def on_close(self, ws):
		print("WebSocket connection closed")  # Debug statement
		


# Other functions --------------------------------------------------------
def plot_order_book(bids: 'SortedDict', asks: 'SortedDict', last_price: Optional[float] = None, title: Optional[str] = None) -> None:
	fig, ax = plt.subplots(figsize=(10, 6))

	# Consistent color scheme
	bid_color = sns.color_palette("deep")[2]
	ask_color = sns.color_palette("deep")[3]

	# Plot bids and asks
	ax.bar(bids.index, bids.values, width=0.1, color=bid_color, label='Bids')
	ax.bar(asks.index, asks.values, width=0.1, color=ask_color, label='Asks')

	# Add last price line
	if last_price:
		ax.axvline(last_price, color='black', linestyle='--', linewidth=1, label='Last Price')

	# Add spread text
	spread = asks.index.min() - bids.index.max()
	ax.annotate(f'Spread: {spread:.2f}', xy=(0.5, 0.95), xycoords='axes fraction', 
				ha='center', va='center', fontsize=10, color='blue')

	# Labels, legends, and title
	ax.set_xlabel('Price', fontsize=12)
	ax.set_ylabel('Size', fontsize=12)
	ax.legend(fontsize=10)
	if title:
		ax.set_title(title, fontsize=14)

	# Consistent axis scale
	ax.set_xlim([bids.index.min() - spread, asks.index.max() + spread])

	# Font sizes for ticks
	ax.tick_params(axis='both', which='major', labelsize=10)

	# Grid
	ax.grid(True)

	sns.despine(left=True)
	plt.tight_layout()
	plt.show()
  

# Unit Tests --------------------------------------------------------
class UnitTests(Enum):
	MIDPRICE = 1
	WEIGHTED_MIDPRICE = 2
	BEST_BID = 3
	BEST_ASK = 4
	SNAPSHOT = 5
	PROCESS = 6
	PLOT = 7
	REALTIME_PLOT = 8
	
 
def run_unit_test(unit_test: UnitTests):
	
	current_dir = os.path.dirname(os.path.abspath(__file__)) 
	directory = os.path.join(current_dir, '..', 'orderbook_data')  
 
	files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json') and 'BTCUSDT' in f]
	latest_file = max(files, key=os.path.getmtime)
 
	order_book = OrderBook(0.01, latest_file)

 
	if unit_test == UnitTests.MIDPRICE:
		print("Midprice:", order_book.midprice())
	elif unit_test == UnitTests.WEIGHTED_MIDPRICE:
		print("Weighted Midprice:", order_book.weighted_midprice())	
	elif unit_test == UnitTests.BEST_BID:
		print("Best Bid:", order_book.get_best_bid())
	elif unit_test == UnitTests.BEST_ASK:
		print("Best Ask:", order_book.get_best_ask())
	elif unit_test == UnitTests.SNAPSHOT:
		print("Snapshot:", order_book.snapshot())
	elif unit_test == UnitTests.PROCESS:
		# Test the process function with some sample events
		events = [
			Event(1, 1, False, True, 100.0, 10.0),
			Event(2, 2, False, False, 101.0, 5.0),
			Event(3, 3, True, True, 100.0, 3.0)
		]
		# for event in events:	
		# 	order_book.process(event)
   
	elif unit_test == UnitTests.PLOT:
	 
		order_book = OrderBook(0.01)	
		plot_order_book(files, 0.01, interval=1000)
  
	elif unit_test == UnitTests.REALTIME_PLOT:
	 
		print("Starting real-time order book plot...")

		order_book = OrderBook(0.01)				# Create an order book with a tick size of 0.01
		plotter = RealTimePlotter(order_book)		# Create a real-time plotter for the order book

		symbol = 'BTCUSDT'
		ws_handler = WebSocketHandler(url="wss://stream.binance.com:9443/ws", order_book=order_book, symbol=symbol)
		ws_handler.start()

		# To display the real-time plot
		plotter.start_animation()
  
		time.sleep(3)
  
		# To save the animation
		plotter.save_animation('orderbook.mp4', fps=50)

	else:
		print("Invalid unit test specified.")


if __name__ == '__main__':
	
	unit_test = UnitTests.REALTIME_PLOT
	is_run_all_tests = False
 
	if is_run_all_tests:
		for unit_test in UnitTests:
			run_unit_test(unit_test=unit_test)
	else:
		run_unit_test(unit_test=unit_test)