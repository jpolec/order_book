"""

	Order Book Analysis
	=====================

	This module provides classes for analyzing and visualizing order book data from a cryptocurrency exchange.
	Includes classes for representing market events, price levels, and the order book itself.
 	The order book can process events to update its state and provide information such as the best bid and ask prices.	
	It also includes functions for visualizing the order book using matplotlib.

	The `Event` class represents a market event with a timestamp, sequence number, trade indicator, buy/sell indicator, price, and size.
	The `Level` class represents a price level in the order book with a price and size.
	The `OrderBook` class represents the order book itself and includes methods for processing events, updating the order book, and retrieving information such as the best bid and ask prices.
	
	For OrderBook class:
		- The order book is initialized with a tick size and an optional path to a JSON file.
		- The load_orderbook method loads order book data from a JSON file.
		- The update method updates the order book from WebSocket data.
		- The process method processes an event to update the order book.
		- The get_best_bid method returns the highest bid from the order book.
		- The get_best_ask method returns the lowest ask from the order book.
		- The snapshot method creates a snapshot of the current state of the order book.
		- The store_snapshot method stores a snapshot of the order book at regular intervals.
		- The analyze_historical_data method performs analysis on historical snapshots.
		- The midprice method calculates the midprice between the best bid and ask.
		- The weighted_midprice method calculates the weighted midprice based on sizes and prices.
		- The calculate_wvap method calculates the weighted volume average price (WVAP) over a given time interval.
		- The simulate_market_impact method simulates market impact for a given trade size and direction.
		- The detect_mispricing method detects mispricing based on an external price.
		
	For OrderBookAnalysis class (also using CCXT):
		- The `OrderBookAnalysis` class provides methods for analyzing order book data from a cryptocurrency exchange.
		- The `fetch_order_book` method fetches the order book data from the exchange.
		- The `analyze_order_book` method analyzes the order book data.
		- The `depth_analysis` method calculates the total depth of the order book.
		- The `imbalance_analysis` method calculates the order book imbalance.
		- The `clustering_analysis` method performs clustering analysis on the order book data.
		- The `slope_analysis` method calculates the slopes of the order book profiles.
		- The `fair_value_strategy` method implements a fair value trading strategy based on the order book analysis.
		- The `imbalance_clustering_strategy` method implements a trading strategy based on order book imbalance and clustering.
		- The `visualize_order_book` method visualizes the order book data.

	author: jpolec
	data: 2024-04-14 & 2024-04-16
	
"""


import json
import os
import sys
import time
import datetime
from dataclasses import dataclass
import numpy as np

from typing import Optional, Tuple, Dict
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import seaborn as sns

import websocket
import logging
import threading

from sortedcontainers import SortedDict

from sklearn.cluster import KMeans
from scipy.stats import linregress

import ccxt

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
		# if self.size <= 0:
		# 	raise ValueError("Size must be greater than zero")


	@classmethod
	def minimum(cls):
		return cls(price=float('-inf'), size=0.0)

	@classmethod
	def maximum(cls):
		return cls(price=float('inf'), size=0.0)

	def __lt__(self, other):
		if self.price == other.price:
			return self.size < other.size
		return self.price < other.price

	def __eq__(self, other):
		return self.price == other.price and self.size == other.size

	def __str__(self):
		return f"({self.price} : {self.size})"

	@classmethod
	def from_event(cls, event: Event):
		return cls(price=event.price, size=event.size)


# OrderBook class --------------------------------------------------------
class OrderBook:
	"""
 	A class to represent the order book.
  	"""
	def __init__(self, tick_size: float, json_path: str = None, history=False):
		"""
  		Initialize the order book with tick size and optional path to a JSON file.
		"""
		self.bids = SortedDict()
		self.asks = SortedDict()
		self.tick_size = tick_size
		self.last_updated = 0
		self.last_sequence = 0
		self.anomaly_threshold = 0.1
		self.mispricing_threshold = 0.01
		if json_path:
			self.load_orderbook(json_path)
   
		# Collect data for history
		if history:
			self.history = True
			self.trade_history = []
			self.spread_history = []
			self.liquidity_history = []
			self.order_flow_history = []
		else:
			self.history = False


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


	def update(self, data: dict) -> None:
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

		if self.history and event.is_trade:
			self.trade_history.append((event.timestamp, event.price, event.size))
			self.update_spread()
			self.update_liquidity()
			self.update_order_flow(event)

		self.detect_anomalies(event)

		self.last_updated = event.timestamp
		self.last_sequence = event.seq


	def process_level2(self, event: Event) -> None:
		"""
  		Update or remove a price level based on the event.
		"""
		book = self.bids if event.is_buy else self.asks
		price_ticks = event.price_ticks(self.tick_size)

		if event.size == 0:
			book.pop(price_ticks, None)
		else:
			book[price_ticks] = Level.from_event(event)


	def process_trade(self, event: Event) -> None:
		"""
  		Process a trade event by adjusting sizes or removing levels.
		"""
		book = self.bids if not event.is_buy else self.asks
		price_ticks = event.price_ticks(self.tick_size)

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


	def update_spread(self):
		"""
		Update the spread history with the current spread.
  		"""
		best_bid = self.get_best_bid()
		best_ask = self.get_best_ask()
		if best_bid and best_ask:
			spread = best_ask.price - best_bid.price
			self.spread_history.append((self.last_updated, spread))


	def update_liquidity(self, depth_percentage: float = 0.01):
		"""
		Update the liquidity history with the current liquidity at a given depth percentage.
		"""
		best_bid = self.get_best_bid()
		best_ask = self.get_best_ask()
		if best_bid and best_ask:
			bid_liquidity = self.calculate_liquidity(self.bids, best_bid.price * (1 - depth_percentage))
			ask_liquidity = self.calculate_liquidity(self.asks, best_ask.price * (1 + depth_percentage))
			self.liquidity_history.append((self.last_updated, bid_liquidity, ask_liquidity))


	def update_order_flow(self, event: Event) -> None:
		"""
		Update the order flow history with the current event.
  		"""
		if event.is_trade:
			direction = 'Buy' if event.is_buy else 'Sell'
			self.order_flow_history.append((event.timestamp, direction, event.size))
			

	def calculate_liquidity(self, book: dict, price_threshold: float) -> float:
		""""
		Calculate the total liquidity at or above a given price threshold.
  		"""
		liquidity = 0.0
		for price, level in book.items():
			if price >= price_threshold:
				liquidity += level.size
		return liquidity


	def detect_anomalies(self, event: Event):
		"""
		Detect anomalies in the order book based on a given event.
  		"""
		if event.is_trade:
			best_bid = self.get_best_bid()
			best_ask = self.get_best_ask()
			if best_bid and best_ask:
				mid_price = (best_bid.price + best_ask.price) / 2
				if abs(event.price - mid_price) / mid_price > self.anomaly_threshold:
					print(f"Anomaly detected: Trade price deviation at {event.timestamp}")
		else:
			# Detect anomalies in order book updates
			pass


	def detect_mispricing(self, external_price: float):
		"""
		Detect mispricing based on an external price.
  		"""
		best_bid = self.get_best_bid()
		best_ask = self.get_best_ask()
		if best_bid and best_ask:
			mid_price = (best_bid.price + best_ask.price) / 2
			if abs(external_price - mid_price) / mid_price > self.mispricing_threshold:
				print(f"Mispricing detected: External price {external_price} deviates from order book mid-price {mid_price}")


	def snapshot(self) -> dict:
		"""
  		Create a snapshot of the current state of the order book.
		"""
		return {
			"bids": [(level.price, level.size) for level in self.bids.values()],
			"asks": [(level.price, level.size) for level in self.asks.values()]
		}


	def store_snapshot(self, interval: int = 60):
		"""
		Store a snapshot of the order book at regular intervals.
  		"""
		if self.last_updated % interval == 0:
			snapshot = {
				'timestamp': self.last_updated,
				'bids': [(level.price, level.size) for level in self.bids.values()],
				'asks': [(level.price, level.size) for level in self.asks.values()]
			}
			self.historical_snapshots.append(snapshot)


	def analyze_historical_data(self):
		"""
  		Perform analysis on historical snapshots
		
  		Calculate statistical measures, such as volatility, volume profiles, order imbalances, etc.
		"""
		pass


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


	def calculate_wvap(self, start_time: int, end_time: int) -> Optional[float]:
		"""
		Calculate the weighted volume average price (WVAP) over a given time interval.
  		"""
		total_volume = 0.0
		total_weighted_price = 0.0

		for timestamp, price, size in self.trade_history:
			if start_time <= timestamp <= end_time:
				total_volume += size
				total_weighted_price += price * size

		if total_volume == 0:
			return None

		wvap = total_weighted_price / total_volume
		return wvap


	def simulate_market_impact(self, trade_size: float, is_buy: bool) -> Tuple[float, float]:
		"""
		Simulate market impact for a given trade size and direction (buy/sell).

		Args:
			trade_size (float): The size of the trade.
			is_buy (bool): Whether the trade is a buy (True) or sell (False).
		
		Returns:
			Tuple[float, float]: The total filled size and price impact of the trade.
  		"""
		book = self.bids if is_buy else self.asks
		total_filled = 0.0
		total_cost = 0.0
		remaining_size = trade_size

		for price, level in book.items():
			if remaining_size <= 0:
				break
			fill_size = min(remaining_size, level.size)
			total_filled += fill_size
			total_cost += fill_size * price
			remaining_size -= fill_size

		if total_filled == 0:
			return 0.0, 0.0

		avg_price = total_cost / total_filled
		price_impact = abs(avg_price - self.get_best_bid().price if is_buy else self.get_best_ask().price)
		return total_filled, price_impact


	def test_level_display():
		"""
		Test the display of a price level.
  		"""
		level = Level(price=1.1, size=2.1)
		assert str(level) == "(1.1 : 2.1)"


	def test_event_to_level():
		"""
		Test the conversion of an event to a price level.
		"""
		event = Event(timestamp=0, seq=0, is_trade=False, is_buy=False, price=10.0, size=1.0)
		level = Level.from_event(event)
		assert level.price == 10.0
		assert level.size == 1.0


# OrderBookAnalysis class --------------------------------------------------------
class OrderBookAnalysis:
	def __init__(self, api_key, secret_key, symbol='BTC/USDT', n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=None):
		self.exchange = ccxt.binance({
			'apiKey': api_key,
			'secret': secret_key,
			'enableRateLimit': True,
		})
		self.symbol = symbol
		self.n_clusters = n_clusters
		self.kmeans_params = {
			'init': init,
			'max_iter': max_iter,
			'n_init': n_init,
			'random_state': random_state,
		}

	def fetch_order_book(self):
		orderbook = self.exchange.fetch_order_book(self.symbol)
		self.asks = pd.DataFrame(orderbook['asks'], columns=['price', 'quantity'])
		self.bids = pd.DataFrame(orderbook['bids'], columns=['price', 'quantity'])

	def analyze_order_book(self):
		self.depth_analysis()
		self.imbalance_analysis()
		self.clustering_analysis()
		self.slope_analysis()

	def depth_analysis(self):
		self.depth_ask = self.asks['quantity'].sum()
		self.depth_bid = self.bids['quantity'].sum()

	def imbalance_analysis(self):
		self.order_book_imbalance = (self.depth_ask - self.depth_bid) / (self.depth_ask + self.depth_bid)

	def clustering_analysis(self):
		X = np.column_stack((self.asks['price'], self.bids['price']))
		kmeans = KMeans(n_clusters=self.n_clusters, **self.kmeans_params).fit(X)
		self.order_book_clusters = kmeans.labels_

	def slope_analysis(self):
		self.ask_slope, _, _, _, _ = linregress(self.asks['quantity'].cumsum(), self.asks['price'])
		self.bid_slope, _, _, _, _ = linregress(self.bids['quantity'].cumsum(), self.bids['price'])

	def fair_value_strategy(self, threshold=0.01):
		if abs(self.ask_slope - self.bid_slope) > threshold:
			if self.ask_slope > self.bid_slope:
				# Place a sell order (fair value is lower than the current price)
				print("Sell signal detected (Fair Value Strategy)")
			else:
				# Place a buy order (fair value is higher than the current price)
				print("Buy signal detected (Fair Value Strategy)")

	def imbalance_clustering_strategy(self):
		if self.order_book_imbalance > 0.2 and np.mean(self.order_book_clusters == 0) > 0.6:
			# Place a sell order (imbalance towards asks, clustering suggests selling pressure)
			print("Sell signal detected (Imbalance + Clustering Strategy)")
		elif self.order_book_imbalance < -0.2 and np.mean(self.order_book_clusters == self.n_clusters - 1) > 0.6:
			# Place a buy order (imbalance towards bids, clustering suggests buying pressure)
			print("Buy signal detected (Imbalance + Clustering Strategy)")

	def visualize_order_book(self, plot_type='depth'):
		fig, ax = plt.subplots(figsize=(10, 6))

		if plot_type == 'depth':
			self.plot_depth(ax)
		elif plot_type == 'profile':
			self.plot_profile(ax)
		elif plot_type == 'slope':
			self.plot_slope(ax)

		ax.set_title(f'Order Book Visualization ({self.symbol})', fontsize=16)
		ax.grid(True)
		plt.show()

	def plot_depth(self, ax):
		ax.bar(self.asks['price'], self.asks['quantity'], color='r', alpha=0.5, label='Asks')
		ax.bar(self.bids['price'], -self.bids['quantity'], color='g', alpha=0.5, label='Bids')
		ax.axhline(0, color='k', linestyle='--')
		ax.set_xlabel('Price')
		ax.set_ylabel('Quantity')
		ax.legend()

	def plot_profile(self, ax):
		ask_profile = self.asks.groupby('price')['quantity'].sum().sort_index(ascending=False).cumsum()
		bid_profile = self.bids.groupby('price')['quantity'].sum().sort_index(ascending=True).cumsum()

		ax.plot(ask_profile.index, ask_profile.values, color='r', label='Asks')
		ax.plot(bid_profile.index, bid_profile.values, color='g', label='Bids')
		ax.set_xlabel('Price')
		ax.set_ylabel('Cumulative Quantity')
		ax.legend()

	def plot_slope(self, ax):
		ask_prices = np.array(self.asks['price'])
		bid_prices = np.array(self.bids['price'])
		ask_slope = self.ask_slope
		bid_slope = self.bid_slope

		ax.scatter(self.asks['quantity'].cumsum(), ask_prices, color='r', label='Asks', alpha=0.5)
		ax.plot(self.asks['quantity'].cumsum(), ask_slope * self.asks['quantity'].cumsum() + ask_prices.mean(), color='r')

		ax.scatter(self.bids['quantity'].cumsum(), bid_prices, color='g', label='Bids', alpha=0.5)
		ax.plot(self.bids['quantity'].cumsum(), bid_slope * self.bids['quantity'].cumsum() + bid_prices.mean(), color='g')

		ax.set_xlabel('Cumulative Quantity')
		ax.set_ylabel('Price')
		ax.legend()
		
	def train_pattern_detector(self, data, epochs=10, batch_size=32):
		# Preprocess data
		X = np.array([self.preprocess_order_book(ob) for ob in data])
		y = np.array([self.label_order_book(ob) for ob in data])

		# Define the neural network architecture
		model = keras.Sequential([
			keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
			keras.layers.Dense(32, activation='relu'),
			keras.layers.Dense(1, activation='sigmoid')
		])

		# Compile the model
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

		# Train the model
		self.pattern_detector = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

	def preprocess_order_book(self, order_book):
		# Convert order book data to a suitable input format for the neural network
		# (e.g., flattening the ask and bid data into a single vector)
		asks = order_book['asks']
		bids = order_book['bids']
		return np.concatenate([np.array(asks).flatten(), np.array(bids).flatten()])

	def label_order_book(self, order_book):
		# Assign a label to the order book based on your desired pattern
		# (e.g., 1 for a bullish pattern, 0 for a bearish pattern)
		# This is just a simple example, you would need to define your own labeling logic
		depth_ask = sum(order_book['asks'][:, 1])
		depth_bid = sum(order_book['bids'][:, 1])
		if depth_bid > depth_ask:
			return 1  # Bullish pattern
		else:
			return 0  # Bearish pattern

	def detect_pattern(self, order_book):
		X = np.array([self.preprocess_order_book(order_book)])
		prediction = self.pattern_detector.predict(X)[0][0]
		if prediction > 0.5:
			print("Bullish pattern detected")
		else:
			print("Bearish pattern detected")


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


	def update_plot_with_depth(self, i: int) -> None:
		"""
		Update the plot with depth information.
  		"""
		self.ax.clear()
		self.ax.set_ylim(0, 5)

		if self.order_book.bids and self.order_book.asks:
			bid_prices, bid_sizes = zip(*sorted(self.order_book.bids.items(), reverse=True))
			ask_prices, ask_sizes = zip(*sorted(self.order_book.asks.items()))

			bid_color = sns.color_palette("deep")[2]
			ask_color = sns.color_palette("deep")[3]

			bid_cumulative_sizes = np.cumsum(bid_sizes)
			ask_cumulative_sizes = np.cumsum(ask_sizes)

			self.ax.step(bid_prices, bid_cumulative_sizes, where='post', color=bid_color, label='Bids')
			self.ax.step(ask_prices, ask_cumulative_sizes, where='post', color=ask_color, label='Asks')

			#TBC


# WebSocketHandler class --------------------------------------------------------
class WebSocketHandler:
	"""
	A class to handle WebSocket connections for real-time order book updates.
	"""
	def __init__(	self, 
			  		url: str, 
					order_book: 'OrderBook', 
				 	symbol: str = 'BTCUSDT', 
				  	depth: int = 20, 
				   	store_data: bool = False, 
					max_stored_data: int = 1000, 
					output_file: str = 'stored_data.json'):
		self.ws = websocket.WebSocketApp(url, on_open=self.on_open, on_message=self.on_message)
		self.url = url
		self.order_book = order_book
		self.symbol = symbol.lower()
		self.depth = depth
		self.store_data = store_data
		self.event_count = 0
		self.max_events = max_stored_data
		self.max_stored_data = max_stored_data
		self.stored_data = []
		self.output_file = output_file


	def on_open(self, ws: websocket.WebSocketApp) -> None:
		"""
		Open the WebSocket connection and subscribe to the order book updates.
  
		Args:
			ws (websocket.WebSocketApp): The WebSocket connection.
  		"""
		print("WebSocket connection opened")
		ws.send(json.dumps({
			'method': 'SUBSCRIBE',
			'params': [f'{self.symbol}@depth{self.depth}'],
			'id': 1
		}))
		print("Subscription message sent")


	def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
		"""
		Handle incoming WebSocket messages.
  
		Args:
			ws (websocket.WebSocketApp): The WebSocket connection.
  		"""
		print("Received message:", message)
		data = json.loads(message)
		if 'bids' in data and 'asks' in data:
			print("Received data:", data)
			self.order_book.update(data)
			if self.store_data:
				self.store_received_data(data)
				self.write_data_to_file(data)  # Write data to file
				self.event_count += 1
		else:
			print("Data does not contain 'bids' and 'asks'")


	def store_received_data(self, data: dict) -> None:
		"""
		Store the received data in memory.
	
		Args:
			data (dict): The data to store.
  		"""
		if len(self.stored_data) >= self.max_stored_data:
			self.stored_data.pop(0)
		self.stored_data.append(data)


	def start(self):
		"""
		Start the WebSocket in a separate thread.
		"""
		print("Starting WebSocket connection...")
		self.ws = websocket.WebSocketApp(self.url,
										 on_open=self.on_open,
										 on_message=self.on_message,
										 on_error=self.on_error,
										 on_close=self.on_close)
		threading.Thread(target=self.ws.run_forever).start()


	def write_data_to_file(self, data: dict) -> None:
		"""
		Write the data to a file.
  		"""
		with open(self.output_file, 'a') as file:
			json.dump(data, file)
			file.write('\n')


	def on_error(self, ws, error):
		print("WebSocket error:", error)

	def on_close(self, ws):
		print("WebSocket connection closed")
		

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
	STORED_DATA = 9
	ANALYZE_REGRESSION = 10
	ANALYZE_CLUSTERING = 11
	
def analyze_stored_data(file_path: str) -> None:
	"""
	Analyze stored order book data from a JSON file.
 	"""
	order_book = OrderBook(0.01)

	with open(file_path, 'r') as file:
		for line in file:
			data = json.loads(line)
			order_book.update(data)

	# Perform analysis on the loaded order book data
	print("Midprice:", order_book.midprice())
	print("Weighted Midprice:", order_book.weighted_midprice())
	print("Best Bid:", order_book.get_best_bid())
	print("Best Ask:", order_book.get_best_ask())
	
 
def analyze_stored_data_regression(file_path: str) -> None:
	"""
	Analyze stored order book data from a JSON file using regression.
 
	Args:
		file_path (str): The path to the stored order book data file.
  
	Returns:
		None
	"""
	order_book = OrderBook(0.01)
	order_book_analysis = OrderBookAnalysis(order_book, n_clusters=5, init='random', max_iter=500, n_init=20, random_state=42)

	with open(file_path, 'r') as file:
		for line in file:
			data = json.loads(line)
			order_book.update(data)
			order_book_analysis.analyze()

			# Apply trading strategies
			order_book_analysis.fair_value_strategy(threshold=0.005)
			order_book_analysis.imbalance_clustering_strategy()

	print("Analysis completed for stored data.")


def analyze_real_time_data(symbol: str, n_clusters: int, init: str, max_iter: int, n_init: int, random_state: int) -> None:
	"""
	Analyze real-time order book data from a WebSocket connection.

	Args:
		symbol (str): The trading symbol to analyze.
		n_clusters (int): The number of clusters for K-means clustering.
		init (str): The initialization method for K-means clustering.
		max_iter (int): The maximum number of iterations for K-means clustering.
		n_init (int): The number of initializations for K-means clustering.
		random_state (int): The random seed for K-means clustering.

	Returns:
		None
	"""
	order_book = OrderBook(0.01)
	order_book_analysis = OrderBookAnalysis(order_book, n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
	ws_handler = WebSocketHandler(url="wss://stream.binance.com:9443/ws", order_book=order_book, symbol=symbol)
	ws_handler.start()

	while True:
		order_book_analysis.analyze()

		# Apply trading strategies
		order_book_analysis.fair_value_strategy(threshold=0.005)
		order_book_analysis.imbalance_clustering_strategy()

		time.sleep(1)  # Adjust the delay as needed
 
 
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

	elif unit_test == UnitTests.STORED_DATA:
	 
		to_read = False	
		data_file = 'stored_data.json'
		file_number = 2
		
		if to_read:			# Read stored data from a file
			analyze_stored_data(data_file)
		else:				# Store real-time data to a file
			print("No data to read.")
			print("Starting real-time order book store...")

			order_book = OrderBook(0.01)

			symbol = 'BTCUSDT'
   
			current_datetime = datetime.datetime.now().strftime("%Y_%m_%d")
			
			output_file = f"stored_data_{current_datetime}_{file_number}.json"
			while os.path.exists(output_file):
				file_number += 1
				output_file = f"stored_data_{current_datetime}_{file_number}.json"
			ws_handler = WebSocketHandler(url="wss://stream.binance.com:9443/ws", order_book=order_book, symbol=symbol, store_data=True, max_stored_data=50, output_file=output_file)
			ws_handler.start()

			while ws_handler.event_count < ws_handler.max_events:
				time.sleep(1)

			print("Data retrieval completed. Stored in file:", output_file)

	elif unit_test == UnitTests.ANALYZE_REGRESSION:
			stored_data_file = 'stored_data.json'
			analyze_stored_data(stored_data_file)

			# Example usage for real-time data analysis
			symbol = 'BTCUSDT'
			n_clusters = 5
			init = 'random'
			max_iter = 500
			n_init = 20
			random_state = 42
			analyze_real_time_data(symbol, n_clusters, init, max_iter, n_init, random_state)
		
	elif unit_test == UnitTests.ANALYZE_CLUSTERING:
			stored_data_file = 'stored_data.json'
			analyze_stored_data(stored_data_file)

			# Example usage for real-time data analysis
			symbol = 'BTCUSDT'
			n_clusters = 5
			init = 'random'
			max_iter = 500
			n_init = 20
			random_state = 42
			analyze_real_time_data(symbol, n_clusters, init, max_iter, n_init, random_state)
  
	else:
		print("Invalid unit test specified.")


if __name__ == '__main__':
	
	unit_test = UnitTests.STORED_DATA
	is_run_all_tests = False
 
	if is_run_all_tests:
		for unit_test in UnitTests:
			run_unit_test(unit_test=unit_test)
	else:
		run_unit_test(unit_test=unit_test)