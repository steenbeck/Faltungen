import random
import math
import json


file_name = "ts_example.txt"


def generate_time_series_example(n):
	r = random.random()
	ts = [50 + 20*(math.sin(i/(n/5)*math.pi + random.random())*math.sin(random.random()) + math.sin(random.random())) for i in range(0, n)]
	#ts = [50 + 20*random.uniform(0.8, 1.0)*math.sin((i/(n/20))*random.uniform(0.8, 1.0)*math.pi + random.random()) for i in range(0, n)]
	return ts

def save_time_series_example(time_series):
	with open(file_name, "w") as file:
		json.dump(time_series, file)

def read_time_series_example():
	with open(file_name, "r") as file:
		return list(json.load(file))




if __name__ == '__main__':
	save_time_series_example(generate_time_series_example(200))


