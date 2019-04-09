import numpy
import matplotlib.pyplot as pyplot

import genrand_ts


### Berechne Moving Average n-ter Ordnung
def moving_average(vec, n):
	smooth_filter = [(1/n) for i in range(0, n)]
	return list(numpy.convolve(vec, smooth_filter))
###


if __name__ == '__main__':
	# Periode des Moving Average
	n = 4

	# Lese gespeichertes Beispiel ein
	time_series = genrand_ts.read_time_series_example()


	# Koordinaten der Zeitserie
	Y_ts = time_series
	X_ts = [i for i in range(1, len(Y_ts) + 1)]

	# Koordinaten der "geglätten" Zeitserie
	Y_ma = moving_average(time_series, n)[(n-1):len(Y_ts)] # Schneide zurecht, sodass Einträge i = n,...,N
	X_ma = [i for i in range(n, len(Y_ts) + 1)] 



	# Stelle Plotter ein
	pyplot.xlabel('t')
	pyplot.ylabel('')


	# plotte Zeitserie und geglättete Zeitserie
	pyplot.plot(X_ts, Y_ts, color = "c", marker = "o", markersize = 5)
	pyplot.plot(X_ma, Y_ma, color = "r", marker = "o", markersize = 5)

	# Zeige Plot
	pyplot.show()