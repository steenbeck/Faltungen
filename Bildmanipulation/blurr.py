import filter_image


# Erstelle Blurr-Filter
n = 10
blurr_filter = [[1/(n**2) for j in range(1, n+1)] for i in range(1, n+1)] # Bspw. [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]


# Zeige gefiltertes Bild an
filter_image.filter_and_show_image([blurr_filter])


