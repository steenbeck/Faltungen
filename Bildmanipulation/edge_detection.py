import filter_image


# Erstelle Edge-Detection-Filter
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]



# Zeige gefiltertes Bild an
filter_image.filter_and_show_image([sobel_y])


