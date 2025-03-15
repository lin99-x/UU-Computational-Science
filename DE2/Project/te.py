y_list = [15.1712, 15.2771, 15.8901, 16.9501, 16.8580, 17.5116]
i = 1
speedup_list = []
for y in y_list:
    speedup = y_list[0] * i / y
    speedup_list.append(speedup)
    i = i * 2

print(speedup_list)