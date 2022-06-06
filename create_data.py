import random
for count in range(1, 11):
    city_num = random.randint(10, 100)
    distance_x = [random.randint(50, 800) for i in range(city_num)]
    distance_y = [random.randint(50, 800) for i in range(city_num)]
    with open('data/data' + str(count) + '.txt', 'wt') as f:
        f.write(str(city_num) + '\n')
        for i in range(city_num):
            f.write(str(i) + ' ' + str(distance_x[i]) + ' ' + str(distance_y[i]) + '\n')

