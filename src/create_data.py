import csv
import random

data = []
i = 0

fid = open("../data/chessData.csv", "r")
li = fid.readlines()[1:]
fid.close()
random.shuffle(li)
fid = open("../data/shuffledAll.csv", "w")
fid.writelines(li)
fid.close()

print('shuffled')

def writeToFile():
    with open('../data/data'+str(i)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FEN', 'Evaluation'])
        random.shuffle(data)
        writer.writerows(data)

with open('../data/shuffledAll.csv', mode ='r') as file:
    counter = 0
    csvFile = csv.reader(file)
    for row in csvFile:
        if i == 10: break
        if counter == 500000:
            writeToFile()
            i += 1
            counter = 0
            data = []
        data.append(row)
        counter += 1