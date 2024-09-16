import csv

data = []
i = 0

def writeToFile():
    with open('../data/data'+str(i)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FEN', 'Evaluation'])
        writer.writerows(data)

with open('../data/chessData.csv', mode ='r') as file:
    counter = 0
    csvFile = csv.reader(file)
    next(csvFile)
    for row in csvFile:
        if i == 10: break
        if counter == 100000:
            writeToFile()
            i += 1
            counter = 0
            data = []
        data.append(row)
        counter += 1