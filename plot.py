import matplotlib.pyplot as plt
import csv

def plotter(filename):
    with open(filename, 'r') as csvfile:
        titles = []
        data = []
        reader = csv.reader(csvfile, delimiter=',')

        sortedlist = sorted(reader, key=lambda row: row[0])

        for line in sortedlist:
            eval_num = int(line[0])
            L = [float(n) for n in line[1:] if n]
            ave = sum(L)/float(len(L)) if L else '-'
            
            titles.append(eval_num)
            data.append(ave)

        return titles, data

titles, data = plotter("ga.csv")

print(titles)

fig, ax = plt.subplots()
ax.plot(titles, data)
ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()