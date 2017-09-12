from gzip import open
from sys import argv
from json import loads, dumps

if __name__ == '__main__':

	datafile = argv[1]
	filterfile = argv[2]

	cread = 0
	cnofield = 0

	fout = open(filterfile, 'w')

	fin = open(datafile, 'r')
	for line in fin:
		jdata = loads(line.strip())
		cread = cread + 1

		if 'pubMillis' not in jdata or 'line' not in jdata:
			cnofield = cnofield + 1
			continue

		coords = jdata['line']
		time = jdata['pubMillis']
		jdata_new = {'line': coords, 'pubMillis': time}
		sdata = dumps(jdata_new)
		fout.write('%s\n' % sdata)

		if cread % 10000 == 0:
			print '%s instances read' % (cread)

	fin.close()
	fout.close()

	print '%s instances read, %s instances ignored' % (cread, cnofield)
