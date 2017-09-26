from gzip import open
from sys import argv
from json import loads, dumps

if __name__ == '__main__':

	datafile = argv[1]
	target = argv[2]
	filterfile = argv[3]

	cread = 0
	cfilter = 0
	cnofield = 0

	fout = open(filterfile, 'w')

	fin = open(datafile, 'r')
	for line in fin:
		jdata = loads(line.strip())
		cread = cread + 1

		if 'city' not in jdata:
			cnofield = cnofield + 1
			continue

		city = jdata['city']
		if city.lower() == target.lower():
			cfilter = cfilter + 1
			sdata = dumps(jdata)
			fout.write('%s\n' % sdata)

		if cread % 10000 == 0:
			print '%s instances read, %s instances filtered' % (cread, cfilter)

	fin.close()
	fout.close()

	print '%s instances read, %s instances filtered, %s instances ignored' % (cread, cfilter, cnofield)
