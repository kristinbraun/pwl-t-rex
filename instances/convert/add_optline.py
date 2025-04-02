filename = input()

f = open(filename, "r")
g = open(filename[:-4] + '_add.gms', "w")

fr = f.read()
fr = fr.split('\n')
for r in fr:
	if 'solve' in r or 'Solve' in r:
		g.write('m.optfile=1;\n')
	g.write(r + '\n')

f.close()
g.close()
