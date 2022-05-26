import sys
filename = sys.argv[1]

for line in open(filename):
    if "||" in line:
        print(line.split("||")[0])
        print()
