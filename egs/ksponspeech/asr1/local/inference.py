import sys

test=sys.argv[1]

with open('data/'+str(test)+'/tpath.scp', 'r', encoding="utf8") as f:
    with open('data/'+str(test)+'/text.txt', 'a', encoding="utf8") as h:
        tpaths = f.readlines()
        for tpath in tpaths:
            tpath=tpath.strip('\n')
            with open(tpath, 'r', encoding="utf8") as g:
                text = str(g.readlines()[0])
                h.write(text.strip('\n') + '\n')
