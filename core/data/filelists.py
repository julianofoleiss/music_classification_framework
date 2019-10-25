

def parse_filelist(filename):
    with open(filename, 'r') as f:
        c = f.readlines()

    c = [ s.strip().split('\t') for s in c]
    files = [ s[0] for s in c  ]

    if len(c[0]) > 1:
        tags = [ [i for i in s[1].split(',') ] for s in c ]
    else:
        tags = [None for s in c]

    return files, tags