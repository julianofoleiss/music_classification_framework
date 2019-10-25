from sklearn.model_selection import StratifiedKFold
import os
import fnmatch
import numpy as np
from pprint import pformat

def create_metafile(root_data_dir, data_extension, output_filename, full_path=True):
    """
    This function creates a metafile by scanning root_data_dir and subdirectories for
    files with data_extension. 

    It assumes that the label is the first substring of each filename up to '_'. Writes
    the metadata to output_filename. Includes full path to files, except when full_path==False.

    :param root_data_dir:
    :param data_extension:
    :param output_filename:
    :param full_path:

    :type root_data_dir: str
    :type data_extension: str
    :type output_filename: str
    :type full_path: bool
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_data_dir):
        for filename in fnmatch.filter(filenames, '*.%s' % data_extension):
            matches.append(os.path.join(root, filename))

    matches = sorted(matches)

    names = [os.path.split(i) for i in matches]
    if not full_path:
        names = [ ('.', i[1]) for i in names ]

    labels = [ i[1].split('_')[0] for i in names]

    out = [ "\t".join( [names[i][0] + '/' + names[i][1], labels[i] ] ) + "\n" for i in xrange(len(labels))  ]

    output = open(output_filename, 'w')
    output.writelines(out)
    output.close()

#DELETE THIS LATER
def parse_filelist(filename):
    with open(filename, 'r') as f:
        c = f.readlines()

    c = [ s.strip().split('\t') for s in c]
    files = [ s[0] for s in c  ]

    if len(c[0]) > 1:
        tags = [ [i for i in s[1].split(',') ][0] for s in c ]
    else:
        tags = [None for s in c]

    return files, tags


def _output_metafile(X, Y, output_filename=None, test_file=False):

    if test_file:
        lines = [ X[i] + "\n" for i in xrange(len(X)) ]
    else:
        lines = [ '\t'.join((X[i], Y[i])) + '\n' for i in xrange(len(X)) ]

    if output_filename is not None:
        with open(output_filename, 'w') as f:
            f.writelines(lines)

    return lines

def create_kfold_files(meta_filename, k, output_directory, shuffle=False, rns=None):
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=rns)

    if output_directory[-1] != '/':
        output_directory = output_directory + '/'

    files, tags = parse_filelist(meta_filename)

    files = np.array(files)
    tags = np.array(tags)

    fn = 1
    for train_idx, test_idx in skf.split(files, tags):
        
        _output_metafile(files[train_idx], tags[train_idx], output_directory + 'f%d_train.txt' % fn)
        _output_metafile(files[test_idx], tags[test_idx], output_directory + 'f%d_test.txt' % fn, test_file=True)
        _output_metafile(files[test_idx], tags[test_idx], output_directory + 'f%d_evaluate.txt' % fn)

        fn+=1

def rename_file_prefixes(meta_filename, new_prefix, old_prefix=None):
    pass

def get_label_dict(meta_filename, out_filename=None):

    _, tags = parse_filelist(meta_filename)
    tags = sorted(list(set(tags)))

    label_dict = dict()
    for i, t in enumerate(tags):
        label_dict[t] = i

    if out_filename is not None:
        f = open(out_filename, 'w')
        f.write('label_dict = %s\n' % pformat(label_dict, ) )
        f.close()

    return label_dict

if __name__ == '__main__':

    #create_metafile('/home/juliano/Doutorado/datasets/gtzan44', 'wav', '/home/juliano/Doutorado/music_classification_framework/gtzan_labels.txt')

    #create_kfold_files('/home/juliano/Doutorado/music_classification_framework/gtzan_labels.txt', 10, '/home/juliano/Doutorado/music_classification_framework/gtzan_folds/', shuffle=False)

    get_label_dict('/home/juliano/Doutorado/music_classification_framework/gtzan_labels.txt', '/home/juliano/Doutorado/music_classification_framework/gtzan_dict.py')

