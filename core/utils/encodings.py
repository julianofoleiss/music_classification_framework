# -*- coding: utf-8 -*-
import numpy as np

def to_label_matrix(labels, label_dict):
    label_values = [ label_dict[l] for l in labels]
    Y_M = np.zeros((len(labels), len(set(label_dict.keys()))))
    k = np.arange(len(labels))
    Y_M[k, label_values] = 1
    
    return Y_M.astype('float32')

def tags_to_matrix(label_lists, label_dict):
    idxs = np.array([[label_dict[k] for k in l] for l in label_lists])
    Y_M = np.zeros((len(label_lists), len(set(label_dict.keys()))))
    k = np.arange(len(label_lists))
    Y_M[k, idxs.T] = 1
    return Y_M.astype('float32')

def label_matrix_to_list(label_matrix, label_dict=None):
    l_idx = np.argmax(label_matrix, axis=1)
    if label_dict:
        ild = inv_dict(label_dict)
        return [ ild[x] for x in l_idx  ]
    return l_idx

def get_label_array(labels, nframes):
    out = []
    for i,f in enumerate(nframes):
        out.extend( [ labels[i] ] * f ) 
    return np.array(out)

def inv_dict(d):
    return { d[x] : x for x in d }
