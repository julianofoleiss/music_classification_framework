# -*- coding: utf-8 -*-

from __future__ import division
import dill
import tables
import numpy as np
import gc
from multiprocess import Pool
from data.filelists import parse_filelist
from data.predicted_features import PredictedFeatures

from joblib import Parallel, delayed

def from_filelist(inputfile, batchsize, worker_load, procs, feature_extraction_func, fef_params, output, out_format='np', inscript=True):
    """
    Extracts features from all files in a filelist using thread (process) pool.

    This function may be used to extract features from multiple files using a pool of threads
    You may pass a feature extraction function and arguments for custom feature extraction
    procedures.

    :param inputfile: a list to a filelist that contains two tab-separated columns: 
    one with full path to files and a comma separated list of tags.
    :param batchsize: number of files to extract features from before dumping them to disk.
    :param worker_load: number of files to pass onto each worker at a time.
    :param procs: number of parallel processes that should be used to compute features.
    :param feature_extraction_func: the function that gets called for every worker_load files in the 
    filelist. This function should return a PredictedFeatures object. When called,
    it always receives a tuple with three parameters: (fef_args, input_files, tag_lists).
    :param params: parameter dictionary from yaml configuration file passed onto feature_extraction_func.
    :param output: path to an output dill (PredictedFeatures) file. if out_format=='h5', a h5 datafile will also 
    be created with the same prefix.
    :returns: None

    :type inputfile: str
    :type batchsize: int
    :type worker_load: int
    :type procs: int
    :type feature_extraction_func: callable
    :type fef_params: dict
    :type output: str
    :type out_format: if 'np', a numpy array is used to save features. If 'h5' a h5 archive is used. 'np' may be
    not usable on large datasets.
    :rtype: None
    """

    files, tags = parse_filelist(inputfile)

    #pool = Pool(procs)

    #TODO: sanitize outfile
    outdata = None
    filenames = []
    nframes = []
    labels = []
    h5 = None
    
    if not inscript:
        from progress.bar import Bar
        #prog_bar = Bar('Processing files...', max=(len(files) // batchsize)+1)
        prog_bar = Bar('Processing files...', max=(len(files)), suffix='%(percent)d%%')

    for k in range(0, len(files), batchsize):

        kf = files[k:k+batchsize]
        kt = tags[k:k+batchsize]
        
        kfi = kti = len(kf) // worker_load
        kfm = ktm = len(kf) %  worker_load
        
        wf = [ [kf[(worker_load*i)+j] for j in range(worker_load)] for i in range(kfi)]
        wt = [ [kt[(worker_load*i)+j] for j in range(worker_load)] for i in range(kti)]

        if kfm > 0:
            wfm = kf[kfi*worker_load:]
            wtm = kt[kti*worker_load:]

            wf.append(wfm)
            wt.append(wtm)

        fef_args = [fef_params] * len(wf)

        args = zip(fef_args, wf, wt)

        #res = pool.map(feature_extraction_func, args)

        try:
            res = Parallel(n_jobs=procs)(delayed(feature_extraction_func)(arg) for arg in args)
        except:
            if h5 is not None:
                print('exception during extraction procedure: closing output h5 storage file.')
                h5.close()
            raise

        res =  [track for bundle in res for track in bundle]

        for i in res:
            filenames.extend(i.track_filenames)
            nframes.extend(i.track_nframes)
            labels.append(i.labels)

        #res_data = np.array([ i.estimator_output for i in res ]).reshape((-1, res[0].estimator_output.shape[1]))

        res_data = np.concatenate([ i.estimator_output for i in res ])
        res_data = res_data.reshape((-1, res[0].estimator_output.shape[1]))
        res_len = len(res)
        label_idx_dict = res[0].label_idx_dict

        res = None

        if outdata is None:
            if out_format == 'h5':
                a = tables.Float32Atom()
                h5 = tables.open_file(output + '.h5', mode='w')
                f = tables.Filters(complevel=1, complib='lzo')
                outdata = h5.create_earray(h5.root, 'features', a, (0, res_data.shape[1]), filters=f )
                outdata.attrs.h5filename = h5.filename
            else:
                if out_format == 'np':
                    outdata = []

        outdata.append(res_data)

        res_data_shape1 = res_data.shape[1]

        res_data = None

        gc.collect()

        if inscript:
            print('processed %d out of %d files' % (k+res_len, len(files)))
        else:
            prog_bar.next(n=res_len)

    if out_format == 'np':
        outdata = np.concatenate(outdata).reshape((-1, res_data_shape1))

    pf = PredictedFeatures( filenames, nframes, outdata, label_idx_dict, [None] * len(filenames) , np.array(labels).reshape((-1, labels[0].shape[1])) )

    dill.dump(pf, open(output, 'w'))

    if out_format =='h5':
        h5.flush()
        h5.close()

    if not inscript:
        prog_bar.finish()

    return pf

if __name__ == '__main__':
    import yaml
    from exp_gtzan_selframes import gtzan_selected

    params = yaml.load(open('parameters.yaml').read())

    from_filelist('gtzan_labels_small.txt', 10, 4, 2, gtzan_selected, params, 'teste.dill')

    