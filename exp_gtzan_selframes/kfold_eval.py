from __future__ import print_function
from pprint import pprint
from core.utils.kfold_eval import get_avg_std_kfold_scores
import os, sys
import itertools
import glob
import numpy as np

def get_kfold_scores_ae(folder):

    hidden_size = [64, 128, 256, 512, 1024]
    n_frames = [40, 20, 5, 1]
    f_selector = ['kmeansc','linspace']

    all_params = [hidden_size, f_selector, n_frames ]

    filename_pattern = folder + '/%d-nf_%s-fs_%d-hs_*-f.eval'
    ivector_pattern = folder + '/ivector-fs_%d-hs_*-f.eval'
    
    all_avgs = []
    all_stds = []
    all_configs = []

    n = 0

    for p in itertools.product(*all_params):
        configs = [filename_pattern % (p[2], str(p[1]), p[0] ) , ivector_pattern % (p[0]) ]
        for config in configs:
            files = glob.glob(config)
            if os.path.basename(config) not in all_configs:
                if len(files) > 0:
                    avgs, stds = get_avg_std_kfold_scores(files)
                    all_avgs.append( avgs )
                    all_stds.append( stds )
                    all_configs.append(os.path.basename(config))

                n+=1

    return zip(all_configs, all_avgs, all_stds)

def get_kfold_scores_rp(folder):

    # deltas = [False, True]
    # noanova = [True, False]
    # nonlinearity = ['none', 'tanh']
    # n_feats = [30, 100, 200, 300]
    # n_frames = [40, 20, 5]
    # f_selector = ['linspace', 'kmeansc']

    deltas = [False, True]
    noanova = [True, False]
    nonlinearity = ['none', 'tanh']
    n_feats = [8, 26, 51, 75, 100]
    n_frames = [40, 20, 5]
    f_selector = ['linspace', 'kmeansc']

    #all_params = [deltas, noanova, nonlinearity, n_feats, n_frames, f_selector]
    all_params = [n_feats, deltas, noanova, nonlinearity,  n_frames, f_selector]

    filename_pattern = folder + '/%d-nf_%s-fs_%d-nfeats_%s-nl_%s-noanova_%s-delta_*-f.eval'
    ivector_pattern = folder + 'ivector-fs_%d-nfeats_%s-nl_%s-noanova_%s-delta_*-f.eval'

    all_avgs = []
    all_stds = []
    all_configs = []

    n = 0

    #get results for frame selection
    for p in itertools.product(*all_params):
        #configs = [filename_pattern % (p[4], p[5], p[3], str(p[2]), str(p[1]), p[0]), ivector_pattern % (p[3], str(p[2]), str(p[1]), p[0])]
        configs = [filename_pattern % (p[4], p[5], p[0], str(p[3]), str(p[2]), p[1]), ivector_pattern % (p[0], str(p[3]), str(p[2]), p[1])]
        for config in configs:
            files = glob.glob(config)
            #print("%s %d" % (config, len(files)))
            if os.path.basename(config) not in all_configs:
                if len(files) > 0:
                    avgs, stds = get_avg_std_kfold_scores(files)
                    all_avgs.append( avgs )
                    all_stds.append( stds )
                    all_configs.append(os.path.basename(config))

                n+=1
    
    print (n)

    return zip(all_configs, all_avgs, all_stds)

def get_times_rp(xlog):

    def current_config(delta, no_anova, nonlinearity, n_feats, n_frames, fselector):
        return "%d-nf_%s-fs_%d-nfeats_%s-nl_%s-noanova_%s-delta" % (n_frames, fselector, n_feats, nonlinearity, str(no_anova), str(delta))

    def to_sec(timestr):
        ftr = [3600,60,1]
        ftr = ftr[-timestr.count(':')-1:]
        return sum([a*b for a,b in zip(ftr, map(int,timestr.split(':')))])

    if xlog is None:
        return None

    with open(xlog) as f:
        lines = f.readlines()
    
    train_times = []
    test_times = []
    extract_times = []
    delta = no_anova = nonlinearity = n_feats = n_frames = fselector = None
    ret = dict()
    prev_config = None
    this_config = None
    task = None
    l_idx = 0

    while l_idx < len(lines):
        l = lines[l_idx]

        #print (l_idx)

        if None not in [delta, no_anova, nonlinearity, n_feats, n_frames, fselector]:
            prev_config = current_config(delta, no_anova, nonlinearity, n_feats, n_frames, fselector)
        
        if "@@@" in l:
            delta = l.split(' ')[-1].strip()

        if "???" in l:
            no_anova = l.split(' ')[-1].strip()

        if "&&&" in l:
            nonlinearity = l.split(' ')[-1].strip()

        if "%%%" in l:
            n_feats = int(l.split(' ')[-1])
        
        if "###" in l:
            n_frames = int(l.split(' ')[-1])

        if "***" in l:
            fselector = l.split(' ')[-1].strip()

        if "^^^" in l:
            fselector = 'ivector'

        if "-train" in l:
            task = "train"
        
        if "-extract" in l:
            task = "extract"
        
        if "-test" in l:
            task = "test"

        if None not in [delta, no_anova, nonlinearity, n_feats, n_frames, fselector]:
            this_config = current_config(delta, no_anova, nonlinearity, n_feats, n_frames, fselector)
        
            #print (this_config, prev_config)

            if (this_config != None) and (prev_config != None):
                if this_config != prev_config:
                    # ret[prev_config] = (np.mean(extract_times), np.std(extract_times), np.mean(train_times), np.std(train_times),
                    #     np.mean(test_times), np.std(test_times))
                    # extract_times = []
                    if len(train_times) > 0 and len(test_times) > 0:
                        ret[prev_config] = (np.mean(train_times), np.std(train_times), np.mean(test_times), np.std(test_times))
                        train_times = []
                        test_times = []

            # if task == "extract":
            #     if "processed" in l:
            #         li = l_idx
            #         while "processed" in lines[li]:
            #             li+=1
                    
            #         data = lines[li].split(' ')
            #         elapsed = data[2].split('.')[0]
            #         print(data, to_sec(elapsed))
            #         extract_times.append(to_sec(elapsed))
            #         l_idx = li

            #     l_idx+=1

            # else:
            #     l_idx += 1

            if "model training took" in l:
                train_times.append( float(l.split(' ')[3]) )

            if "Voting..." in l:
                data = lines[l_idx+1].split(' ')
                elapsed = data[2].replace('elapsed','').split(".")[0]
                #print (data[2], elapsed     )
                test_times.append(float(to_sec(elapsed)))

        l_idx += 1

    ret[this_config] = (np.mean(train_times), np.std(train_times), np.mean(test_times), np.std(test_times))

    print (len(ret))

    return ret

def get_times_ae(xlog):

    def current_config(n_frames, fselector, hidden_size):
        return "%d-nf_%s-fs_%d-hs" % (n_frames, fselector, hidden_size)

    def to_sec(timestr):
        ftr = [3600,60,1]
        ftr = ftr[-timestr.count(':')-1:]
        return sum([a*b for a,b in zip(ftr, map(int,timestr.split(':')))])

    if xlog is None:
        return None

    with open(xlog) as f:
        lines = f.readlines()
    
    train_times = []
    test_times = []
    extract_times = []
    hidden_size = n_frames = fselector = None
    ret = dict()
    prev_config = None
    this_config = None
    task = None
    l_idx = 0

    while l_idx < len(lines):
        l = lines[l_idx]

        #print (l_idx)

        if None not in [hidden_size, n_frames, fselector]:
            prev_config = current_config(n_frames, fselector, hidden_size)
        
        if "@@@" in l:
            hidden_size = int(l.split(' ')[-1].strip())

        if "###" in l:
            n_frames = int(l.split(' ')[-1].strip())

        if "***" in l:
            fselector = l.split(' ')[-1].strip()

        if "^^^" in l:
            fselector = 'ivector'

        if "-train" in l:
            task = "train"
        
        if "-extract" in l:
            task = "extract"
        
        if "-test" in l:
            task = "test"

        if None not in [hidden_size, n_frames, fselector]:
            this_config = current_config(n_frames, fselector, hidden_size)
        
            #print (this_config, prev_config)

            if (this_config != None) and (prev_config != None):
                if this_config != prev_config:
                    if len(train_times) > 0 and len(test_times) > 0:
                        ret[prev_config] = (np.mean(train_times), np.std(train_times), np.mean(test_times), np.std(test_times))
                        train_times = []
                        test_times = []

            if "model training took" in l:
                train_times.append( float(l.split(' ')[3]) )

            if "Voting..." in l:
                data = lines[l_idx+1].split(' ')
                elapsed = data[2].replace('elapsed','').split(".")[0]
                #print (data[2], elapsed     )
                test_times.append(float(to_sec(elapsed)))

        l_idx += 1

    ret[this_config] = (np.mean(train_times), np.std(train_times), np.mean(test_times), np.std(test_times))

    print (len(ret))

    return ret

def main(args):

    experiment = args[1]
    folder = args[2]

    try:
        output = args[3]
    except IndexError:
        output = None

    try:
        xlog = args[4]
    except IndexError:
        xlog = None

    if experiment == 'rp':
        res = get_kfold_scores_rp(folder)
        times = get_times_rp(xlog)
    
    elif experiment == 'gtzan':
        pass
    
    elif experiment == 'ae':
        res = get_kfold_scores_ae(folder)
        times = get_times_ae(xlog)

    else:
        print ('invalid experiment code. It should be one of (rp, gtzan, ae)')

    print(len(res))

    heads = ';'.join([k.split('-')[1] for k in res[0][0].split('_')[:-1] ]) + ';f1_avg;f1_std;'

    if output is None:
        print(heads)
    else:
        output = open(output, 'w')
        output.write(heads + '\n')

    for cfg, avg, std in res:
        out_cfg = ';'.join([k.split('-')[0] for k in cfg.split('_')[:-1] ])
        if 'ivector' in out_cfg:
            out_cfg = ';' + out_cfg

        out = "%s;%.3f;%.3f" % (out_cfg, avg[1], std[1])

        if times is not None:
            k = cfg.replace('_*-f.eval', '')
            if k in times:
                train_avg, train_std, test_avg, test_std = times[k]

                out += ";%.3f;%.3f;%.3f;%.3f;" % (train_avg, train_std, test_avg, test_std)

        if output is None:
            print(out)
        else:
            output.write(out + '\n')

    if output is not None:
        output.close()

if __name__ == "__main__":
    main(sys.argv)
