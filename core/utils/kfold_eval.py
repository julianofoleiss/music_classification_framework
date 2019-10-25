import numpy as np

def get_avg_std_kfold_scores(filenames):
    txts = []

    fold_scores = []

    for i in filenames:
        f = open(i)
        txts.append(f.readlines())
        f.close()
    
    for i in txts:
        for l in i:
            if l.strip().startswith('Accuracy:'):
                acc = l.split(' ')[-1].strip()
            if l.strip().startswith('F1-score:'):
                f1 = l.split(' ')[-1].strip()
            if l.strip().startswith('avg'):
                d = filter(None, l.split(' '))
                prec = d[3]
                rec = d[4]
        fold_scores.append( [ float(acc), float(f1), float(prec), float(rec) ])

    fold_scores = np.array(fold_scores)

    avgs = np.average(fold_scores, axis=0)
    stds = np.std(fold_scores, axis=0)

    return avgs, stds

if __name__ == '__main__':
    import sys
    import glob
    import matplotlib.pyplot as plt

    def plot_bars( data, err, labels, group_masks ):

        print data

        fig, ax = plt.subplots()

        ind = np.arange(len(group_masks[0]))

        w = 0.15

        offset = 0
        rs = []
        for m in group_masks:
            
            print ind

            r = ax.bar(ind + offset, data[m], w, yerr=err[m])

            offset += w

            rs.append(r[0])

        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score by Number of Frames and Selection Strategy')
        ax.set_xticks(ind + ((w/2) * len(labels)) )
        ax.set_xticklabels( ('linspace', 'kmeansc', 'kmeansf' ) )
        ax.set_xlim(right=4.5)

        ax.legend( rs, [ str(l) + ' frames' for l in labels ] )

        plt.show()

    all_avgs = []
    all_stds = []
    configs = []
    prefix = '/home/juliano/Doutorado/music_classification_framework/res_darthvader/res_folds1/res_gtzan44_rosa_norm_feats_shok_beats'

    #for nf in [40, 30, 20, 10, 5, 1]:
    for nf in [10, 5, 1]:
        for fs in ['linspace', 'kmeansc', 'kmeansf']:
            #files = sorted(glob.glob('%s/%d-nf_%s-fs_*-f.eval' % (prefix, nf, fs)  ))
            files = sorted(glob.glob('%s/%d-nf_%s-fs_median-agg_*-f.eval' % (prefix, nf, fs)  ))
            avgs, stds = get_avg_std_kfold_scores(files)
            all_avgs.append( avgs )
            all_stds.append( stds )
            configs.append('%d-nf_%s-fs' % (nf, fs) )

    print all_avgs

    all_avgs = np.array(all_avgs)
    all_stds = np.array(all_stds)

    print all_avgs
    print all_stds
    print configs, len(configs)

    group_masks = np.arange(len(all_avgs)).reshape((-1, 3))

    plot_bars(all_avgs[:,1], all_stds[:,1], [10, 5, 1], group_masks)

    #plot_bars(all_avgs[:,1], all_stds[:,1], [40, 30, 20, 10, 5, 1], group_masks)


