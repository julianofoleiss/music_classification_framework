import os
import sys
import click
import numpy as np
import copy
from pprint import pprint

@click.command()
@click.argument(
    'experiment_output',
    type=click.File('r'),
    )
@click.option(
    '--csv_file',
    type=click.File('w'),
    help='Name of the output csv file. If not provided, output will be printed to stdout.'
)
def parse(experiment_output, csv_file):
    texture_selector = None
    textures_number = None
    prev_texture_selector = None
    prev_textures_number = None

    fold_scores_zeroed = {
        'svm' : {
            'acc': [],
            'f1': []
        },
        'svm_anova' : {
            'acc': [],
            'f1': []
        },        
        'knn' : {
            'acc': [],
            'f1': []
        },
        'knn_anova' : {
            'acc': [],
            'f1': []
        }
    }

    contents = experiment_output.readlines()
    contents = [c.strip() for c in contents]

    #pprint (contents)

    fold_scores = None
    scores = []

    def calculate_stats(fold_scores, scores, prev_texture_selector, prev_textures_number):
        if fold_scores is not None:
            for c in ['svm', 'knn', 'knn_anova', 'svm_anova']:
                for s in ['acc', 'f1']:
                    fold_scores[c]['avg_' + s ] = np.average(fold_scores[c][s])
                    fold_scores[c]['std_' + s ] = np.std(fold_scores[c][s])
            scores.append((prev_texture_selector, prev_textures_number, fold_scores))
        fold_scores = copy.deepcopy(fold_scores_zeroed)
        return fold_scores

    for ln in range(len(contents)):
        l = contents[ln]

        if l.startswith('+++'):
            prev_textures_number = textures_number
            textures_number = int(l.split(' ')[3])
            #print textures_number

        if l.startswith('###'):
            prev_texture_selector = texture_selector
            texture_selector = l.split(' ')[3]
            #print texture_selector

        if l.startswith('@@@'):
            if int(l.split(' ')[2]) == 1:
                #print texture_selector, textures_number
                fold_scores = calculate_stats(fold_scores, scores, prev_texture_selector, prev_textures_number if prev_texture_selector == 'linspace' else textures_number)

        if l.startswith('max voting & evaluating SVM'):
            acc = float(contents[ln+1].split(' ')[1])
            f1 = float(contents[ln+2].split(' ')[1])
            fold_scores['svm']['acc'].append(acc)
            fold_scores['svm']['f1'].append(f1)

        if l.startswith('max voting & evaluating KNN (NO ANOVA) results'):
            acc = float(contents[ln+1].split(' ')[1])
            f1 = float(contents[ln+2].split(' ')[1])
            fold_scores['knn']['acc'].append(acc)
            fold_scores['knn']['f1'].append(f1)

        if l.startswith('max voting & evaluating KNN (ANOVA) results'):
            try:
                acc = float(contents[ln+1].split(' ')[1])
                f1 = float(contents[ln+2].split(' ')[1])
                fold_scores['knn_anova']['acc'].append(acc)
                fold_scores['knn_anova']['f1'].append(f1)
            except ValueError:
                print('Error parsing line %d, skipping fold for KNN (ANOVA)' % ln )
                acc = 0
                f1 = 0
                
        if l.startswith('max voting & evaluating SVM (ANOVA) results'):
            acc = float(contents[ln+1].split(' ')[1])
            f1 = float(contents[ln+2].split(' ')[1])
            fold_scores['svm_anova']['acc'].append(acc)
            fold_scores['svm_anova']['f1'].append(f1)    
    
    calculate_stats(fold_scores, scores, texture_selector, textures_number)

    header = "classifier;ts;nt;avg_acc;std_acc;avg_f1;std_f1\n"

    svm_output = header
    knn_output = header
    knn_anova_output = header
    svm_anova_output = header
	
    for i in scores:
        svm_output += "svm;%s;%s;%.2f;%.2f;%.2f;%.2f\n" % (i[0], i[1], 
            i[2]['svm']['avg_acc'], i[2]['svm']['std_acc'], i[2]['svm']['avg_f1'], i[2]['svm']['std_f1'])

        svm_anova_output += "svm_anova;%s;%s;%.2f;%.2f;%.2f;%.2f\n" % (i[0], i[1], 
            i[2]['svm_anova']['avg_acc'], i[2]['svm_anova']['std_acc'], i[2]['svm_anova']['avg_f1'], i[2]['svm_anova']['std_f1'])
        
        knn_output += "knn;%s;%s;%.2f;%.2f;%.2f;%.2f\n" % (i[0], i[1], 
            i[2]['knn']['avg_acc'], i[2]['knn']['std_acc'], i[2]['knn']['avg_f1'], i[2]['knn']['std_f1'])

        knn_anova_output += "knn_anova;%s;%s;%.2f;%.2f;%.2f;%.2f\n" % (i[0], i[1], 
            i[2]['knn_anova']['avg_acc'], i[2]['knn_anova']['std_acc'], i[2]['knn_anova']['avg_f1'], i[2]['knn_anova']['std_f1'])

    if csv_file is not None:
        csv_file.writelines(svm_output)
        csv_file.writelines(svm_anova_output)
        csv_file.writelines(knn_output)
        csv_file.writelines(knn_anova_output)
    else:
        print svm_output
        print svm_anova_output
        print knn_output
        print knn_anova_output

if __name__ == '__main__':
    parse()
