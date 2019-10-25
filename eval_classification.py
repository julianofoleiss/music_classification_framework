import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def print_cm(cm, labels, hide_labels=True, file=None, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    #TODO: implementar hide_labels para saida em arquivo
    columnwidth = max([len(x) for x in labels]+[0]) # 5 is value length

    if hide_labels:
        columnwidth = 3

    empty_cell = " " * columnwidth
    # Print header
    if file is None:
        print "    " + empty_cell,
    else:
        file.write("    " + empty_cell)
    i = 0
    for label in labels:
        if file is None:
            if hide_labels:
                print "%{0}d".format(columnwidth) % i,
            else:
                print "%{0}s".format(columnwidth) % label,
        else:
            file.write("%{0}s".format(columnwidth) % label)
        i+=1
    if file is None:
        print
    else:
        file.write("\n")
    # Print rows
    for i, label1 in enumerate(labels):
        if file is None:
            if hide_labels:
                print "    %{0}d".format(columnwidth) % i,
            else:
                print "    %{0}s".format(columnwidth) % label1,
        else:
            file.write("    %{0}s".format(columnwidth) % label1)
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if file is None:
                print cell,
            else:
                file.write(cell)
        if file is None:
            print
        else:
            file.write("\n")

    for i, l in enumerate(labels):
        print "%d: %s" % (i, l)

def canon_file(contents):
    canon = []
    for k in contents:
        d = k.split("\t")
        d[1] = d[1].strip()
        canon.append(d)
    canon = sorted(canon, key=lambda x: x[0])

    return canon

def eval_classification(args):

    predict_filename = args[1]
    gt_filename = args[2]

    with open(gt_filename) as f:
        gt = f.readlines()
    gt = canon_file(gt)

    gtd = dict()
    for i in gt:
        gtd[i[0]] = i[1]

    with open(predict_filename) as f:
        pred = f.readlines()
    pred = canon_file(pred)

    predicted = []
    truth = []

    for i in pred:
        predicted.append(i[1])
        if gtd.has_key(i[0]):
	        truth.append(gtd[i[0]])
        else:
	        print "truth for %s not in label file!" % (i[0])
	        exit(1)


    print "Accuracy: %.2f" % accuracy_score(truth, predicted)
    print "F1-score: %.2f" % f1_score(truth, predicted, average='weighted')

    cm = confusion_matrix(truth, predicted)

    print_cm(cm, sorted(list(set(truth))))

    print classification_report(truth, predicted)

if __name__ == "__main__":
    eval_classification(sys.argv)

