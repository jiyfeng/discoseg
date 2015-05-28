## main.py
## Author: Yangfeng Ji
## Date: 05-03-2015
## Time-stamp: <yangfeng 05/03/2015 09:55:57>

"""
All in one module
1, build vocab
2, build training samples
3, train a segmentation model
"""

import buildvocab, buildsample, buildmodel, buildedu

# Fixed
trainpath = "data/train/"
devpath = "data/dev/"
fvocab = "data/sample/vocab.pickle.gz"
ftrain = "data/sample/train.pickle.gz"
fdev = "data/sample/dev.pickle.gz"
fmodel = "model/model.pickle.gz"
# Changable
testpath = "data/test/neg/"
writepath = "data/test/negedu/"

def main():
    ## Build vocab
    thresh = 5
    # buildvocab.main(trainpath, thresh, fvocab)
    ## Build training data
    # buildsample.main(trainpath, ftrain)
    ## Build dev data
    # buildsample.main(devpath, fdev)
    ## Training
    # clf = buildmodel.main(ftrain, fdev, fmodel)
    ## Segmentation
    buildedu.main(fmodel, fvocab, testpath, writepath)


if __name__ == '__main__':
    main()
