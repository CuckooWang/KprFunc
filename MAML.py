from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold,StratifiedKFold
from keras import metrics
from keras.models import Model
import numpy as np
import copy
from keras import backend as K
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
import random
from Tools import getMMScoreType,WeightAndMatrix
def fetshot():

    length = 30
    nfold = 5
    f1 = open(r"functionsite.txt", "r")
    f2 = open(r"POS.txt", "r")
    funcinf = set()
    for line in f1.readlines():
        site = line.strip()
        funcinf.add(site)
    pos = []
    neg = []
    for line in f2.readlines():
        sp = line.strip().split("\t")
        pep = sp[0]
        site = sp[1] + "\t" + sp[2]
        if site in funcinf:
            pos.append(pep)
        else:
            neg.append(pep)
    print(len(pos))
    fw = open("MAML_AUCs.txt", "a")
    pos_size = len(pos)
    for a in range(10000):
        if len(neg) > pos_size*6:
            new_neg = random.sample(neg,pos_size*5)
            tem_neg = copy.deepcopy(neg)
            for j in range(len(tem_neg)):
                negpep = tem_neg[j]
                if negpep in new_neg:
                    neg.remove(negpep)
            print(len(neg))

            AAscores, l_aas, weight_coef, AAs = \
                WeightAndMatrix("traningout_best.txt")
            l_scores, l_type, peps = getMMScoreType(pos, new_neg, AAscores, weight_coef, l_aas, AAs, length)
            raw_scores = []
            for i in range(len(l_scores)):
                total = 0.0
                for j in range(len(l_scores[i])):
                    total += l_scores[i][j]
                raw_scores.append(total)
            X = np.array(l_scores)
            Y = np.array(l_type)
            PEP = np.array(peps)
            parameter = [512, 0.2, 2, X.shape[1]]
            auc_all,best_model = dnn(X,Y,nfold,parameter,PEP,a)
            fw.write(str(a+1) + "\tBest:" + "\t" + str(auc_all) + "\t" + str(best_model) + "\n")
            fw.flush()
        else:
            AAscores, l_aas, weight_coef, AAs = \
                WeightAndMatrix("traningout_best.txt")
            l_scores, l_type, peps = getMMScoreType(pos, neg, AAscores, weight_coef, l_aas, AAs, length)
            raw_scores = []
            for i in range(len(l_scores)):
                total = 0.0
                for j in range(len(l_scores[i])):
                    total += l_scores[i][j]
                raw_scores.append(total)
            X = np.array(l_scores)
            Y = np.array(l_type)
            PEP = np.array(peps)
            parameter = [512, 0.2, 2, X.shape[1]]
            auc_all, best_model = dnn(X, Y, nfold, parameter, PEP, a)
            fw.write(str(a + 1) + "\tBest:" + "\t" + str(auc_all) + "\t" + str(best_model) + "\n")
            fw.flush()
            break
    fw.flush()
    fw.close()


def dnn(X,Y,nfold,parameter,PEP,a):
    skf = StratifiedKFold(n_splits=nfold)
    num = 0
    best_auc = 0.0
    best_model = 0
    Y_last = []
    Score_last = []
    for train_index, test_index in skf.split(X, Y):
        num += 1
        print("dnn_" + str(num))
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model = load_model("original.model")
        for i in range(6):
            model.layers[i].trainable = False
        model.compile(optimizer=optimizers.Adam(lr=1e-3, decay=3e-5), loss='binary_crossentropy',
                      metrics=[metrics.AUC(name="auc")])
        model.fit(X_train, Y_train, epochs=300, batch_size=8,validation_data=(X_test,Y_test),verbose=1,
                  callbacks=[EarlyStopping(monitor="val_auc", mode="max", min_delta=0, patience=10),
                             ModelCheckpoint(str(a+1) + "_" + str(num) +'.model', monitor="val_auc", mode="max", save_best_only=True)])
        model = load_model(str(a+1) + "_" + str(num) +".model")
        predict_x = model.predict(X_test)[:, 0]
        auc = roc_auc_score(Y_test, predict_x)
        if auc > best_auc:
            best_auc = auc
            best_model = num
        Y_last.extend(Y_test)
        Score_last.extend(predict_x)
        K.clear_session()
    auc_all = roc_auc_score(np.array(Y_last), np.array(Score_last))
    return auc_all,best_model


fetshot()
