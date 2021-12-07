from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from keras import metrics
import numpy as np
from keras import backend as K
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint

def blosum62():
    f1 = open("BLOSUM62","r")
    l_AAS = []
    AAs = []
    scores = {}
    for line in f1.readlines():
        sp = line.split()
        aa = sp[0]
        AAs.append(aa)
    num = 0
    f1 = open("BLOSUM62","r")
    for line in f1.readlines():
        sp = line.split()
        for i in range(len(sp)):
            if i == 0:
                continue
            else:
                score = float(sp[i])
                aas = AAs[num] + "_" + AAs[i-1]
                aas2 = AAs[i-1] + "_" + AAs[num]
                if aas not in l_AAS and aas2 not in l_AAS:
                    l_AAS.append(aas)
                    scores[aas] = score
        num += 1
    return scores,l_AAS,AAs

def logistic_GPS(X: bytearray, Y: bytearray,PEP,type,turn,fold):
    best_coef = []
    best_auc=0
    solverchose = 'liblinear'
    Y_last = []
    Score_last = []
    num = 0
    skf = StratifiedKFold(n_splits=fold)
    for train_index, test_index in skf.split(X,Y):
        num += 1
        print("lg_" + str(num))
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clscv = LogisticRegressionCV(max_iter=10000, cv=10, solver=solverchose,scoring='roc_auc')
        clscv.fit(X_train, Y_train)
        regularization = clscv.C_[0] * (100**(-turn))
        print("C=" + str(regularization))
        cls = LogisticRegression(max_iter=10000,solver=solverchose,C=regularization)
        cls.fit(X_train, Y_train)
        list_coef = cls.coef_[0]
        predict_prob_x = cls.predict_proba(X_test)
        predict_x = predict_prob_x[:, 1]
        tem_auc = roc_auc_score(Y_test,np.array(predict_x))
        if tem_auc > best_auc:
            best_auc = tem_auc
            best_coef = list_coef
        Y_last = np.hstack((Y_last, Y_test))
        Score_last = np.hstack((Score_last, predict_x))
    auc = roc_auc_score(np.array(Y_last), np.array(Score_last))

    return best_coef,auc

def WeightAndMatrix(path):
    f1 = open(path, "r")
    weights = []
    l_AAS = []
    AAs = []
    scores = {}
    for line in f1.readlines():
        if line.startswith(" A "):
            sp = line.strip().split()
            for i in range(len(sp)):
                aa = sp[i]
                AAs.append(aa)
        if line.startswith("@weight"):
            sp = line.strip().split("\t")
            for i in range(len(sp))[1:]:
                w = float(sp[i])
                weights.append(w)
    num = 0
    f1 = open(path, "r")
    t = False
    for line in f1.readlines():
        if t:
            sp = line.strip().split()
            for i in range(len(sp)):
                if i == 0:
                    continue
                else:
                    score = float(sp[i])
                    aas = AAs[num] + "_" + AAs[i - 1]
                    aas2 = AAs[i - 1] + "_" + AAs[num]
                    if aas not in l_AAS and aas2 not in l_AAS:
                        l_AAS.append(aas)
                    scores[aas] = score
            num += 1
        if line.startswith(" A "):
            t = True
    return scores, l_AAS, weights, AAs

def dnn(X,Y,nfold,parameter,PEP):
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
        my_class_weight = class_weight.compute_class_weight('balanced'
                                                            , np.unique(Y_train)
                                                            , Y_train).tolist()
        class_weight_dict = dict(zip([x for x in np.unique(Y_train)], my_class_weight))
        model = create_model(parameter)
        model.fit(X_train, Y_train, epochs=1000, batch_size=100,validation_data=(X_test,Y_test),verbose=1,class_weight=class_weight_dict,
                  callbacks=[EarlyStopping(monitor="val_auc", mode="max", min_delta=0, patience=100),
                             ModelCheckpoint(str(num) +'.model', monitor="val_auc", mode="max", save_best_only=True)])
        model = load_model(str(num) +".model")
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

def create_model(parameter):
    model = Sequential()
    model.add(Dense(parameter[0], activation='linear', input_dim=parameter[3]))
    model.add(Dropout(parameter[1]))
    for i in range(parameter[2]):
        fold = 2 ** (i+1)
        model.add(Dense(parameter[0] / fold,activation='linear'))
        model.add(Dropout(parameter[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(lr=1e-3,decay=3e-5), loss='binary_crossentropy', metrics=[metrics.AUC(name="auc")])
    model.summary()

    return model