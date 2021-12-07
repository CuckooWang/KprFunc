import numpy as np
from Tools import WeightAndMatrix
from Tools import dnn
from sklearn.metrics import roc_auc_score
def final_dnn():
        length = 30
        nfold = 10
        f1 = open("POS.txt", "r")
        f2 = open("NEG.txt", "r")

        pos = set()
        neg = set()
        print("Reading positive dataset")
        for line in f1.readlines():
            sp = line.strip().split("\t")
            pep = sp[0]
            pos.add(pep)
        print("Reading negative dataset")
        for line in f2.readlines():
            sp = line.strip().split("\t")
            pep = sp[0]
            if pep not in pos:
                neg.add(pep)

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
        auc_all,best_model = dnn(X,Y,nfold,parameter,PEP)
        print(auc_all)

def getWeightScoreType(pos, neg, matrix, AAs,length):
    scores = []
    for i in range(length*2+1):
        pos_score = []
        for j in range(len(AAs)):
            aa1 = AAs[j]
            score = 0.0
            for oth in pos:
                aa2 = oth[i:i + 1]
                aas = aa1 + "_" + aa2
                aas2 = aa2 + "_" + aa1
                if aas in matrix:
                    score += matrix[aas]
                else:
                    score += matrix[aas2]
            pos_score.append(score)
        scores.append(pos_score)

    l_scores = []
    l_type = []
    l_peps = []

    for pep in pos:
        score = []
        for i in range(len(pep)):
            aa = pep[i:i + 1]
            index = AAs.index(aa)
            aascore = (scores[i][index] - matrix[aa + "_" + aa]) / (len(pos) - 1)
            score.append(aascore)
        l_scores.append(score)
        l_type.append(1)
        l_peps.append(pep)

    for pep in neg:
        score = []
        for i in range(len(pep)):
            aa = pep[i:i + 1]
            index = AAs.index(aa)
            aascore = scores[i][index] / len(pos)
            score.append(aascore)
        l_scores.append(score)
        l_type.append(0)
        l_peps.append(pep)

    return l_scores, l_type,l_peps

def getMMScoreType(pos, neg, matrix, weights, l_aas, AAs, length):
    scorespos = []
    scoresneg = []
    for i in range(length * 2 + 1):
        score_pos = []
        score_neg = []
        for j in range(len(AAs)):
            aa1 = AAs[j]
            score = []
            for z in range(len(l_aas)):
                score.append(0.0)
            for oth in pos:
                aa2 = oth[i:i + 1]
                aas1 = aa1 + "_" + aa2
                aas2 = aa2 + "_" + aa1
                if aas1 in l_aas:
                    index = l_aas.index(aas1)
                    score[index] += matrix[aas1] * weights[i]
                elif aas2 in l_aas:
                    index = l_aas.index(aas2)
                    score[index] += matrix[aas2] * weights[i]
            scoreneg = np.array(score)
            index2 = l_aas.index(aa1 + "_" + aa1)
            score[index2] -= matrix[aa1 + "_" + aa1] * weights[i]
            scorepos = np.array(score)
            score_pos.append(scorepos)
            score_neg.append(scoreneg)
        scorespos.append(score_pos)
        scoresneg.append(score_neg)

    l_scores = []
    l_type = []
    l_peps = []

    for pep in pos:
        score = getArray(l_aas)
        for i in range(len(pep)):
            aa = pep[i:i + 1]
            index = AAs.index(aa)
            scoreary = scorespos[i][index]
            score += scoreary

        score = (score / (len(pos) - 1)).tolist()

        l_scores.append(score)
        l_type.append(1)
        l_peps.append(pep)

    for pep in neg:
        score = getArray(l_aas)
        for i in range(len(pep)):
            aa = pep[i:i + 1]
            index = AAs.index(aa)
            scoreary = scoresneg[i][index]
            score += scoreary

        score = (score / len(pos)).tolist()

        l_scores.append(score)
        l_type.append(0)
        l_peps.append(pep)
    return l_scores, l_type, l_peps

def getArray(l_aas):
    score = []
    for i in range(len(l_aas)):
        score.append(0.0)
    scoreary = np.array(score)

    return scoreary

final_dnn()