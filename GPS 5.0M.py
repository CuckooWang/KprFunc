from Tools import blosum62
import numpy as np
from Tools import logistic_GPS
from sklearn.metrics import roc_auc_score

def trainning():
    length = 30
    fold = 10
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

    print("The first round of training。。。。。。。。。。。。。")
    AAscores, l_aas, AAs = blosum62()
    l_scores, l_type,l_peps = getWeightScoreType(pos, neg, AAscores, AAs,length)
    raw_scores = []
    for i in range(len(l_scores)):
        total = 0.0
        for j in range(len(l_scores[i])):
            total += l_scores[i][j]
        raw_scores.append(total)
    R = np.array(raw_scores)
    X = np.array(l_scores)
    Y = np.array(l_type)
    PEP = np.array(l_peps)
    raw_auc = roc_auc_score(Y, R)
    weight_coef, weight_auc = logistic_GPS(X, Y,PEP,"WW",0,fold)
    #MM training
    l_scores, l_type,peps = getMMScoreType(pos, neg, AAscores, weight_coef, l_aas, AAs,length)
    raw_scores = []
    for i in range(len(l_scores)):
        total = 0.0
        for j in range(len(l_scores[i])):
            total += l_scores[i][j]
        raw_scores.append(total)
    R = np.array(raw_scores)
    X = np.array(l_scores)
    Y = np.array(l_type)
    PEP = np.array(l_peps)
    MM_coef,MM_auc= logistic_GPS(X, Y,PEP,"MM",0,fold)
    best_weight_auc = weight_auc
    best_MM_auc = MM_auc
    file = "traningout_first_" + str(fold) + ".txt"
    writeParameter_MM(file, length, length, AAscores, l_aas, weight_coef, MM_coef, MM_auc)
    AAscores = newAAScore(AAscores, l_aas, AAs, MM_coef)
    for i in range(100)[1:]:
        print("The" + str(i + 1) + "th round of training。。。。。。。。。。。。。")
        l_scores, l_type,l_peps = getWeightScoreType(pos, neg, AAscores, AAs, length)
        X = np.array(l_scores)
        Y = np.array(l_type)
        PEP = np.array(l_peps)
        weight_coef, weight_auc = logistic_GPS(X, Y,PEP,"WW",i,fold)
        if weight_auc > best_weight_auc:
            best_weight_auc = weight_auc
            file2 = "traningout_weight_best_" + str(fold) + "_" + str(i+1) + ".txt"
            writeParameter_WW(file2, length, length, AAscores, l_aas, weight_coef, weight_auc)
        # MM training
        l_scores, l_type, peps = getMMScoreType(pos, neg, AAscores, weight_coef, l_aas, AAs, length)
        X = np.array(l_scores)
        Y = np.array(l_type)
        PEP = np.array(l_peps)
        MM_coef, MM_auc = logistic_GPS(X, Y,PEP,"MM",i,fold)
        if MM_auc > best_MM_auc:
            best_MM_auc = MM_auc
            file2 = "traningout_MM_best_" + str(fold) + "_" + str(i+1) + ".txt"
            writeParameter_MM(file2, length, length, AAscores, l_aas, weight_coef, MM_coef, MM_auc)
        else:
            break
        AAscores = newAAScore(AAscores, l_aas, AAs, MM_coef)

def newAAScore(AAscores, l_aas, AAs, MM_coef):
    dict_weight = {}
    for i in range(len(l_aas)):
        aas = l_aas[i]
        score = AAscores[aas]
        mweight = MM_coef[i]
        newscore = score * mweight
        dict_weight[aas] = newscore
    return dict_weight

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

def getMMScoreType(pos, neg, matrix, weights, l_aas, AAs,length):
    scorespos = []
    scoresneg = []
    for i in range(length*2+1):
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

        score = (score / (len(pos) -1 )).tolist()

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

def writeParameter_WW(file,left,right,AAscores,l_aas,weight_coef,weight_auc):
    fw = open(file, "w")
    list_aa = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
               "B", "Z", "X", "*"]
    dict_weight = {}
    for i in range(len(l_aas)):
        aas = l_aas[i]
        score = AAscores[aas]
        dict_weight[aas] = score
    fw.write("#GPS-Uber 1.0 Parameters\n")
    fw.write("#Version: 1.0\n")
    fw.write("#By Chenwei Wang    @HUST\n")
    fw.write("@param\tCode=K\tUp=" + str(left) + "\tDown=" + str(right) + "\n")
    fw.write("@AUC=" + str(weight_auc) + "\n")
    fw.write("@weight")
    for i in range(len(weight_coef)):
        fw.write("\t" + str(weight_coef[i]))
    fw.write("\n")

    for i in range(len(list_aa)):
        a = list_aa[i]
        fw.write(" " + a)
    fw.write("\n")

    for i in range(len(list_aa)):
        a1 = list_aa[i]
        fw.write(a1)
        for j in range(len(list_aa)):
            a2 = list_aa[j]
            aas1 = a1 + "_" + a2
            aas2 = a2 + "_" + a1
            score = 0.0
            if aas1 in dict_weight:
                score = dict_weight[aas1]
            elif aas2 in dict_weight:
                score = dict_weight[aas2]
            else:
                print(aas1 + "no score!!")
            fw.write(" " + str(score))
        fw.write("\n")

    fw.flush()
    fw.close()

def writeParameter_MM(file,left,right,AAscores,l_aas,weight_coef,MM_coef,MM_auc):
    fw = open(file, "w")
    list_aa = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
               "B", "Z", "X", "*"]
    dict_weight = {}
    for i in range(len(l_aas)):
        aas = l_aas[i]
        score = AAscores[aas]
        mweight = MM_coef[i]
        newscore = score * mweight
        dict_weight[aas] = newscore
    fw.write("#GPS-Uber 1.0 Parameters\n")
    fw.write("#Version: 1.0\n")
    fw.write("#By Chenwei Wang    @HUST\n")
    fw.write("@param\tCode=K\tUp=" + str(left) + "\tDown=" + str(right) + "\n")
    fw.write("@AUC=" + str(MM_auc) + "\n")
    fw.write("@weight")
    for i in range(len(weight_coef)):
        fw.write("\t" + str(weight_coef[i]))
    fw.write("\n")

    for i in range(len(list_aa)):
        a = list_aa[i]
        fw.write(" " + a)
    fw.write("\n")

    for i in range(len(list_aa)):
        a1 = list_aa[i]
        fw.write(a1)
        for j in range(len(list_aa)):
            a2 = list_aa[j]
            aas1 = a1 + "_" + a2
            aas2 = a2 + "_" + a1
            score = 0.0
            if aas1 in dict_weight:
                score = dict_weight[aas1]
            elif aas2 in dict_weight:
                score = dict_weight[aas2]
            else:
                print(aas1 + "no score!!")
            fw.write(" " + str(score))
        fw.write("\n")

    fw.flush()
    fw.close()



trainning()




