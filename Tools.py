import numpy as np
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

    # num = 0
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
