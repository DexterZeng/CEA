import Levenshtein
from include.Config import Config
import re
import numpy as np

# could not remove comas, as variants...
lowbound = 0; highbound = 10500

inf = open(Config.ill)
ent1ids = []
ent2ids = []
for i1, line in enumerate(inf):
    strs = line.strip().split('\t')
    if i1 >= lowbound and i1 < highbound:
        ent1ids.append(strs[0])
        ent2ids.append(strs[1])


inf1 = open(Config.e1)
id2name1_test = dict()
for i1, line in enumerate(inf1):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
    wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
    # if i1>=lowbound and i1<highbound:
    id2name1_test[strs[0]] = wordline
print(len(id2name1_test))

inf2 = open(Config.e2)
id2name2_test = dict()
for i1, line in enumerate(inf2):
    strs = line.strip().split('\t')
    wordline = strs[1].split('/')[-1].lower().replace('(','').replace(')','')
    wordline = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "",wordline)
    # if i1>=lowbound and i1<highbound:
    id2name2_test[strs[0]] = wordline
print(len(id2name2_test))

overallscores = []
for item in range(highbound-lowbound):
    print(item)
    name1 = id2name1_test[ent1ids[item]]
    scores = []
    for item2 in range(highbound-lowbound):
        name2 = id2name2_test[ent2ids[item2]]
        scores.append(Levenshtein.ratio(name1, name2))
    overallscores.append(scores)

print(np.array(overallscores))


np.save('./data/'+ Config.language + '/string_mat_train.npy', np.array(overallscores))

