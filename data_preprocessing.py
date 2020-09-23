import pandas as pd
file = open("persian_NER_300K.txt","r",encoding="UTF-8")
tags = []
words = []
exel = []
sentence = 0
flag = 0
first = 0
tmp = ["Sentence #","Word","Tag"]
exel.append(tmp)
for line in file:
    line = line.strip()
    newline = line.split("\t")
    temp = []
    if (flag == 1):
        if (len(line) == 0):
            continue
        temp.append("Sentence: "+str(sentence))
        temp.append(newline[0])
        temp.append(newline[-1])
        flag = 0
    else:
        if (len(line) != 0):

            if (first == 0):
                sentence = sentence+1
                temp.append("Sentence: " + str(sentence))
                first = 1
            else:
                temp.append("")
            temp.append(newline[0])
            temp.append(newline[-1])
        else:
            flag = 1
            sentence = sentence + 1
            continue
    exel.append(temp)
print(exel)
pd.DataFrame(exel).to_csv("Processed.csv",index=False,header=False,encoding="UTF-8")