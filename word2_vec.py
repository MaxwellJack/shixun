import logging
import jieba
import pandas as pd
import os
import sys
import gensim
import numpy as np
import codecs
from gensim.models import Word2Vec
#将旅游数据保存word2vec为vvvv.csv
# 返回特征词向量
def getWordVecs(wordList, model):
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        # print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
# 构建文档词向量
def buildVecs(filename, model,numm):
    fileVecs = []
    n=[]
    for line in filename:
        numm = numm + 1
        logger.info("Start line: " + str(line))
        wordList = line
        vecs = getWordVecs(wordList, model)
        # print vecs
        # sys.exit()
        # for each sentence, the mean vector of all its vectors is used to represent this sentence
        if len(vecs) > 0:
            vecsArray = sum(np.array(vecs)) / len(vecs)  # mean
            # print vecsArray
            # sys.exit()
            fileVecs.append(vecsArray)
        else :
            print (numm)
            n.append(numm)
    return fileVecs,n
# stop_words = [line.decode('utf-8', 'ignore').strip() for line in open(r"C:\Users\Administrator\Desktop\senti_analysis-master\data\stopWord.txt",'rb').readlines()]
# logging.info(stop_words)
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
text_list = []
# 读取csv文件
reviews = pd.read_csv(r"C:\Users\Administrator\Documents\WeChat Files\c1300262050\Files\datafinal(1).csv", encoding='utf-8')
for idx, row in reviews.iterrows():
    print (idx)
    review_content = row['comment']
    seg_list = jieba.cut(str(review_content), cut_all=False)
    word_list = [item for item in seg_list if len(item) > 1]
    text_list.append(word_list)
    # text_list.append(list(set(word_list) - set(stop_words)))
print ("加载模型")
model = gensim.models.KeyedVectors.load_word2vec_format(r"J:\word2vec\wiki_zh_word2vec\wiki.zh.text.vector", binary=False)
print("计算vecs")
# sumvecs=buildVecs(text_list,model)
#text_list和revies不一样
num0=0
num1=0
num2=0
num3=0
num4=0
num5=0
s0=reviews[reviews["score"]==0].index.values
s1=reviews[reviews["score"]==1].index.values
s2=reviews[reviews["score"]==2].index.values
s3=reviews[reviews["score"]==3].index.values
s4=reviews[reviews["score"]==4].index.values
s5=reviews[reviews["score"]==5].index.values
s0vecs,num0kong=buildVecs(text_list[s0[0]:s0[-1]+1],model,num0)
s1vecs,num1kong=buildVecs(text_list[s1[0]:s1[-1]+1],model,num1)
s2vecs,num2kong=buildVecs(text_list[s2[0]:s2[-1]+1],model,num2)
s3vecs,num3kong=buildVecs(text_list[s3[0]:s3[-1]+1],model,num3)
s4vecs,num4kong=buildVecs(text_list[s4[0]:s4[-1]+1],model,num4)
s5vecs,num5kong=buildVecs(text_list[s5[0]:s5[-1]+1],model,num5)
sumkong=[]
for i in num1kong:
    sumkong.append(len(s0)+i)
for i in num2kong:
    sumkong.append(len(s0)+i+len(s1))
for i in num3kong:
    sumkong.append(len(s0)+i+len(s1)+len(s2))
for i in num4kong:
    sumkong.append(len(s0)+i+len(s1)+len(s2)+len(s3))
for i in num5kong:
    sumkong.append(len(s0)+i+len(s1)+len(s2)+len(s3)+len(s4))

# s5vecs=buildVecs(reviews[s5[0]:s5[-1]+1],model)
# s5vecs=buildVecs(reviews[reviews["score"]==5]["comment"],model)
Y = np.concatenate((np.zeros(len(s0)), np.ones(len(s1)),np.array(len(s2)*[2]),np.array(len(s3)*[3]),np.array(len(s4)*[4]),np.array(len(s5)*[5])))
# Y=np.concatenate((Y,np.array(len(s4)*[4]),np.array(len(s5)*[5]),np.array(len(s3)*[3])))
Y=np.delete(Y, sumkong, 0)
sumvecs=[]
sumvecs=s0vecs+s1vecs+s2vecs+s3vecs+s4vecs+s5vecs
print("保存文件")
# for i in range(len(sumvecs)):
#     sample = sumvecs[i]
#     for j in range(len(sample)):
#         if np.isnan(sample[j]):
#             sample[j] = 0
df_x = pd.DataFrame(sumvecs)
df_y = pd.DataFrame(Y)
data = pd.concat([df_y, df_x], axis=1)
print( data)
data.to_csv(r"C:\Users\Administrator\Desktop\vvvv.csv")
# logging.info(text_list)
# text=[]
# for i in range(len(text_list)):
#     text.extend(j for j in text_list[i])
# #len(text) 1011537
# text=list(set(text))
# # len(text)
# #  50011
# #停用词
# stopkey = [w.strip() for w in codecs.open(r"C:\Users\Administrator\Desktop\senti_analysis-master\data\stopWord.txt", 'r', encoding='utf-8').readlines()]
# for i in stopkey:
#     if  i in text:
#         text.remove(i)
# model = Word2Vec(text)
# # model.save("word.model")
# print('xijie=',model.most_similar('西街'))
# # print(model['通纳'])
# print(model.similarity("孩子", "儿童票"))
# print(model.vocab)
