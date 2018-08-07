# shixun
大三机器学习实训的代码
这次大三的实训我选的是机器学习，研究的部分是情感分析（都是自己瞎捉摸）。

首先是数据获取，我们爬取的网站有百度旅游、大众点评、驴妈妈、猫途鹰、携程。
爬虫文章百度旅游、大众点评、驴妈妈、猫途鹰、携程关于评论的爬虫总结（附源码） 链接：https://blog.csdn.net/ssssdbucdbod/article/details/81272905
其中爬虫的代码上传到https://github.com/CharlesAlison/pinglun_spyder

接着是词向量获取文章利用Word2vec将旅游评论数据转化为词向量 链接：https://blog.csdn.net/ssssdbucdbod/article/details/81483278 。

代码为https://github.com/CharlesAlison/shixun/blob/master/word2_vec.py

其中有维基百科的word2vec的训练，代码和原理链接：https://github.com/CharlesAlison/wiki_zh_word2vec

最后是对词向量的训练，我们有两种方式
利用最大熵模型来训练词向量 链接：https://blog.csdn.net/ssssdbucdbod/article/details/81487667
其中遇到问题 MaxentClassifier.train（）遇到错误AttributeError: 'list' object has no attribute 'items' 链接：https://blog.csdn.net/ssssdbucdbod/article/details/81149814

KNN、贝叶斯来训练词向量，链接：https://blog.csdn.net/ssssdbucdbod/article/details/81488062



