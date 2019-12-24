# learnDataMining 数据挖掘作业
*谢凡-201934739*

## 实验目的
测试sklearn中K-Means、Affinity propagation、Mean-shift、Spectral、Ward hierarchical clustering、 Agglomerative clustering、DBSCAN、Gaussian mixtures聚类算法在sklearn.datasets.load_digits、sklearn.datasets.fetch_20newsgroups这两个数据集上的聚类效果。

## 数据预处理
**load_digits**手写数字数据，在sklearn中被保存为一个numpy.nparray，这个ndarray共1797行、64列，每一行都是一个手写数字，8*8的像素被保存在一行中。我们需要对digits进行标准化，考虑到example对digit进行了降维并在二维上进行聚类且可视化，本实验针对digits的所有聚类也基于降维：本实验采用PCA（主成分分析）对digits降维，在二维的映射点如下：
![图片标题](https://raw.githubusercontent.com/doubvan/learnDataMining/master/img/1.png)

**fetch_20newsgroups**则是新闻语料数据，是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。一些新闻组的主题特别相似，还有一些却完全不相关 。
我们需要对newsgroups进行提取特征，本实验采用是TF-IDF的方法。但提取的TF-IDF 向量是非常稀疏的，超过30000维的特征才有159个非零特征。所以在后续的处理中，运算速度都比较缓慢。
因为数据的维度问题，Newsgroup并没有做可视化处理。

## 实验内容
### 1.k-means
k-means步骤是随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。聚类中心以及分配给它们的对象就代表一个聚类。

digits:
NMI: 0.464
Homogeneity: 0.458
Completeness: 0.471

Newsgroups:
NMI: 0.464
Homogeneity: 0.458
Completeness: 0.471

### 2.MeanShift
meanshift算法也叫做均值漂移，在目标追踪中应用广泛，本身其实是一种基于密度的聚类算法。

digits:
NMI: 0.454
Homogeneity: 0.442
Completeness: 0.466

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560


### 3.affinity propagation
affinity propagation(AP)是一种基于数据点之间的“信息传递”的聚类算法。与k-means等其它聚类算法不同的是，AP不需要在聚类前确定或估计类的个数。

digits:
NMI: 0.418
Homogeneity: 0.493
Completeness: 0.356

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560


### 4.SpectralClustering
谱聚类对数据分布的适应性更强，聚类效果也很优秀，同时聚类的计算量也小很多。它的主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，通过对所有数据点组成的图进行切图，让切图后不同的子图间边权重和尽可能的低，而子图内的边权重和尽可能的高，从而达到聚类的目的。

digits:
NMI: 0.412
Homogeneity: 0.307
Completeness: 0.553

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560

### 5.Ward Hierarchical clustering
分层聚类通过依次合并或拆分簇来构造嵌套的树形簇结构。Ward Hierarchical clustering即Agglomerative clustering的linkage type为ward：合并策略为最小化所有群集内的平方差之和。

digits:
NMI: 0.451 
Homogeneity: 0.444 
Completeness: 0.458

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560

### 6.DBSCAN
DBSCAN是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。通俗点说就是通过将紧密相连的样本划为一类，得到聚类类别。通过将所有各组紧密相连的样本划为各个不同的类别，则得到了最终的所有聚类类别结果。

digits:
NMI: 0.323
Homogeneity: 0.415
Completeness: 0.252

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560

### 7.Agglomerative clustering
分层聚类通过依次合并或拆分簇来构造嵌套的树形簇结构。

digits:
NMI: 0.397
Homogeneity: 0.366
Completeness: 0.430

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560

### 8.Gaussian mixtures
高斯混合模型(Gaussian Mixture Models)是一种无监督聚类模型。GMM认为不同类别的特征密度函数是不一样的(实际上也不一样)，GMM为每个类别下的特征分布都假设了一个服从高斯分布的概率密度函数。而数据中又可能是由多个类混合而成，所以数据中特征的概率密度函数可以使用多个高斯分布的组合来表示。

digits:
NMI: 0.469
Homogeneity: 0.462
Completeness: 0.476

Newsgroups:
NMI: 0.537
Homogeneity: 0.514
Completeness: 0.560

## 参考资料
[1].	https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
[2].	https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py


