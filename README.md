# Sliding-Window-Face-Detection-Based-on-HOG-features-and-SVM-Classifier
#### 1 实验目的

利用滑动窗口进行面部识别

http://cs.brown.edu/courses/cs143/2011/results/proj4/psastras/

#### 2 运行方法

1. 将data文件夹的三个zip文件解压，仍放在data文件夹中
2. 将test_scences的test.zip和test_jpg.zip解压，仍然放到test_scenes文件夹内
3. 将代码放到matlab中，修改project4.m中的data_path为data文件夹的路径
4. 运行project4.m文件即可



#### 3 实验原理
SUMMARY:
1. 分别用 Caltech Web Faces project 和 Sun Scene database 的数据集作为正负类 训练集，用 CMU+MIT 的数据集作为测试集 
2. 分别提取正负类样本的 Histogram of Oriented Gradients (HOG)特征，步骤包括, 1) 图像归一化和预处理, 2) 梯度计算, 3) 得到梯度统计直方图, 4) cell的合并与归一化 
3. 利用提取到的正负类HOG特征训练Support Vector Machine (SVM) 分类器，为提高准确率&降低假阳性率（fp_rate）将 hard negatives 的样本再次用于训练 SVM 
4. 在测试集上进行不同尺度下的人脸检测，并利用 non-maximum suppression (NMS)方法去除重叠框并选出最优大小的候选框，最终确定人脸位置，准确率为89.3% 

##### 3.1 引入数据集

训练集：利用`Caltech Web Faces project`做正类，`Sun Scene database` 做负类（主要是场景图片），将他们都裁减成大小一致。

测试集：`CMU+MIT test scenes`作测试集

定义变量：
`train_path_pos` ：正类路径，大小为36x36
`non_face_scn_path` ：负类路径
`test_scn_path`：测试集路径
`label_path`：标签集，是txt文本

##### 3.2 分别提取正负类训练集的 Hog 特征

 1. 定义两个参数：`template_size=36`, `hog_cell_size=3`
 > 注意：*template size should be evenly divisible by hog_cell_size*，因为把图片reshape到template_size大小，将图片分成多个cell，每个cell的边长是hog_cell_size，在每个cell里面统计一个梯度直方图，之后串联起来再归一化，得到的就是整个图片的hog描述符，所以需要divisible
 2. 获取正样本的特征
调用`get_positive_features`函数，传入`template_size=36`，`hog_cell_size=3`，`正类样本的路径`，得到特征`features_pos`（一个N \* D的矩阵，N是脸的数量，D是template的维度，也等于(template_size / hog_cell_size)^2 * 31，这个N*D是for循环里面，reshape再连接最后成这样的）

	`get_positive_features`函数：
	对于每个正类的image，调用`vl_hog`提取hog特征：
	```java
	HOG = vl_hog(single(image), feature_params.hog_cell_size)
	```
	每个imge的`HOG`连接成为`features_pos`

3. 获取负样本的特征：
调用`get_random_negative_features`函数，传入`负类路径`, `template_size=36`，`hog_cell_size=3`, `num_negative_examples`，得到`features_neg`（和上面正类的特征一样也是N*D的矩阵）

	`get_random_negative_features`函数：
		也是对每个image调用`HOG = vl_hog`
		每个image的HOG连接成`features_neg`
	
##### （3）将提取得到的训练集 Hog 特征用于训练 SVM 分类器
目的是得到w和b这两个可以完全定义一个svm的参数
1. svm的输入X,Y分别如下：
	```java
	//伪代码
	
	X=[正类特征矩阵features_pos，负类特征矩阵features_neg]
	//相当于把正负类的hog特征之间拼接
	
	Y=[（1,1,1,1….）;(-1,-1,-1,-1….)] 
	//“正类数量”个1的序列和“负类数量”个-1的序列的拼接
	```
2. 调用函数`[w,b] = vl_svmtrain(X', Y', lambda);`得到线性svm分类器的参数w和b

##### （4）计算该svm分类器的精度&画人脸模型
1. 计算准确率
将正负类样本刚刚算出来的特征带入svm的式子，如果分类器有效的话某个正类特征算出来的值应该是正的，负类特征算出来应该是负的。然后每个样本都带入算出一个结果，和答案向量（正类数量个1，负类数量个-1）对比，得到准确率。

2. 画出template
可以根据正脸特征的梯度直方图画出他学习到的人脸模型

##### （5）Mine hard negatives
目的：用难区分的负样本来强化一下svm

先找出hard negative（难以区分的负样本），也就是收集把负样本放进上一步训练的svm时，被以为是正样本的图片（这里只收集被认为是正样本的概率大于一定值的）。

把这些hard negative图片提取hog特征（调用`hog = vl_hog`）后，把（hn+原来的负类）的hog特征一起当成负类的hog特征再次训练svm，这样可以降低假阳性率 （fp_rate）从而提高准确性。 
 
##### （6）在测试集上运行检测器detector，找出人脸
准确率约为89.3%

调用`run_detector`函数，输入`测试集的路径` + `svm的w和b两个参数` + `template_size` + `hog_cell_size`，
返回`bboxes`, `confidences`, `image_ids`三个东西，分别是：

`Bboxes`：Nx4矩阵，N是detections的数量，4代表着[x_min, y_min, x_max, y_max]4个维度
`Confidences`： Nx1，某个detection的confidence
`image_ids`：Nx1，存文件名

对于每个测试集的图片img，把img分别缩小到0.8倍，0.8^2^倍，0.8^3^倍…...0.8^6^倍

在每个尺度上，调用`HOG = vl_hog`提取缩放后的图片（scaled_img）的hog特征，然后把结果带入到svm的公式里算出一个得分（得分如果是正的说明我的svm预测他是正类）

这里还用了一个`threshold`参数。不是>0就认为是正了，而是>threshold才认为是正类。

在正类的地方框出来，并记录一个confidence参数（由刚刚丢进svm算出来的得分，挑出大于threshold的部分再标准化得到的，用于衡量svm认为这个图片是人脸的概率）。

在不同尺度上都得到后，进行非最大抑制。即调用`non_max_supr_bbox`函数，输入每个尺度下的框框和置信度（confidence），得到/保留从不同尺度中挑出来是最大值的。

##### （7）探究参数的影响
`hog_cell_size`：计算hog特征时的cell的变长。随着 hog_cell_size 的增大，精度不断下降，因为更大的区域直接取了一个梯度直方图，更粗糙，但是计算量降低。

`threshold` ：判断为正类的阈值。随着 threshold 的增大（判断为正类的标准越严格），结果精度先减小后增大。 threshold 增大，精度先减小可能是因为将一些正类也判别成为负类了。 
详见project4-report.pdf文件



#### 4 运行效果

详见project4-report.pdf文件
