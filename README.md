# Sliding-Window-Face-Detection-Based-on-HOG-features-and-SVM-Classifier
##### 1 实验目的

利用滑动窗口进行面部识别

http://cs.brown.edu/courses/cs143/2011/results/proj4/psastras/

##### 2 运行方法

下载data文件夹的数据集

将代码放到matlab中，修改project4.m中的data_path为data文件夹的路径

运行project4.m文件即可



##### 3 实验原理

首先分别提取正负类训练集的Hog特征，将提取得到的训练集Hog特征用于训练SVM分类器。再提取测试集的Hog特征，按照step滑动窗口，用刚刚训练得到的SVM分类器筛出人脸窗口。最后利用NMS移除重叠候选框。

详见project4-report.pdf文件



##### 4 运行效果

详见project4-report.pdf文件
