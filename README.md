# 赛事：第四届魔镜杯大赛  
# 队伍名称：sandsand  
# 排名：53/1215  

## 一、	赛题描述  
### 1、任务描述：  
分别预测每个资产标的第一期从成交日期至第一期应还款日期每日的还款金额，并最终在整体上以资产组合每日还款的误差作为评价指标。  
### 2、数据集描述：  
数据集包括样本集（train和test）标的属性表（listing_info）,借款用户基础信息表（user_info）,用户画像标签列表（user_taglist）,借款用户操作行为日志表（user_behavior_logs）和用户还款日志表（user_repay_logs）。  
各数据集字段详情见：[数据集](https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=17&tabindex=1)  
### 3、评分标准：  
![评分标准](https://github.com/XunDong-Shi/ppd_mojing_4/blob/master/image/1.png)  

## 二、	总体思路  
赛题的评分标准与资产组合密切相关，通过对拍拍贷App和业务的摸索，猜测资产组合的方式为拍拍贷新彩虹中每个出借人投资的散标组合（见图1），散标数量大约在10到100个，但因资产组合为拍拍贷核心投资技术，故无法得知其具体方式。  
![推测的拍拍贷资产组合](https://github.com/XunDong-Shi/ppd_mojing_4/blob/master/image/2.png)  

因此，采用全数据集单个标的每月提前还款天数（0~31+逾期，总计33个分类）作为训练和测试的Label，以lightgbm为模型训练，用33分类的概率作为输出，并结合是否逾期的二分类模型对结果进行融合。  

## 三、	特征工程  
最终采用的特征包括以下几个部分：  
### 1、原本特征：  
（1）选取了数据集中due_amt、user_id、rate、principal、term、gender、age、cell_province、user_tag等原本特征；  
（2）删除了数据集中值域较广的id_city特征；  
### 2、新构造特征：  
（1)	借款日/还款截止日的年、月、日和星期几；  
（2)	本次借款与上一次借款的时间差；  
（3)	Repay_log的最大最小值、平均值和方差等；  
（4)	Listing_info中借款金额的平均值、方差，60天内借款次数；  
（5)	提前还款天数的平均值和方差；  
（6)	user_behavior的总次数、在深夜操作的次数，各行为次数、百分比；  
（7)	与同期同类借款标的的平均利率差值；  
（8)	是否id_province与cell_province是否为同省份的标签；  
（9)	用户年龄的分段；  
（10)	是否为新更新的用户标签；  
（11)	本月最长还款天数（如2019年2月为28天）  
（12)	采用word_to_voc对user_tag进行提取（max_features=500）  
### 3、特征选择：  
以20w的数据集采用lightgbm进行特征选择，但线上得分并未提升。  

## 四、	模型  
对类别特征进行编码转换、连续特征标准化后，以lightgbm为模型建模，并在比赛过程中尝试了class_weight参数调试、自定义损失函数、添加二分类规则和简单stacking等方法，但最终结果均不是太理想。  

## 五、	小结  
### 1、收获：  
□ 通过本次比赛对互联网金融的业务有了基本的了解；  
□ 代码基础得到了进一步提高：  
### 2、反思：  
□ 没有主动构建团队，个人精力不足；  
□ 没有进行充分的数据分析  
□ 使用全数据进行模型训练，模型训练时间长、提交次数少；  
□ 浪费时间较多，代码版本管理混乱；  
□ 线上测试的实验思路不好，难得出优化方向；  
### 3、其他参赛队伍解决方案	  
□ 采用部分相近数据集、加入宏观日期模型（[冠军方案分享](https://zhuanlan.zhihu.com/p/75199206?utm_source=com.tencent.tim&utm_medium=social&utm_oi=555381879923224576)）  
□ 未来一个月需还款标的数量等特征、文本主题分布（[亚军方案分享](https://zhuanlan.zhihu.com/p/74749772)）  
□ 多应用规则和二分类模型（[季军方案分享](https://zhuanlan.zhihu.com/p/75234282?utm_source=com.tencent.tim&utm_medium=social&utm_oi=555381879923224576)）  
