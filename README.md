# Flare template
因为在学ml过程中懒得重复完成实现算法以及分割数据集等这些琐碎工作，决定随便弄个模板以节省时间.

## Basic idea
Steps: allocate -> train -> test

 
```
allocate: 
	给出train_set和test_set, 包括留出法，自助法，交叉验证法...

train(classification only for now):
	binary real: 
		Linear Regression, Logistic Regression, LDA...
	binary categorical:
		Decision Tree...
	multi-class wrapper:
		OvO, OvR, MvM...
	
test:
	测试指标设定，时间分析...  
```
