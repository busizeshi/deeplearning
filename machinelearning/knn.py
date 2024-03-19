# TODO K近邻方法训练

# 导包
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

# 导入数据
train_data = pd.read_csv("../data/house_price_forecasts/train.csv")
test_data = pd.read_csv("../data/house_price_forecasts/test.csv")
# 预览数据
print("训练数据")
print(train_data.head()._append(train_data.tail()))
print("测试数据")
print(test_data.head()._append(test_data.tail()))
print("训练数据具体信息")
print(train_data.info())
print("测试数据具体信息")
print(test_data.info())
print("训练数据描述信息")
print(train_data.describe())
print("训练数据的表头")
print(train_data.columns)
print("测试数据的表头")
print(test_data.columns)

# 数据预处理
all_features = pd.concat([train_data.iloc[:, 1:], test_data.iloc[:, 1:]])  # 去掉序号列，并合并训练数据和测试数据
# 处理文本特征和数字特征
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
# 处理NAN数据
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 转换为one-hot编码
all_features = pd.get_dummies(all_features, dummy_na=True)

# 转换为numpy斌进行数据划分
labels = all_features.SalePrice
print(labels)
num_of_train = train_data.shape[0]
train_features = all_features[: num_of_train].values
train_labels = labels[:num_of_train].values
test_features = all_features[num_of_train:].values
test_labels = labels[num_of_train:].values

# 进行交叉验证和网格搜索
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
kf = KFold(n_splits=5, shuffle=True)

grid = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf)

grid.fit(train_features, train_labels)

y_pred = grid.predict(test_features)
print(y_pred)
