import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xcorr

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedGroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')

#====================加载数据=========================
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('训练集大小：',train.shape)
print('测试集大小：',test.shape)
print('\n训练集前几行：')
print(train.head())

#====================快速了解数据======================
print('缺失值情况:\n',train.isnull().sum())

print('数值特征分布:\n',train.describe())

#分类特征的值分布
print(train['Sex'].value_counts())
print(train['Embarked'].value_counts())

#=====================特征工程========================
#目的：把原始表格变成一张全是数字、没有缺失、训练集和测试集列数完全相同的表格
train_clean = train.copy()
test_clean = test.copy()

#1.放弃无用或缺失太多的列
drop_cols = ['PassengerId', 'Ticket', 'Cabin', 'Name']
train_clean.drop(columns=drop_cols, inplace=True, errors='ignore')
test_clean.drop(columns=[c for c in drop_cols if c in test_clean.columns], inplace=True, errors='ignore')

#2.处理缺失值
#age:中位数填充
median_age = train_clean['Age'].median()
train_clean['Age'].fillna(median_age, inplace=True)
test_clean['Age'].fillna(median_age, inplace=True)
#Embarked:2个缺失众数填充
train_clean['Embarked'].fillna(train_clean['Embarked'].mode()[0], inplace=True)
test_clean['Fare'].fillna(test_clean['Fare'].median(), inplace=True)

#3.创造新特征：家庭人数
train_clean['FamilySize'] = train_clean['SibSp'] + train_clean['Parch'] + 1
test_clean['FamilySize'] = test_clean['SibSp'] + test_clean['Parch'] + 1

#4.把性别转为数值
train_clean['Sex'] = train_clean['Sex'].map({'male': 0, 'female': 1})
test_clean['Sex'] = test_clean['Sex'].map({'male': 0, 'female': 1})

#5.创建舱位与性别的交互特征
train_clean['Pclass_Sex'] = train_clean['Pclass'].astype(str) + '_' + train_clean['Sex'].astype(str)
#train_clean = pd.get_dummies(train_clean, columns=['Pclass_Sex'], prefix='Pclass')
test_clean['Pclass_Sex'] = test_clean['Pclass'].astype(str) + '_' + test_clean['Sex'].astype(str)
#test_clean = pd.get_dummies(test_clean, columns=['Pclass_Sex'], prefix='Pclass')

# 定义年龄分桶函数
def age_bucket(age):
    if age <= 12:
        return 'Child'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# 为训练集和测试集添加 AgeBucket 列
train_clean['AgeBucket'] = train_clean['Age'].apply(age_bucket)
test_clean['AgeBucket'] = test_clean['Age'].apply(age_bucket)

# 创建 AgeBucket × Sex 交互特征（Sex 已经是 0/1）
train_clean['AgeSex'] = train_clean['AgeBucket'] + '_' + train_clean['Sex'].astype(str)
test_clean['AgeSex'] = test_clean['AgeBucket'] + '_' + test_clean['Sex'].astype(str)

# 6. 将所有分类特征（Embarked, Pclass_Sex, AgeSex）统一进行 One-Hot 编码
categorical_features = ['Embarked', 'Pclass_Sex', 'AgeSex']
train_clean = pd.get_dummies(train_clean, columns=categorical_features,
                             prefix=['Emb', 'PclassSex', 'AgeSex'])
test_clean = pd.get_dummies(test_clean, columns=categorical_features,
                            prefix=['Emb', 'PclassSex', 'AgeSex'])

# 7.删除已经编码的原始字符串列（这些列现在已经是数值 dummy 列了，原始列可以删除）
cols_to_drop = ['Embarked', 'Pclass_Sex', 'AgeSex', 'AgeBucket']
train_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')


X_train = train_clean.drop('Survived', axis=1)
y_train = train_clean['Survived']

#test缺少的列补0
for col in  X_train.columns:
    if col not in test_clean.columns:
        test_clean[col] = 0

X_test = test_clean[X_train.columns]

#======================训练模型+交叉验证========================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN':KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# ======================强制清洗 NaN==========================
print("X_train 原始NaN数量：", X_train.isnull().sum().sum())

# 1. 对 Sex 列进行二次清洗（防止映射失败）
print("Sex 列唯一值：", X_train['Sex'].unique())
X_train['Sex'] = X_train['Sex'].fillna(0)   # 如果还有 NaN，填 0（男性）

# 2. 对其他数值列用中位数填充
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_fixed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

print("清洗后 X_train 还有 NaN？", X_train_fixed.isnull().any().any())

# 3. 再用清洗后的数据训练
for name, model in models.items():
    scores = cross_val_score(model, X_train_fixed, y_train, cv=cv, scoring='accuracy')
    print(f'{name:20} | 平均准确率：{scores.mean():.4f} (±{scores.std():.4f})')


#========================误差分析==========================
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
#拆分验证集做误差分析
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_fixed, y_train, test_size=0.2, random_state=42, stratify=y_train)

final_model.fit(X_train_part, y_train_part)
y_pred = final_model.predict(X_val)

#找出预测错误样本
error_idx = (y_val != y_pred)
error_df = X_val[error_idx].copy()
error_df['True_survived'] = y_val[error_idx].values
error_df['Predicted'] = y_pred[error_idx]


print(f"\n验证集总数: {len(X_val)}")
print(f"错误样本数: {len(error_df)}")
print(f"错误率: {len(error_df)/len(X_val):.2%}")

print("\n错误样本中性别分布:")
print(error_df['Sex'].value_counts())

print("\n错误样本中舱位等级分布:")
print(error_df['Pclass'].value_counts())

# 对比正确样本和错误样本的平均年龄
correct_idx = (y_val == y_pred)
print(f"\n错误样本平均年龄: {error_df['Age'].mean():.2f}")
print(f"正确样本平均年龄: {X_val[correct_idx]['Age'].mean():.2f}")

#==========================最终模型========================
#转化测试集
X_test_fixed = pd.DataFrame(imputer.transform(X_test), columns=X_train_fixed.columns)
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X_train_fixed, y_train)

test_pred = final_model.predict(X_test_fixed)

submission = pd.DataFrame({
            'PassengerId':test['PassengerId'],
            'Survived': test_pred})
submission.to_csv('submission.csv', index=False)
print('提交文件已生成，共{}行'.format(len(submission)))