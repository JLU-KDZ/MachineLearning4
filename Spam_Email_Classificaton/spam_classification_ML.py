import numpy as np
import pandas as pd
# import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# nltk.download('stopwords')
PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


# 数据预处理
def text_processing(text):
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# 读邮件数据CSV
train_email = pd.read_csv("data/train.csv", usecols=[2], encoding='utf-8')
train_label = pd.read_csv("data/train.csv", usecols=[1], encoding='utf-8')

train_email['Email'] = train_email['Email'].apply(text_processing)

# 将内容转为list类型
train_email = np.array(train_email).reshape((1, len(train_email)))[0].tolist()
train_label = np.array(train_label).reshape((1, len(train_email)))[0].tolist()

# 构造训练集和验证集
train_num = int(len(train_email) * 0.8)
data_train = train_email[:train_num]
data_dev = train_email[train_num:]
label_train = train_label[:train_num]
label_dev = train_label[train_num:]

# 标签编码
label_encoder = LabelEncoder()
label_train_encoded = label_encoder.fit_transform(label_train)
label_dev_encoded = label_encoder.transform(label_dev)

# 使用词袋模型
vectorizer = CountVectorizer()
data_train_cnt = vectorizer.fit_transform(data_train)
data_test_cnt = vectorizer.transform(data_dev)

# 变成TF-IDF矩阵
transformer = TfidfTransformer()
data_train_tfidf = transformer.fit_transform(data_train_cnt)
data_test_tfidf = transformer.transform(data_test_cnt)

# 利用逻辑回归的方法
lr_crf = LogisticRegression(max_iter=150, penalty='l2', solver='lbfgs', random_state=0)
lr_crf.fit(data_train_tfidf, label_train_encoded)
score = lr_crf.score(data_test_tfidf, label_dev_encoded)
print("LR score: ", score)

# 预测概率
prob_lr = lr_crf.predict_proba(data_test_tfidf)

# 计算ROC曲线
fpr_lr, tpr_lr, _ = roc_curve(label_dev_encoded, prob_lr[:, 1])

# 计算AUC值
roc_auc_lr = auc(fpr_lr, tpr_lr)
print("LR AUC:", roc_auc_lr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# 预测结果
result_lr = lr_crf.predict(data_test_tfidf)
print("LR confusion: ", confusion_matrix(label_dev_encoded, result_lr))
