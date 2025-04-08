    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from collections import Counter
    import numpy as np

&emsp;&emsp;导入库

    class FlexibleFeatureExtractor:
    def __init__(self, feature_type='high_freq', top_k=100, **kwargs):
        """
        参数:
        - feature_type: 'high_freq' 高频词特征 或 'tfidf' TF-IDF加权特征
        - top_k: 选择前k个最重要的特征
        - kwargs: 传递给TfidfVectorizer或CountVectorizer的额外参数
        """
        self.feature_type = feature_type
        self.top_k = top_k
        self.vectorizer_kwargs = kwargs
        self.vectorizer = None
        self.feature_names = None

&emsp;&emsp;初始化特征提取器

    def fit(self, documents):
        """拟合特征提取器"""
        if self.feature_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(**self.vectorizer_kwargs)
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # 计算平均TF-IDF分数并排序
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            sorted_indices = np.argsort(avg_tfidf)[::-1]
            self.selected_indices = sorted_indices[:self.top_k]
            
        elif self.feature_type == 'high_freq':
            self.vectorizer = CountVectorizer(**self.vectorizer_kwargs)
            count_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # 计算词频总和并排序
            word_counts = np.asarray(count_matrix.sum(axis=0)).ravel()
            sorted_indices = np.argsort(word_counts)[::-1]
            self.selected_indices = sorted_indices[:self.top_k]
            
        else:
            raise ValueError("feature_type必须是'high_freq'或'tfidf'")
            
        # 更新特征名
        self.selected_features = [self.feature_names[i] for i in self.selected_indices]
        return self
    
&emsp;&emsp;根据选择的特征类型训练特征提取器<br>
&emsp;&emsp;对于TF-IDF模式，使用TfidfVectorizer计算TF-IDF矩阵，计算每个特征的平均TF-IDF值，按TF-IDF值降序排序，选择top_k特征<br>
&emsp;&emsp;对于高频词模式， 使用CountVectorizer计算词频矩阵，计算每个特征的总词频，按词频降序排序，选择top_k特征 ，更新类内部状态（selected_indices和selected_features）

    def transform(self, documents):
        """转换新文档为特征向量"""
        if self.vectorizer is None:
            raise RuntimeError("请先调用fit方法")
            
        matrix = self.vectorizer.transform(documents)
        
        if self.feature_type == 'tfidf':
            # 对于TF-IDF，直接返回选定的特征
            return matrix[:, self.selected_indices]
        else:
            # 对于高频词，返回词频计数
            return matrix[:, self.selected_indices]

&emsp;&emsp;检查是否已训练（fit） ，使用训练好的向量化器转换新文档 ，仅返回预先选定的特征（top_k个），实现将新文档转换为特征向量
  
    def get_feature_names(self):
        """获取选定的特征名称"""
        return self.selected_features

&emsp;&emsp;获取选定的特征名称列表 ，包含top_k个特征名的列表
![图片Alt]("D:\Pycharm-community\NLP\picture\a.png" "任务四第2项")