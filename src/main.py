import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
# from transformers import BertTokenizer, BertModel
import jieba
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

class TextClassificationExperiment:
    def __init__(self, data_path, stopwords_path):
        self.data_path = data_path
        self.stopwords_path = stopwords_path
        self.results = []
        
    def load_data(self):
        """加载数据"""
        self.data = pd.read_excel(self.data_path, engine='openpyxl')
        return self.data
    
    def load_stopwords(self):
        """加载停用词"""
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
        stopwords.update(['\n', '（', '）', ' '])
        return stopwords
    
    def preprocess_text(self):
        """文本预处理"""
        stopwords = self.load_stopwords()
        
        def segment_text(text):
            words = jieba.cut(str(text))
            filtered_words = [w for w in words if w.strip() and w not in stopwords]
            return ' '.join(filtered_words)
        
        self.data['Cut_Text'] = self.data['Text'].apply(segment_text)
        return self.data
    
    def tfidf_vectorize(self, texts, max_features=5000):
        """TF-IDF向量化"""
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectors = vectorizer.fit_transform(texts)
        return vectors, vectorizer
    
    def word2vec_vectorize(self, texts, vector_size=100):
        """Word2Vec向量化"""
        tokenized_texts = [text.split() for text in texts]
        model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, min_count=1, workers=4)
        
        def get_doc_vector(text):
            words = text.split()
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
        
        vectors = np.array([get_doc_vector(text) for text in texts])
        return vectors, model
    
    def bert_vectorize(self, texts, model_name='bert-base-chinese'):
        """BERT向量化"""
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()
        
        vectors = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=512)
                outputs = model(**inputs)
                cls_vector = outputs.last_hidden_state[0][0].numpy()
                vectors.append(cls_vector)
        
        return np.array(vectors), model
    
    def train_classifier(self, classifier, X_train, X_test, y_train, y_test, vec_name, clf_name):
        """训练分类器并返回结果"""
        # 训练模型
        classifier.fit(X_train, y_train)
        
        # 预测
        y_pred = classifier.predict(X_test)
        
        # 计算准确率
        accuracy = classifier.score(X_test, y_test)
        
        # sklearn的classification report（会自动打印）
        print(f"\n{vec_name.upper()} + {clf_name.upper()} 分类报告:")
        print("=" * 50)
        print(classification_report(y_test, y_pred))
        
        # 存储结果
        result = {
            'vectorizer': vec_name,
            'classifier': clf_name,
            'accuracy': accuracy,
            'model': classifier,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        self.results.append(result)
        return result
    
    def plot_results(self):
        """可视化结果"""
        # 创建结果DataFrame
        results_df = pd.DataFrame([{
            'Vectorizer': r['vectorizer'],
            'Classifier': r['classifier'],
            'Accuracy': r['accuracy']
        } for r in self.results])
        
        # 设置绘图风格
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('文本分类实验结果可视化', fontsize=16, fontweight='bold')
        
        # 1. 准确率热力图
        pivot_table = results_df.pivot_table(values='Accuracy', 
                                           index='Vectorizer', 
                                           columns='Classifier')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Accuracy'})
        axes[0, 0].set_title('不同组合准确率热力图')
        
        # 2. 准确率柱状图
        sns.barplot(data=results_df, x='Vectorizer', y='Accuracy', hue='Classifier',
                   ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('不同词向量方法的准确率比较')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 分类器性能比较
        classifier_avg = results_df.groupby('Classifier')['Accuracy'].mean().sort_values()
        axes[1, 0].barh(range(len(classifier_avg)), classifier_avg.values, color='skyblue')
        axes[1, 0].set_yticks(range(len(classifier_avg)))
        axes[1, 0].set_yticklabels(classifier_avg.index)
        axes[1, 0].set_xlabel('平均准确率')
        axes[1, 0].set_title('分类器性能排名')
        
        # 4. 词向量方法比较
        vectorizer_avg = results_df.groupby('Vectorizer')['Accuracy'].mean().sort_values()
        axes[1, 1].bar(range(len(vectorizer_avg)), vectorizer_avg.values, color='lightgreen')
        axes[1, 1].set_xticks(range(len(vectorizer_avg)))
        axes[1, 1].set_xticklabels(vectorizer_avg.index, rotation=45)
        axes[1, 1].set_ylabel('平均准确率')
        axes[1, 1].set_title('词向量方法性能比较')
        
        plt.tight_layout()
        plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. 混淆矩阵可视化
        best_result = max(self.results, key=lambda x: x['accuracy'])
        self.plot_confusion_matrix(best_result['y_true'], best_result['y_pred'], 
                                 best_result['vectorizer'], best_result['classifier'])
    
    def plot_confusion_matrix(self, y_true, y_pred, vec_name, clf_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix: {vec_name.upper()} + {clf_name.upper()}\n'
                 f'Accuracy: {np.mean(y_true == y_pred):.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """运行完整实验"""
        # 1. 加载和预处理数据
        self.load_data()
        self.preprocess_text()
        
        print("数据基本信息:")
        print(f"数据形状: {self.data.shape}")
        print(f"标签分布:\n{self.data['Label'].value_counts()}")
        
        # 2. 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['Cut_Text'], self.data['Label'], 
            test_size=0.2, random_state=42, stratify=self.data['Label']
        )
        
        # 获取原始文本用于BERT
        train_indices = X_train.index
        test_indices = X_test.index
        original_train_texts = self.data.loc[train_indices, 'Text'].tolist()
        original_test_texts = self.data.loc[test_indices, 'Text'].tolist()
        
        # 3. 定义实验组合
        vectorizers = {
            'tfidf': (X_train.tolist(), X_test.tolist()),
            'word2vec': (X_train.tolist(), X_test.tolist()),
            'bert': (original_train_texts, original_test_texts)
        }
        
        classifiers = {
            'svm': SVC(C=1.0, kernel='linear', random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # 4. 执行实验
        for vec_name, (train_texts, test_texts) in vectorizers.items():
            print(f"\n{'='*60}")
            print(f"使用 {vec_name.upper()} 进行向量化:")
            print('='*60)
            
            # 向量化
            if vec_name == 'tfidf':
                X_train_vec, vec_model = self.tfidf_vectorize(train_texts)
                X_test_vec = vec_model.transform(test_texts)
            elif vec_name == 'word2vec':
                X_train_vec, vec_model = self.word2vec_vectorize(train_texts)
                X_test_vec = np.array([self.get_doc_vector_word2vec(vec_model, text) 
                                     for text in test_texts])
            else:  # bert
                X_train_vec, vec_model = self.bert_vectorize(train_texts)
                X_test_vec, _ = self.bert_vectorize(test_texts)
            
            # 训练分类器
            for clf_name, classifier in classifiers.items():
                print(f"\n训练分类器: {clf_name.upper()}")
                self.train_classifier(classifier, X_train_vec, X_test_vec, 
                                    y_train, y_test, vec_name, clf_name)
        
        # 5. 可视化结果
        self.plot_results()
        
        # 6. 保存详细结果
        self.save_results()
    
    def get_doc_vector_word2vec(self, model, text):
        """Word2Vec文档向量生成"""
        words = text.split()
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    def save_results(self):
        """保存结果到文件"""
        results_df = pd.DataFrame([{
            'Vectorizer': r['vectorizer'],
            'Classifier': r['classifier'],
            'Accuracy': r['accuracy']
        } for r in self.results])
        
        results_df.to_csv('experiment_results.csv', index=False, encoding='utf-8-sig')
        
        # 保存详细结果
        with open('detailed_results.txt', 'w', encoding='utf-8') as f:
            f.write("文本分类实验详细结果\n")
            f.write("="*50 + "\n")
            for result in self.results:
                f.write(f"\n{result['vectorizer'].upper()} + {result['classifier'].upper()}\n")
                f.write(f"准确率: {result['accuracy']:.4f}\n")
                f.write("-"*30 + "\n")

def main():
    """主函数"""
    # 初始化实验
    experiment = TextClassificationExperiment(
        data_path='./data/gastric.xlsx',
        stopwords_path='./data/stop_words.txt'
    )
    
    # 运行实验
    experiment.run_experiment()

if __name__ == "__main__":
    main()