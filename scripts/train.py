"""
模型训练脚本
使用 XGBoost 训练获奖等级预测模型
"""

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, ensure_dir, save_pickle, load_pickle


class MCMClassifier:
    """MCM/ICM 获奖等级分类器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化分类器"""
        self.config = load_config(config_path)
        self.classifier_config = self.config['classifier']
        self.xgb_params = self.classifier_config['xgboost']
        self.training_config = self.classifier_config['training']
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self, processed_dir: str = "data/processed"):
        """加载处理后的数据"""
        print("加载数据...")
        
        features = np.load(Path(processed_dir) / "features.npy")
        labels = np.load(Path(processed_dir) / "labels.npy", allow_pickle=True)
        
        # 加载特征名称（用于特征重要性分析）
        try:
            self.feature_names = load_pickle(str(Path(processed_dir) / "feature_names.pkl"))
        except:
            self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        print(f"  特征矩阵: {features.shape}")
        print(f"  标签数量: {len(labels)}")
        
        # 检查并处理异常值
        features = self._handle_invalid_values(features)
        
        return features, labels
    
    def _handle_invalid_values(self, features: np.ndarray) -> np.ndarray:
        """处理 NaN 和 Inf 值"""
        # 将 NaN 替换为 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def preprocess_data(self, X, y):
        """数据预处理"""
        print("\n数据预处理...")
        
        # 1. 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"  标签编码: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # 2. 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. 分割训练集和测试集
        test_size = self.training_config['test_size']
        random_state = self.training_config['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"  训练集: {X_train.shape[0]} 样本")
        print(f"  测试集: {X_test.shape[0]} 样本")
        
        # 4. SMOTE 过采样（可选，处理类别不平衡）
        if self.training_config['use_smote']:
            print("\n应用 SMOTE 过采样...")
            try:
                smote = SMOTE(random_state=random_state)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"  过采样后训练集: {X_train.shape[0]} 样本")
                
                # 显示平衡后的标签分布
                unique, counts = np.unique(y_train, return_counts=True)
                for label_idx, count in zip(unique, counts):
                    label_name = self.label_encoder.inverse_transform([label_idx])[0]
                    print(f"    {label_name}: {count}")
            except Exception as e:
                print(f"  SMOTE 失败 ({e})，跳过过采样")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """训练模型"""
        print("\n训练 XGBoost 模型...")
        
        # 设置参数
        params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': self.xgb_params['max_depth'],
            'learning_rate': self.xgb_params['learning_rate'],
            'n_estimators': self.xgb_params['n_estimators'],
            'subsample': self.xgb_params['subsample'],
            'colsample_bytree': self.xgb_params['colsample_bytree'],
            'min_child_weight': self.xgb_params['min_child_weight'],
            'reg_alpha': self.xgb_params['reg_alpha'],
            'reg_lambda': self.xgb_params['reg_lambda'],
            'eval_metric': self.xgb_params['eval_metric'],
            'random_state': self.training_config['random_state'],
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        print(f"  参数: {params}")
        
        # 创建并训练模型
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            verbose=True
        )
        
        print("训练完成！")
    
    def cross_validate(self, X, y):
        """交叉验证"""
        print("\n执行交叉验证...")
        
        cv_folds = self.training_config['cv_folds']
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.training_config['random_state'])
        
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        
        print(f"  {cv_folds}-折交叉验证准确率:")
        for i, score in enumerate(scores):
            print(f"    Fold {i+1}: {score:.4f}")
        print(f"  平均准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\n评估模型...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  测试集准确率: {accuracy:.4f}")
        
        # 分类报告
        print("\n分类报告:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)
        
        return accuracy, y_pred, y_pred_proba
    
    def _plot_confusion_matrix(self, cm, target_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_path = Path(self.config['data']['processed_dir']) / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n混淆矩阵已保存: {output_path}")
        plt.close()
    
    def analyze_feature_importance(self, top_n=20):
        """分析特征重要性"""
        print(f"\nTop {top_n} 重要特征:")
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 排序
        indices = np.argsort(importances)[::-1][:top_n]
        
        # 显示
        for i, idx in enumerate(indices):
            feature_name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
            print(f"  {i+1}. {feature_name}: {importances[idx]:.4f}")
        
        # 绘图
        plt.figure(figsize=(10, 6))
        top_features = [self.feature_names[i] if self.feature_names else f"F{i}" for i in indices]
        top_importances = importances[indices]
        
        plt.barh(range(top_n), top_importances)
        plt.yticks(range(top_n), top_features, fontsize=8)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        output_path = Path(self.config['data']['processed_dir']) / "feature_importance.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"特征重要性图已保存: {output_path}")
        plt.close()
    
    def save_model(self, model_dir: str = None):
        """保存模型"""
        if model_dir is None:
            model_dir = self.config['data']['models_dir']
        
        ensure_dir(model_dir)
        
        # 保存 XGBoost 模型
        model_path = Path(model_dir) / "xgb_model.json"
        self.model.save_model(str(model_path))
        print(f"\n模型已保存: {model_path}")
        
        # 保存标签编码器和标准化器
        save_pickle(self.label_encoder, str(Path(model_dir) / "label_encoder.pkl"))
        save_pickle(self.scaler, str(Path(model_dir) / "scaler.pkl"))
        
        # 保存特征名称
        if self.feature_names:
            save_pickle(self.feature_names, str(Path(model_dir) / "feature_names.pkl"))
        
        print("辅助文件已保存")
    
    def load_model(self, model_dir: str = None):
        """加载模型"""
        if model_dir is None:
            model_dir = self.config['data']['models_dir']
        
        # 加载 XGBoost 模型
        model_path = Path(model_dir) / "xgb_model.json"
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        
        # 加载辅助文件
        self.label_encoder = load_pickle(str(Path(model_dir) / "label_encoder.pkl"))
        self.scaler = load_pickle(str(Path(model_dir) / "scaler.pkl"))
        self.feature_names = load_pickle(str(Path(model_dir) / "feature_names.pkl"))
        
        print(f"模型已加载: {model_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCM/ICM 模型训练")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据目录")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MCM/ICM 获奖等级预测 - 模型训练")
    print("="*60)
    
    # 初始化分类器
    classifier = MCMClassifier(args.config)
    
    # 加载数据
    X, y = classifier.load_data(args.data_dir)
    
    # 预处理
    X_train, X_test, y_train, y_test = classifier.preprocess_data(X, y)
    
    # 训练
    classifier.train(X_train, y_train)
    
    # 交叉验证
    X_scaled = classifier.scaler.transform(X)
    y_encoded = classifier.label_encoder.transform(y)
    classifier.cross_validate(X_scaled, y_encoded)
    
    # 评估
    accuracy, y_pred, y_pred_proba = classifier.evaluate(X_test, y_test)
    
    # 特征重要性分析
    classifier.analyze_feature_importance(top_n=20)
    
    # 保存模型
    classifier.save_model()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最终测试准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()
