import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持 (Set Chinese font support)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题


# ============================ 1. 数据加载 ============================
print("正在加载数据...")
try:
    # First try UTF-8
    df = pd.read_csv('特征_建模前预处理.csv', encoding='utf-8')
    print("使用UTF-8编码成功")
except UnicodeDecodeError:
    try:
        # Then try GBK (Common for Chinese Windows)
        df = pd.read_csv('特征_建模前预处理.csv', encoding='gbk')
        print("使用GBK编码成功")
    except UnicodeDecodeError:
        # Finally try GB2312
        df = pd.read_csv('特征_建模前预处理.csv', encoding='gb2312')
        print("使用GB2312编码成功")

# ============================ 2. 特征选择 ============================
# 定义用于训练的特征列与目标变量
features = ['融资频率_次每年', '实缴资本完成率_%', '投资方质量评分', '股东数量',
            '对数注册资本', '近一年案件数', '被告次数占比(%)', '高科技标签数',
            '政府支持标签数', 'cluster']
target = 'is_potential_enterprise'

X = df[features]
y = df[target]

# 输出数据基本信息
print(f"数据形状: {X.shape}")
print(f"目标变量分布:\n{y.value_counts()}")
print(f"正样本比例: {y.mean():.3f}")

# ============================ 3. 数据划分 ============================
print("\n正在进行数据分割 (7:3)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)# stratify保持正负样本比例一致

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"训练集正样本比例: {y_train.mean():.3f}")
print(f"测试集正样本比例: {y_test.mean():.3f}")

# ============================ 4. 数据标准化 ============================
# 仅对需要标准化的模型（逻辑回归、SVM）使用
print("\n正在进行数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================ 5. 模型定义与调优 ============================
print("\n=== 模型参数调优 ===")

# 定义多个候选模型
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# 定义参数搜索网格（可自行扩展范围）
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1]
    }
}

best_estimators = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 遍历模型逐一进行GridSearchCV调优
for name in models.keys():
    print(f"\n正在调优 {name}...")
    model = models[name]
    param_grid = param_grids[name]
    
    # 标准化数据仅用于LR与SVM
    X_train_data = X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train

    # Use F1 score for optimization as it balances precision and recall
    grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_data, y_train)
    
    best_estimators[name] = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# ============================ 6. 模型比较 ============================
print("\n=== 调优后模型在测试集上的表现 ===")
results = {}
for name, model in best_estimators.items():
    # Prepare test data based on model type
    X_test_data = X_test_scaled if name in ['Logistic Regression', 'SVM'] else X_test
    
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    print(f"{name} - F1: {results[name]['f1']:.4f}, AUC: {results[name]['auc']:.4f}, Precision: {results[name]['precision']:.4f}, Recall: {results[name]['recall']:.4f}")

# ============================ 7. 选择最佳模型 ============================
best_model_name = max(results.keys(), key=lambda name: results[name]['f1'])
best_model = best_estimators[best_model_name]
print(f"\n最佳模型: {best_model_name} (测试集 F1: {results[best_model_name]['f1']:.4f})")

# ============================ 8. 最佳模型详细评估 ============================
print(f"\n=== {best_model_name} 详细评估结果 ===")
y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

print("\n分类报告:")
print(classification_report(y_test, y_pred_best))

# 可视化混淆矩阵 (Visualize Confusion Matrix)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['非潜力企业', '潜力企业'],
            yticklabels=['非潜力企业', '潜力企业'])
plt.title(f'{best_model_name} - 调优后混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('混淆矩阵_最佳模型.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC曲线 (ROC Curve)
plt.figure(figsize=(10, 8))
for name in results:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('各模型调优后ROC曲线对比')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('ROC曲线_调优后对比.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================ 9. 特征重要性分析 ============================
if hasattr(best_model, 'feature_importances_'):
    print("\n特征重要性排序:")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'{best_model_name} - 特征重要性')
    plt.tight_layout()
    plt.savefig('特征重要性_最佳模型.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================ 10. 多维度交叉验证 ============================
print("\n=== 多维度交叉验证 (基于调优后的模型) ===")
scoring_metrics = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
cv_results_tuned = {}

for name, model in best_estimators.items():
    print(f"\n对 {name} 进行5折交叉验证...")
    X_cv_data = X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train
    
    cv_scores = {}
    for metric_name, metric in scoring_metrics.items():
        scores = cross_val_score(model, X_cv_data, y_train, cv=cv_strategy, scoring=metric)
        cv_scores[metric_name] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"  {metric_name}: {scores.mean():.4f} (±{scores.std():.4f})")
    cv_results_tuned[name] = cv_scores

# 交叉验证结果可视化 (Visualize Cross-Validation Results)
metrics_to_plot = list(scoring_metrics.keys())
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()
fig.suptitle('各模型5折交叉验证性能对比', fontsize=16)

for i, metric in enumerate(metrics_to_plot):
    model_names = list(cv_results_tuned.keys())
    means = [cv_results_tuned[name][metric]['mean'] for name in model_names]
    stds = [cv_results_tuned[name][metric]['std'] for name in model_names]
    
    bars = axes[i].bar(model_names, means, yerr=stds, capsize=5, alpha=0.8)
    axes[i].set_title(f'{metric.capitalize()} 交叉验证结果')
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylim(bottom=max(0, min(means)-max(stds)-0.1), top=min(1.0, max(means)+max(stds)+0.1))

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., height, f'{mean:.3f}', ha='center', va='bottom')

for i in range(len(metrics_to_plot), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('交叉验证结果_调优后.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================ 11. 学习曲线分析 ============================
print(f"\n=== {best_model_name} 学习曲线分析 ===")
X_train_best = X_train_scaled if best_model_name in ['Logistic Regression', 'SVM'] else X_train
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_best, y_train, cv=cv_strategy,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1', n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='训练得分')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='交叉验证得分')
plt.xlabel('训练样本数量')
plt.ylabel('F1 分数')
plt.title(f'{best_model_name} 学习曲线')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('学习曲线.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================ 12. 最终结果总结 ============================
print("\n=== 最终模型总结 ===")
final_results = results[best_model_name]
print(f"最佳模型: {best_model_name}")
print(f"测试集准确率: {final_results['accuracy']:.4f}")
print(f"测试集精确率: {final_results['precision']:.4f}")
print(f"测试集召回率: {final_results['recall']:.4f}")
print(f"测试集F1分数: {final_results['f1']:.4f}")
print(f"测试集AUC: {final_results['auc']:.4f}")
print(f"使用特征: {len(features)}个")
print(f"训练样本数: {X_train.shape[0]}")
print(f"测试样本数: {X_test.shape[0]}")

# ============================ 13. 预测新数据函数 ============================
def predict_new_data(model, scaler, features, new_data, model_name):
    """
    输入新数据并输出预测结果
    参数:
        model: 训练好的模型
        scaler: 标准化器
        features: 特征列名
        new_data: 新样本 (DataFrame或字典)
        model_name: 模型名称 (用于判断是否标准化)
    返回:
        predictions: 类别预测结果
        probabilities: 属于正类的概率
    """

    if isinstance(new_data, pd.DataFrame):
        new_data_df = new_data[features]
    else:
        new_data_df = pd.DataFrame(new_data, columns=features)
    
    # 根据模型类型决定是否标准化 (Scale data based on model type)
    if model_name in ['Logistic Regression', 'SVM']:
        new_data_scaled = scaler.transform(new_data_df)
        predictions = model.predict(new_data_scaled)
        probabilities = model.predict_proba(new_data_scaled)[:, 1]
    else:
        predictions = model.predict(new_data_df)
        probabilities = model.predict_proba(new_data_df)[:, 1]
    
    return predictions, probabilities

# 示例：预测前5个测试样本 (Example: Predict first 5 test samples)
print("\n测试集前5个样本的预测结果:")
sample_predictions, sample_probabilities = predict_new_data(
    best_model, scaler, features, X_test.iloc[:5], best_model_name
)

for i, (pred, prob, actual) in enumerate(zip(sample_predictions, sample_probabilities, y_test.iloc[:5])):
    status = "✓" if pred == actual else "✗"
    print(f"样本 {i+1}: 预测={pred}, 概率={prob:.3f}, 实际={actual} {status}")

