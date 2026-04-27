import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# ตั้งค่าฟอนต์และสไตล์กราฟให้ดูมีความเป็น Professional/Technical
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={"font.family": "Tahoma"})

# 1. โหลดข้อมูลที่เตรียมพร้อมเข้า Model
df_train = pd.read_csv('Model_Ready_Train_Data_Selected.csv')

# สมมติว่า Target ของเราคือ 'Purchase_Target_Encoded' (ดูจากชื่อคอลัมน์ในไฟล์)
target_col = 'Purchase_Target_Encoded'
X = df_train.drop(columns=[target_col])
y = df_train[target_col]

# สร้าง Model เพื่อหา Feature Importance ไว้ใช้สำหรับกราฟ
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
top_features = feature_imp_df['Feature'].head(5).tolist()

# ==========================================
# กราฟที่ 1: Scaled Feature Distributions (KDE Plot)
# ==========================================
plt.figure(figsize=(10, 6))
for feature in top_features[:3]:  # เอาแค่ Top 3 มาดูการกระจายตัว
    sns.kdeplot(df_train[feature], fill=True, label=feature, alpha=0.5)
plt.title('Distribution of Top 3 Scaled Features', fontsize=14, fontweight='bold')
plt.xlabel('Standardized Value (Z-Score)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 2: Multicollinearity Check (Correlation Heatmap ของ Top 10 Features)
# ==========================================
top_10_features = feature_imp_df['Feature'].head(10).tolist()
corr_matrix = df_train[top_10_features].corr()

plt.figure(figsize=(10, 8))
# ใช้ mask เพื่อซ่อนครึ่งบนของ Heatmap ให้ดูสะอาดตาแบบฉบับเปเปอร์วิชาการ
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('Feature Correlation Matrix (Top 10 Features)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 3: PCA 2D Projection (Dimensionality Reduction)
# ==========================================
# ลดมิติข้อมูลจากหลายสิบ Features ให้เหลือแค่ 2 แกน เพื่อดูความสามารถในการแยกกลุ่ม (Separability)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Target'] = y.values

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', 
                hue='Target', data=pca_df, palette='Set1', alpha=0.7)
plt.title(f'PCA 2D Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)', 
          fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='Target')
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 4: Violin Plot (Feature Separability per Class)
# ==========================================
# ดูกระจายตัวของฟีเจอร์ที่สำคัญที่สุดอันดับ 1 ว่ามันแยก Target 0 กับ 1 ออกจากกันได้ดีแค่ไหน
best_feature = top_features[0]

plt.figure(figsize=(8, 5))
sns.violinplot(x=target_col, y=best_feature, data=df_train, palette='muted', inner='quartile')
plt.title(f'Data Distribution by Target Class\nFeature: {best_feature}', fontsize=14, fontweight='bold')
plt.xlabel('Target (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Scaled Feature Value', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 5: Feature Importance with Threshold (Horizontal Bar Plot)
# ==========================================
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='mako')
plt.axvline(x=feature_imp_df['Importance'].mean(), color='red', linestyle='--', label='Mean Importance')
plt.title('Top 10 Feature Importances from Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Gini Importance (Information Gain)', fontsize=12)
plt.ylabel('Encoded Features', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

