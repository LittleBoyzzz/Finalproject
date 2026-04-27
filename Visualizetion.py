import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier




# Constants

FONT_FAMILY = 'Tahoma'
TOP_PACKAGE_FEATURES = 5
TOP_IMPORTANCE_FEATURES = 7
RANDOM_STATE = 42
N_ESTIMATORS = 100

# ตั้งค่าฟอนต์ภาษาไทย
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={"font.family": FONT_FAMILY})



# โหลดข้อมูล

try:
    df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')
    print(f"โหลด Cleaned_Actual_Data_Full.csv สำเร็จ ({len(df)} rows)")
except FileNotFoundError:
    print("ไม่พบไฟล์ Cleaned_Actual_Data_Full.csv")
    exit()

try:
    df_model = pd.read_csv('Model_Ready_Train_Data_Selected.csv', encoding='utf-8-sig')
    print(f"โหลด Model_Ready_Train_Data_Selected.csv สำเร็จ ({len(df_model)} rows)")
except FileNotFoundError:
    print("ไม่พบไฟล์ Model_Ready_Train_Data_Selected.csv")
    exit()



# กราฟที่ 1 ข้อมูลทั่วไปของผู้ตอบแบบสอบถาม (Countplot)

plt.figure(figsize=(8, 5))
sns.countplot(x='อายุของคุณ', hue='อายุของคุณ', data=df, palette='Set2', legend=False) 
plt.title('จำนวนผู้ตอบแบบสอบถาม แบ่งตามช่วงอายุ', fontsize=14)
plt.xlabel('ช่วงอายุ', fontsize=12)
plt.ylabel('จำนวน (คน)', fontsize=12)
plt.tight_layout()
plt.savefig('01_respondent_age_distribution.png', dpi=300, bbox_inches='tight')
print("บันทึก: 01_respondent_age_distribution.png")
plt.show()



# กราฟที่ 2 เปรียบเทียบคะแนนความชอบแพ็กเกจจิ้ง (Barplot)

pkg_cols = [col for col in df.columns if 'บรรจุภัณฑ์ของอาหารแมว' in col][:TOP_PACKAGE_FEATURES]
pkg_mean = df[pkg_cols].mean().sort_values(ascending=False)


pkg_labels = [col.split('[')[-1].rstrip(']') for col in pkg_mean.index]

plt.figure(figsize=(10, 5))
sns.barplot(x=pkg_mean.values, y=pkg_labels, hue=pkg_labels, palette='Blues_r', legend=False)
plt.title('5 อันดับแรก: ลักษณะแพ็กเกจจิ้งที่ลูกค้าให้ความสำคัญ (คะแนนเฉลี่ย)', fontsize=14)
plt.xlabel('คะแนนเฉลี่ย (เต็ม 5)', fontsize=12)
plt.ylabel('ลักษณะบรรจุภัณฑ์', fontsize=12)
plt.tight_layout()
plt.savefig('02_top_package_features.png', dpi=300, bbox_inches='tight')
print("บันทึก: 02_top_package_features.png")
plt.show()



# กราฟที่ 3 ความสัมพันธ์ของตัวแปรแบบง่ายๆ (Correlation Heatmap)

plt.figure(figsize=(8, 6))

cols_for_heatmap = df_model.columns[:6] 
corr_matrix = df_model[cols_for_heatmap].corr()

sns.heatmap(corr_matrix, annot=True, cmap='Reds', fmt=".2f", linewidths=.5)
plt.title('ความสัมพันธ์ระหว่างตัวแปรที่สำคัญ (Correlation Heatmap)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("บันทึก: 03_correlation_heatmap.png")
plt.show()



# กราฟที่ 4 เปรียบเทียบคะแนนความอยากซื้อของ 2 ตัวเลือกหลัก (Boxplot)

options_to_compare = [
    col for col in df.columns 
    if ('Option 6' in col and 'อยากซื้อ' in col) or ('Option 7' in col and 'อยากซื้อ' in col)
]


print(f"คอลัมน์ที่ดึงมาทำกราฟ 4: {len(options_to_compare)} คอลัมน์")

plt.figure(figsize=(8, 5))
df_compare = df[options_to_compare].melt(var_name='ตัวเลือก', value_name='คะแนนความอยากซื้อ')


df_compare['ตัวเลือก'] = df_compare['ตัวเลือก'].apply(lambda x: 'Option 6' if 'Option 6' in x else 'Option 7')


sns.boxplot(x='ตัวเลือก', y='คะแนนความอยากซื้อ', hue='ตัวเลือก', data=df_compare, palette='pastel', legend=False)
plt.title('เปรียบเทียบการกระจายตัวของคะแนนความอยากซื้อ: Option 6 vs Option 7', fontsize=14)
plt.xlabel('ตัวเลือกแพ็กเกจ', fontsize=12)
plt.ylabel('คะแนน', fontsize=12)
plt.tight_layout()
plt.savefig('04_purchase_intention_comparison.png', dpi=300, bbox_inches='tight')
print("บันทึก: 04_purchase_intention_comparison.png")
plt.show()



# 5. กราฟที่ 5 แสดงความสำคัญของ Feature จากโมเดล Random Forest (Feature Importance)

df_model = pd.read_csv('Model_Ready_Train_Data_Selected.csv')

X = df_model.drop(columns=['Purchase_Target_Encoded'])
y = df_model['Purchase_Target_Encoded']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importances = model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 8))

sns.barplot(data=feature_imp_df.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)

plt.title('Top 15 Feature Importances (ปัจจัยที่สำคัญที่สุด)', fontsize=16, fontweight='bold')
plt.xlabel('ระดับความสำคัญ (Importance Score)', fontsize=12)
plt.ylabel('ชื่อคอลัมน์', fontsize=12)

plt.tight_layout()
plt.savefig('05_top_15_feature_importances.png', dpi=300, bbox_inches='tight')
print("บันทึก: 05_top_15_feature_importances.png")
plt.show()