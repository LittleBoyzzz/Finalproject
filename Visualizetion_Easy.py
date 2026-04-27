import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# ⚙️ ตั้งค่า Constants
# ==========================================
FONT_FAMILY = 'Tahoma'
TOP_PACKAGE_FEATURES = 5
TOP_IMPORTANCE_FEATURES = 7
RANDOM_STATE = 42
N_ESTIMATORS = 100

# ตั้งค่าฟอนต์ภาษาไทยและสไตล์กราฟให้ดูสะอาดตา
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={"font.family": FONT_FAMILY})

# ==========================================
# 📂 โหลดข้อมูล
# ==========================================
try:
    df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')
    print(f"✅ โหลด Cleaned_Actual_Data_Full.csv สำเร็จ ({len(df)} rows)")
except FileNotFoundError:
    print("❌ ไม่พบไฟล์ Cleaned_Actual_Data_Full.csv")
    exit()

try:
    df_model = pd.read_csv('Model_Ready_Train_Data_Selected.csv', encoding='utf-8-sig')
    print(f"✅ โหลด Model_Ready_Train_Data_Selected.csv สำเร็จ ({len(df_model)} rows)")
except FileNotFoundError:
    print("❌ ไม่พบไฟล์ Model_Ready_Train_Data_Selected.csv")
    exit()

# ==========================================
# กราฟที่ 1: ข้อมูลทั่วไปของผู้ตอบแบบสอบถาม (Countplot)
# ==========================================
plt.figure(figsize=(8, 5))
# สมมติว่ามีคอลัมน์ช่วงอายุ ลองเปลี่ยนชื่อคอลัมน์ให้ตรงกับในไฟล์จริง
# หากคอลัมน์ชื่ออื่นให้แก้ตรง x='...'
sns.countplot(x='อายุของคุณ', hue='อายุของคุณ', data=df, palette='Set2', legend=False) 
plt.title('จำนวนผู้ตอบแบบสอบถาม แบ่งตามช่วงอายุ', fontsize=14)
plt.xlabel('ช่วงอายุ', fontsize=12)
plt.ylabel('จำนวน (คน)', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 2: เปรียบเทียบคะแนนความชอบแพ็กเกจจิ้ง (Barplot)
# ==========================================
# เลือกคอลัมน์ที่ถามเรื่องความสำคัญของแพ็กเกจจิ้งมาหาค่าเฉลี่ย
pkg_cols = [col for col in df.columns if 'บรรจุภัณฑ์ของอาหารแมว' in col][:TOP_PACKAGE_FEATURES]
pkg_mean = df[pkg_cols].mean().sort_values(ascending=False)

# สร้าง labels ที่อ่านง่ายขึ้น
pkg_labels = [col.split('[')[-1].rstrip(']') for col in pkg_mean.index]

plt.figure(figsize=(10, 5))
sns.barplot(x=pkg_mean.values, y=pkg_labels, hue=pkg_labels, palette='Blues_r', legend=False)
plt.title('5 อันดับแรก: ลักษณะแพ็กเกจจิ้งที่ลูกค้าให้ความสำคัญ (คะแนนเฉลี่ย)', fontsize=14)
plt.xlabel('คะแนนเฉลี่ย (เต็ม 5)', fontsize=12)
plt.ylabel('ลักษณะบรรจุภัณฑ์', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 3: ความสัมพันธ์ของตัวแปรแบบง่ายๆ (Correlation Heatmap)
# ==========================================
plt.figure(figsize=(8, 6))
# ดึงข้อมูลจากไฟล์ Model_Ready มาสัก 5-6 ตัวแปรที่สำคัญ เพื่อดูความสัมพันธ์
cols_for_heatmap = df_model.columns[:6] 
corr_matrix = df_model[cols_for_heatmap].corr()

sns.heatmap(corr_matrix, annot=True, cmap='Reds', fmt=".2f", linewidths=.5)
plt.title('ความสัมพันธ์ระหว่างตัวแปรที่สำคัญ (Correlation Heatmap)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 4 (แก้ไขแล้ว): เปรียบเทียบคะแนนความ "อยากซื้อ" ของ 2 ตัวเลือกหลัก (Boxplot)
# ==========================================
# 1. ค้นหาคอลัมน์ที่มีคำว่า Option 6/7 และ อยากซื้อ (รับรองว่าหาเจอแน่นอน)
options_to_compare = [
    col for col in df.columns 
    if ('Option 6' in col and 'อยากซื้อ' in col) or ('Option 7' in col and 'อยากซื้อ' in col)
]

# ปริ้นท์เช็คให้ชัวร์ว่าดึงคอลัมน์มาได้แล้ว
print(f"✅ คอลัมน์ที่ดึงมาทำกราฟ 4: {len(options_to_compare)} คอลัมน์")

plt.figure(figsize=(8, 5))
df_compare = df[options_to_compare].melt(var_name='ตัวเลือก', value_name='คะแนนความอยากซื้อ')

# 2. แปลงชื่อคอลัมน์ยาวๆ ให้เหลือแค่คำว่า 'Option 6' หรือ 'Option 7' ไว้โชว์ใต้กราฟ
df_compare['ตัวเลือก'] = df_compare['ตัวเลือก'].apply(lambda x: 'Option 6' if 'Option 6' in x else 'Option 7')

# 3. วาดกราฟ
sns.boxplot(x='ตัวเลือก', y='คะแนนความอยากซื้อ', hue='ตัวเลือก', data=df_compare, palette='pastel', legend=False)
plt.title('เปรียบเทียบการกระจายตัวของคะแนนความอยากซื้อ: Option 6 vs Option 7', fontsize=14)
plt.xlabel('ตัวเลือกแพ็กเกจ', fontsize=12)
plt.ylabel('คะแนน', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# กราฟที่ 5: ผลลัพธ์จากโมเดล AI ของเรา (Feature Importance)
# ==========================================
# รันโค้ด Random Forest แบบคลีนๆ เพื่อเอากราฟไปส่งอาจารย์
X = df_model.drop(columns=['Purchase_Target_Encoded'], errors='ignore')
if 'Purchase_Target_Encoded' not in df_model.columns:
    print("❌ ไม่พบ column 'Purchase_Target_Encoded' ใน Model_Ready ไฟล์")
    exit()
    
y = df_model['Purchase_Target_Encoded']

rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
rf.fit(X, y)
print(f"✅ Random Forest trained ({N_ESTIMATORS} estimators)")

imp_df = pd.DataFrame({'ปัจจัย': X.columns, 'ความสำคัญ': rf.feature_importances_})
imp_df = imp_df.sort_values(by='ความสำคัญ', ascending=False).head(TOP_IMPORTANCE_FEATURES)

plt.figure(figsize=(10, 5))
sns.barplot(x='ความสำคัญ', y='ปัจจัย', hue='ปัจจัย', data=imp_df, palette='viridis', legend=False)
plt.title('Top 7 ปัจจัยที่ส่งผลต่อการตัดสินใจซื้อมากที่สุด (จากโมเดล Random Forest)', fontsize=14)
plt.xlabel('ระดับความสำคัญ (ยิ่งมากยิ่งมีผล)', fontsize=12)
plt.ylabel('ตัวแปร', fontsize=12)
plt.tight_layout()
plt.show()