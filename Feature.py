import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import logging

# ==========================================
# Setup Logging & Font
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
plt.rcParams['font.family'] = 'Tahoma'

# ==========================================
# 1. โหลดข้อมูลและจัดการชื่อคอลัมน์
# ==========================================
df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')
logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

def shorten_col_name(col):
    # กลุ่มปัจจัยและแพ็กเกจจิ้ง
    col = col.replace('คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [', 'ปัจจัย_')
    col = col.replace('บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [', 'แพ็กเกจ_')
    
    # กลุ่มความเห็นต่อ Option
    col = col.replace(' คุณคิดเห็นอย่างไรกับการออกแบบบรรจุภัณฑ์นี้', '')
    col = col.replace(' [', '_').replace(']', '')
    
    # ย่อคำ
    replacements = {
        'ใช้วัตถุดิบจากธรรมชาติ': 'ธรรมชาติ',
        'ใช้วัตถุดิบนำเข้าจากต่างประเทศ เช่น เนื้อปลาทูน่าจากญี่ปุ่น': 'นำเข้า',
        'รสชาติกลมกล่อมอร่อยถูกปากแมว เช่น เทไว้แล้วแมวกินหมดไม่เหลือ, หยิบถุงแล้วแมวรอกิน': 'รสชาติดี',
        'เป็นผลิตภัณฑ์จากต่างประเทศ เช่น ญี่ปุ่น, อเมริกา': 'แบรนด์นอก',
        'แบรนด์มีชื่อเสียงเป็นที่รู้จัก': 'แบรนด์ดัง',
        'บรรจุภัณฑ์ดูดีพรีเมียม': 'พรีเมียม',
        'บรรจุภัณฑ์มีภาพแมว': 'รูปแมว',
        'บรรจุภัณฑ์มีภาพอาหารเม็ด รูปทรงของอาหารเม็ดจริงให้เห็น': 'รูปอาหาร',
        'บรรจุภัณฑ์มีภาพวัตถุดิบและส่วนผสมจริงให้เห็น': 'รูปส่วนผสม',
        'บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม': 'รักษ์โลก',
        'มีสัญลักษณ์สื่อถึงแหล่งผลิตหรือที่มา เช่น นำเข้าจากประเทศx': 'ระบุที่มา',
        'มีสัญลักษณ์์สื่อถึงประโยชน์หรือฟังก์ชั่น เช่น ช่วยลดก้อนขน': 'ระบุประโยชน์',
        'มีการการันตี เช่น ได้รับรางวัล, ยอดขายอันดับ 1': 'การันตี',
        'รูู้สึก': 'รู้สึก', 
        'รู้สึกอยากซื้อสินค้า': 'อยากซื้อ',
        'รู้สึกโดดเด่นและแตกต่างจากแบรนด์อื่นๆ': 'โดดเด่น',
        'รู้สึกว่าสินค้าพรีเมียม คุณภาพดี': 'พรีเมียม',
        'รู้สึกถึงรสชาติที่ดีกลมกล่อม แมวจะชอบทาน': 'รสชาติดี',
        'รู้สึกว่าเป็นดีไซน์ที่เหมาะกับตัวฉัน': 'เหมาะกับฉัน'
    }
    for old_text, new_text in replacements.items():
        col = col.replace(old_text, new_text)
    return col

df.columns = [shorten_col_name(c) for c in df.columns]

# ==========================================
# 1.5 ลบคอลัมน์และกำหนดตัวแปร
# ==========================================
# ลบคอลัมน์ Metadata 
cols_to_drop = ['Timestamp', 'Brand_Cleaned', 'แมวมีความหมาย', 'คุณเลี้ยงแมว', 
                'ปัจจุบันคุณซื้ออาหาร', 'เมื่อนึกถึง', 'ถ้าคุณสามารถเพิ่ม', 'จากตัวเลือก']
existing_cols_to_drop = [c for c in df.columns if any(k in c for k in cols_to_drop)]

# แยก X และ y อย่างปลอดภัย
X = df.drop(columns=existing_cols_to_drop + ['Purchase_Target'])
y = df['Purchase_Target']
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

# ==========================================
# 2. Train/Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# Step 1 & 2: Encoding และ Scale Transformation (แบบ Production)
# ==========================================
logging.info("Step 1 & 2: Encoding and Scaling...")

# Label Encode สำหรับ Target (y)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# แยกประเภทคอลัมน์ให้อัตโนมัติ
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# สร้าง Pipeline สำหรับแปลงข้อมูล
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols), # สเกลข้อมูลตัวเลข/Likert Scale
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols) # แปลงข้อมูลหมวดหมู่
    ],
    remainder='passthrough'
)

# Apply Pipeline
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ดึงชื่อคอลัมน์กลับมาหลังจากแปลงร่างเสร็จ
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)

# ==========================================
# Step 3: Feature Selection (ใช้ค่าสถิติ ANOVA F-test ตรงตามโจทย์)
# ==========================================
logging.info("Step 3: Statistical Feature Selection (ANOVA F-test)...")

# ใช้ f_classif เพื่อหาความสัมพันธ์เชิงสถิติระหว่าง Feature และ Target
f_stats, p_values = f_classif(X_train_df, y_train_encoded)

# สร้างตารางดูผลทางสถิติ
stat_df = pd.DataFrame({
    'Feature': all_feature_names,
    'F_Statistic': f_stats,
    'P_Value': p_values
}).sort_values('P_Value', ascending=True) # เรียงจาก P-value น้อยไปมาก (ยิ่งน้อยยิ่งสัมพันธ์กันเยอะ)

# เลือกเฉพาะตัวแปรที่มีนัยสำคัญทางสถิติ (P-Value < 0.05)
significant_features = stat_df[stat_df['P_Value'] < 0.05]['Feature'].tolist()
logging.info(f"Found {len(significant_features)} statistically significant features (P-value < 0.05)")

# ==========================================
# พล็อตกราฟค่าสถิติ (F-Statistic) ให้อาจารย์ดู
# ==========================================
plt.figure(figsize=(12, 8))
# พล็อต Top 15 ตัวแปรที่มีค่า F-Statistic สูงสุด (มีผลต่อ Target มากสุด)
top_15_stats = stat_df.head(15)
sns.barplot(data=top_15_stats, x='F_Statistic', y='Feature', palette='magma')

plt.title('Top 15 Most Important Features (Statistical ANOVA F-Test)', fontsize=16, fontweight='bold')
plt.xlabel('F-Statistic Score (ค่าสถิติความสำคัญ)', fontsize=12)
plt.ylabel('ชื่อตัวแปร', fontsize=12)
plt.axvline(x=0, color='grey', linestyle='--')
plt.tight_layout()
plt.savefig('statistical_feature_importance.png', dpi=300, bbox_inches='tight')
logging.info("Statistical feature importance plot saved as 'statistical_feature_importance.png'")
plt.show()

# ==========================================
# บันทึกไฟล์พร้อมสำหรับ Train Model (เฉพาะ Feature ที่ผ่านเกณฑ์)
# ==========================================
# คัดกรองเก็บเฉพาะคอลัมน์ที่ผ่านเกณฑ์สถิติ
final_train_df = X_train_df[significant_features].copy()
final_train_df['Purchase_Target_Encoded'] = y_train_encoded
final_train_df.to_csv('Model_Ready_Train_Data_Selected.csv', index=False, encoding='utf-8-sig')

final_test_df = X_test_df[significant_features].copy()
final_test_df['Purchase_Target_Encoded'] = y_test_encoded
final_test_df.to_csv('Model_Ready_Test_Data_Selected.csv', index=False, encoding='utf-8-sig')

logging.info(f"✅ บันทึกไฟล์ที่คัดเลือก Feature เรียบร้อยแล้ว!")
logging.info(f"   - Model_Ready_Train_Data_Selected.csv ({final_train_df.shape})")
logging.info(f"   - Model_Ready_Test_Data_Selected.csv ({final_test_df.shape})")