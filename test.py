import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging

# ==========================================
# Setup Logging
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 0. ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ
# ==========================================
plt.rcParams['font.family'] = 'Tahoma'

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')
logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ==========================================
# 🌟 เพิ่มส่วนนี้: ฟังก์ชันตัดคำยาวๆ ให้สั้นลง
# ==========================================
def shorten_col_name(col):
    # 1. กลุ่มปัจจัยอาหารแมว (เปลี่ยนคำยาวๆ ให้เหลือแค่ "ปัจจัย_")
    col = col.replace('คุณสมบัติของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [', 'ปัจจัย_')
    
    # 2. กลุ่มแพ็กเกจจิ้ง (เปลี่ยนคำยาวๆ ให้เหลือแค่ "แพ็กเกจ_")
    col = col.replace('บรรจุภัณฑ์ของอาหารแมวสำเร็จรูปชนิดเม็ดที่ส่งผลต่อการตัดสินใจซื้อ [', 'แพ็กเกจ_')
    
    # 3. กลุ่มความเห็นต่อ Option (ลบคำถามยาวๆ ทิ้ง)
    col = col.replace(' คุณคิดเห็นอย่างไรกับการออกแบบบรรจุภัณฑ์นี้', '')
    col = col.replace(' [', '_') # เปลี่ยนเว้นวรรคและวงเล็บเป็น Underscore
    
    # 4. ตัดวงเล็บปิด ] ทิ้งให้หมด
    col = col.replace(']', '')
    
    # 5. ย่อคำที่เหลือในวงเล็บให้สั้นลง
    replacements = {
        # ย่อฝั่งปัจจัยอาหาร
        'ใช้วัตถุดิบจากธรรมชาติ': 'ธรรมชาติ',
        'ใช้วัตถุดิบนำเข้าจากต่างประเทศ เช่น เนื้อปลาทูน่าจากญี่ปุ่น': 'นำเข้า',
        'รสชาติกลมกล่อมอร่อยถูกปากแมว เช่น เทไว้แล้วแมวกินหมดไม่เหลือ, หยิบถุงแล้วแมวรอกิน': 'รสชาติดี',
        'เป็นผลิตภัณฑ์จากต่างประเทศ เช่น ญี่ปุ่น, อเมริกา': 'แบรนด์นอก',
        'แบรนด์มีชื่อเสียงเป็นที่รู้จัก': 'แบรนด์ดัง',
        
        # ย่อฝั่งแพ็กเกจจิ้ง
        'บรรจุภัณฑ์ดูดีพรีเมียม': 'พรีเมียม',
        'บรรจุภัณฑ์มีภาพแมว': 'รูปแมว',
        'บรรจุภัณฑ์มีภาพอาหารเม็ด รูปทรงของอาหารเม็ดจริงให้เห็น': 'รูปอาหาร',
        'บรรจุภัณฑ์มีภาพวัตถุดิบและส่วนผสมจริงให้เห็น': 'รูปส่วนผสม',
        'บรรจุภัณฑ์เป็นมิตรต่อสิ่งแวดล้อม': 'รักษ์โลก',
        'มีสัญลักษณ์สื่อถึงแหล่งผลิตหรือที่มา เช่น นำเข้าจากประเทศx': 'ระบุที่มา',
        'มีสัญลักษณ์์สื่อถึงประโยชน์หรือฟังก์ชั่น เช่น ช่วยลดก้อนขน': 'ระบุประโยชน์',
        'มีการการันตี เช่น ได้รับรางวัล, ยอดขายอันดับ 1': 'การันตี',
        
        # ย่อฝั่ง Option 1-10
        'รูู้สึก': 'รู้สึก', # แก้คำผิดใน Survey ต้นฉบับ
        'รู้สึกอยากซื้อสินค้า': 'อยากซื้อ',
        'รู้สึกโดดเด่นและแตกต่างจากแบรนด์อื่นๆ': 'โดดเด่น',
        'รู้สึกว่าสินค้าพรีเมียม คุณภาพดี': 'พรีเมียม',
        'รู้สึกถึงรสชาติที่ดีกลมกล่อม แมวจะชอบทาน': 'รสชาติดี',
        'รู้สึกว่าเป็นดีไซน์ที่เหมาะกับตัวฉัน': 'เหมาะกับฉัน'
    }
    
    for old_text, new_text in replacements.items():
        col = col.replace(old_text, new_text)
        
    return col

# ใช้งานฟังก์ชันเพื่อเปลี่ยนชื่อทุกคอลัมน์ใน DataFrame
df.columns = [shorten_col_name(c) for c in df.columns]

logging.info("✅ ชื่อคอลัมน์หลังเปลี่ยน:")
for col in df.columns:
    logging.info(f"  - {col}")

# ==========================================
# 1.5 ลบคอลัมน์ที่ไม่ต้องใช้
# ==========================================
# ลบคอลัมน์เพิ่มเติมที่ไม่จำเป็นสำหรับโมเดล
cols_to_drop = [col for col in df.columns if any(keyword in col for keyword in [
    'Timestamp', 'Brand_Cleaned', 'Purchase_Target',
    'แมวมีความหมาย', 'คุณเลี้ยงแมว', 'ปัจจุบันคุณซื้ออาหาร',
    'เมื่อนึกถึง', 'ถ้าคุณสามารถเพิ่ม', 'จากตัวเลือก'
])]

logging.info(f"Dropping {len(cols_to_drop)} columns")

# แยก Features (X) และตัวแปรเป้าหมาย Target (y)
X = df.drop(columns=cols_to_drop)
y = df['Purchase_Target']
logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

# ==========================================
# 2. Train/Test Split (เพื่อหลีกเลี่ยง Data Leakage)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ==========================================
# Step 1: Data Encoding (แปลงข้อความเป็นตัวเลข)
# ==========================================
logging.info("Step 1: Encoding...")

# 1.1 แปลงตัวแปรเป้าหมาย (y) เป็นตัวเลข (เช่น 0, 1)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

logging.info(f"Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 1.2 แปลงตัวแปรอิสระ (X) ด้วย One-Hot Encoding บน Train set เท่านั้น
# 1.2 แปลงตัวแปรอิสระ (X) ด้วย One-Hot Encoding บน Train set เท่านั้น
# ให้โปรแกรมหาเองว่าคอลัมน์ไหนเป็นข้อความ คอลัมน์ไหนเป็นตัวเลข
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

logging.info(f"Categorical columns: {cat_cols}")
logging.info(f"Numerical columns: {num_cols}")

# ทำ One-Hot Encoding เฉพาะคอลัมน์ข้อความ (drop_first=True เพื่อลดความซ้ำซ้อน)
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align columns (ถ้า test set มีคอลัมน์ต่างจาก train set)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
logging.info(f"After encoding - Train: {X_train_encoded.shape}, Test: {X_test_encoded.shape}")

# ==========================================
# Step 2: Feature Transformation (ปรับสเกลข้อมูล)
# ==========================================
logging.info("Step 2: Transformation...")

# ปรับสเกลด้วย Z-Score เฉพาะคอลัมน์ที่เป็นตัวเลขมาตั้งแต่ต้น 
# (ไม่ไปยุ่งกับคอลัมน์ 0, 1 ที่เพิ่งแปลงมา) - Fit บน Train set!
scaler = StandardScaler()
if len(num_cols) > 0:
    X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])
    logging.info(f"Scaling applied to {num_cols}")

# ==========================================
# Step 3: Feature Selection (คัดเลือกตัวแปรที่สำคัญ)
# ==========================================
logging.info("Step 3: Feature Selection with Cross-Validation...")

# ใช้โมเดล Random Forest เพื่อหาว่าคอลัมน์ไหนส่งผลต่อ Target มากที่สุด
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_encoded, y_train_encoded)

# Cross-Validation Score
cv_scores = cross_val_score(model, X_train_encoded, y_train_encoded, cv=5)
logging.info(f"Cross-Validation Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Test Set Score
test_score = model.score(X_test_encoded, y_test_encoded)
logging.info(f"Test Set Score: {test_score:.4f}")

# ดึงคะแนนความสำคัญ (Feature Importances) ออกมา
importances = model.feature_importances_
feature_names = X_train_encoded.columns

# สร้างเป็นตาราง DataFrame เพื่อให้ดูง่ายๆ
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

logging.info("=== Top 10 Feature Importances ===")
for idx, row in feature_imp_df.head(10).iterrows():
    logging.info(f"{row['Feature']}: {row['Importance']:.6f}")

# ==========================================
# พล็อตกราฟ Feature Importance ให้อาจารย์ดู
# ==========================================
plt.figure(figsize=(12, 8))
# เลือกแสดงแค่ Top 15 คอลัมน์ จะได้ไม่รกเกินไป
sns.barplot(data=feature_imp_df.head(15), x='Importance', y='Feature', palette='viridis')

plt.title('Top 15 Feature Importances (ปัจจัยที่สำคัญที่สุด)', fontsize=16, fontweight='bold')
plt.xlabel('ระดับความสำคัญ (Importance Score)', fontsize=12)
plt.ylabel('ชื่อคอลัมน์', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
logging.info("Feature importance plot saved as 'feature_importance.png'")
plt.show()

# ==========================================
# Optional: เซฟไฟล์เก็บไว้ใช้รัน Machine Learning ในอนาคต
# ==========================================
# เอา X ที่แปลงร่างเสร็จแล้ว มารวมกับ y แล้วเซฟเป็นไฟล์ใหม่
final_train_df = X_train_encoded.copy()
final_train_df['Purchase_Target_Encoded'] = y_train_encoded
final_train_df.to_csv('Model_Ready_Train_Data.csv', index=False, encoding='utf-8-sig')

final_test_df = X_test_encoded.copy()
final_test_df['Purchase_Target_Encoded'] = y_test_encoded
final_test_df.to_csv('Model_Ready_Test_Data.csv', index=False, encoding='utf-8-sig')

logging.info(f"✅ บันทึกไฟล์สำหรับทำขั้นตอนต่อไปเรียบร้อยแล้ว!")
logging.info(f"   - Model_Ready_Train_Data.csv ({final_train_df.shape})")
logging.info(f"   - Model_Ready_Test_Data.csv ({final_test_df.shape})")
logging.info(f"Class mapping saved: {dict(zip(le.classes_, le.transform(le.classes_)))}")