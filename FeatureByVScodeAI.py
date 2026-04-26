import pandas as pd

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')

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

# เช็คผลลัพธ์ว่าชื่อเปลี่ยนสำเร็จไหม
print("✅ ชื่อคอลัมน์หลังเปลี่ยน:")
for col in df.columns:
    print("-", col)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 0. ตั้งค่าฟอนต์ภาษาไทยสำหรับกราฟ
# ==========================================
plt.rcParams['font.family'] = 'Tahoma' # หรือ 'Sarabun'

# ==========================================
# 1. ลบคอลัมน์ที่ไม่ต้องใช้
# ==========================================
# ลบคอลัมน์เพิ่มเติมที่ไม่จำเป็นสำหรับโมเดล
cols_to_drop = [col for col in df.columns if 'Timestamp' in col or 'Brand_Cleaned' in col or 'Purchase_Target' in col 
                or 'แมวมีความหมาย' in col or 'คุณเลี้ยงแมว' in col or 'ปัจจุบันคุณซื้ออาหาร' in col 
                or 'เมื่อนึกถึง' in col or 'ถ้าคุณสามารถเพิ่ม' in col or 'จากตัวเลือก' in col]

# แยก Features (X) และตัวแปรเป้าหมาย Target (y)
X = df.drop(columns=cols_to_drop)
y = df['Purchase_Target']

# ==========================================
# Step 1: Data Encoding (แปลงข้อความเป็นตัวเลข)
# ==========================================
print("กำลังทำ Step 1: Encoding...")

# 1.1 แปลงตัวแปรเป้าหมาย (y) เป็นตัวเลข (เช่น 0, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 1.2 แปลงตัวแปรอิสระ (X) ด้วย One-Hot Encoding
# ให้โปรแกรมหาเองว่าคอลัมน์ไหนเป็นข้อความ คอลัมน์ไหนเป็นตัวเลข
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ทำ One-Hot Encoding เฉพาะคอลัมน์ข้อความ (drop_first=True เพื่อลดความซ้ำซ้อน)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ==========================================
# Step 2: Feature Transformation (ปรับสเกลข้อมูล)
# ==========================================
print("กำลังทำ Step 2: Transformation...")

# ปรับสเกลด้วย Z-Score เฉพาะคอลัมน์ที่เป็นตัวเลขมาตั้งแต่ต้น 
# (ไม่ไปยุ่งกับคอลัมน์ 0, 1 ที่เพิ่งแปลงมา)
scaler = StandardScaler()
if len(num_cols) > 0:
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

# ==========================================
# Step 3: Feature Selection (คัดเลือกตัวแปรที่สำคัญ)
# ==========================================
print("กำลังทำ Step 3: Feature Selection...")

# ใช้โมเดล Random Forest เพื่อหาว่าคอลัมน์ไหนส่งผลต่อ Target มากที่สุด
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_encoded, y_encoded)

# ดึงคะแนนความสำคัญ (Feature Importances) ออกมา
importances = model.feature_importances_
feature_names = X_encoded.columns

# สร้างเป็นตาราง DataFrame เพื่อให้ดูง่ายๆ
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n=== ปัจจัยที่ส่งผลต่อ Target มากที่สุด 10 อันดับแรก ===")
print(feature_imp_df.head(10))

# ==========================================
# พล็อตกราฟ Feature Importance ให้อาจารย์ดู
# ==========================================
plt.figure(figsize=(10, 8))
# เลือกแสดงแค่ Top 15 คอลัมน์ จะได้ไม่รกเกินไป
sns.barplot(data=feature_imp_df.head(15), x='Importance', y='Feature', palette='viridis')

plt.title('Top 15 Feature Importances (ปัจจัยที่สำคัญที่สุด)', fontsize=16)
plt.xlabel('ระดับความสำคัญ (Importance Score)', fontsize=12)
plt.ylabel('ชื่อคอลัมน์', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# Optional: เซฟไฟล์เก็บไว้ใช้รัน Machine Learning ในอนาคต
# ==========================================
# เอา X ที่แปลงร่างเสร็จแล้ว มารวมกับ y แล้วเซฟเป็นไฟล์ใหม่
final_df = X_encoded.copy()
final_df[target_col] = y_encoded
final_df.to_csv('Model_Ready_Data.csv', index=False, encoding='utf-8-sig')
print("\n✅ บันทึกไฟล์ Model_Ready_Data.csv สำหรับทำขั้นตอนต่อไปเรียบร้อยแล้ว!")