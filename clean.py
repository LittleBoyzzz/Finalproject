import pandas as pd
import numpy as np
import os

# ตั้งค่าที่อยู่ไฟล์ให้ตรงกับที่เก็บ Script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load Data (ข้ามบรรทัด Brief)
df = pd.read_csv("BU Data from Survey Cases_final(5).csv", encoding='utf-8-sig', skiprows=1)
initial_rows = len(df)

# 2. กรองเฉพาะเนื้อข้อมูลจริง (ผู้ที่มีประสบการณ์เลี้ยง/ซื้อ)
df = df[df.iloc[:, 1] == 'เคย'].copy()

# ---------------------------------------------------------
# 3. Standardization (จัดกลุ่มแบรนด์และสายพันธุ์แมว)
# ---------------------------------------------------------

# ฟังก์ชันจัดกลุ่มสายพันธุ์แมว (เพิ่มตามคำขอ)
def standardize_breed(text):
    if pd.isna(text): return 'อื่นๆ/ไม่ระบุ'
    text = str(text).lower().strip()
    if any(x in text for x in ['ไทย', 'บ้าน', 'จร', 'สลิด', 'mix']): return 'พันธุ์ไทย/ผสม'
    if any(x in text for x in ['persian', 'เปอร์เซีย']): return 'เปอร์เซีย'
    if any(x in text for x in ['british', 'บริติช']): return 'บริติช ช็อตแฮร์'
    if any(x in text for x in ['scottish', 'สก๊อตติช']): return 'สก๊อตติช โฟลด์'
    if any(x in text for x in ['ragdoll', 'แร็กดอลล์']): return 'แร็กดอลล์'
    return 'อื่นๆ'

# ฟังก์ชันจัดกลุ่มแบรนด์
def standardize_brand(text):
    if pd.isna(text): return 'อื่นๆ'
    text = str(text).lower().strip()
    if any(x in text for x in ['whiskas', 'วิสกัส']): return 'Whiskas'
    if any(x in text for x in ['purina', 'one', 'พูริน่า']): return 'Purina'
    if any(x in text for x in ['royal', 'canin', 'โรยัล']): return 'Royal Canin'
    if any(x in text for x in ['kaniva', 'คานิว่า']): return 'Kaniva'
    return 'Other'

# ระบุชื่อคอลัมน์จากไฟล์จริง
breed_col = 'คุณเลี้ยงแมวพันธุ์ใด [โปรดพิมพ์ระบุ]'
brand_col = 'ปัจจุบันคุณซื้ออาหารแมวสำเร็จรูปชนิดเม็ดแบรนด์ใด[โปรดพิมพ์ระบุ]'

# สร้างคอลัมน์ใหม่ที่จัดกลุ่มแล้ว
if breed_col in df.columns:
    df['Breed_Cleaned'] = df[breed_col].apply(standardize_breed)
if brand_col in df.columns:
    df['Brand_Cleaned'] = df[brand_col].apply(standardize_brand)

# ---------------------------------------------------------
# 4. กำจัด Noise (ตัดคอลัมน์ที่ว่างเกิน 30% ออก)
# ---------------------------------------------------------
# เราเก็บคอลัมน์ที่เราเพิ่งสร้างใหม่ไว้ด้วย โดยการรวมเข้ากับเกณฑ์การตัด
limit = len(df) * 0.7
df = df.dropna(thresh=limit, axis=1)

# ---------------------------------------------------------
# 5. แปลงค่าความรู้สึก (Likert Scale) เป็นตัวเลข 1-5
# ---------------------------------------------------------
likert_map = {
    'เห็นด้วยที่สุด': 5, 'มากที่สุด': 5,
    'เห็นด้วย': 4, 'มาก': 4,
    'เฉยๆ': 3, 'ปานกลาง': 3,
    'ไม่เห็นด้วย': 2, 'น้อย': 2,
    'ไม่เห็นด้วยที่สุด': 1, 'น้อยที่สุด': 1,
    'ไม่เห็นด้วยเลย': 1
}

for col in df.columns:
    if df[col].dtype == 'object':
        # เช็คว่ามีค่าใน Likert Map หรือไม่ (ข้ามค่า NaN)
        unique_vals = [v for v in df[col].unique() if pd.notna(v)]
        if any(val in likert_map for val in unique_vals):
            df[col] = df[col].map(likert_map).fillna(df[col])
            df[col] = pd.to_numeric(df[col], errors='ignore')

# ---------------------------------------------------------
# 6. สร้าง Target Variable (Option 6)
# ---------------------------------------------------------
target_candidates = [col for col in df.columns if 'Option 6' in col]
if target_candidates:
    target_col = target_candidates[0]
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df['Purchase_Target'] = df[target_col].apply(lambda x: 1 if pd.notna(x) and x >= 4 else 0)

# 7. บันทึกผลลัพธ์
df.to_csv('Cleaned_Actual_Data_Full.csv', index=False, encoding='utf-8-sig')

print(f"--- สรุปการทำความสะอาดข้อมูล ---")
print(f"แถวเริ่มต้น: {initial_rows} -> เหลือเนื้อข้อมูลจริง: {len(df)} แถว")
print(f"คอลัมน์ที่ถูกเก็บไว้ (ที่มีคนตอบเยอะ): {len(df.columns)} คอลัมน์")
print(f"สร้างคอลัมน์ 'Breed_Cleaned' และ 'Brand_Cleaned' เรียบร้อยแล้ว")