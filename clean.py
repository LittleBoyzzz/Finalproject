import pandas as pd
import numpy as np
import os

# Change working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load Data (ต่อยอดจาก Target.py)
# Skip the first row (description) and use the second row as headers
df = pd.read_csv("BU Data from Survey Cases_final(5).csv", encoding='utf-8-sig', skiprows=1)
initial_rows = len(df)

# กรองเฉพาะผู้ที่ 'เคย' ซื้ออาหารแมว (ตาม Logic เดิมใน Target.py)
df = df[df.iloc[:, 1] == 'เคย'].copy()

# ---------------------------------------------------------
# 2. Handle Incomplete Responses
# ---------------------------------------------------------
# ระบุคอลัมน์ Option ทั้ง 10 (อ้างอิงจากตำแหน่งคอลัมน์ในไฟล์ CSV)
# Option 1 เริ่มที่ index 13 จนถึง Option 10 ที่ index 49 (โดยประมาณ)
option_cols = [col for col in df.columns if 'Option' in col and 'ความพึงพอใจ' in col]

# ลบแถวที่มีค่าว่าง (NaN) ในคอลัมน์ Option เกิน 50% (คือว่างเกิน 5 จาก 10 Option)
threshold = len(option_cols) * 0.5
df = df.dropna(subset=option_cols, thresh=int(len(option_cols) - threshold))

# ---------------------------------------------------------
# 3. Standardization (Open-ended Data)
# ---------------------------------------------------------

# ฟังก์ชันจัดกลุ่มแบรนด์
def standardize_brand(text):
    if pd.isna(text): return 'Other/Unknown'
    text = str(text).lower().strip()
    if any(x in text for x in ['whiskas', 'วิสกัส']): return 'Whiskas'
    if any(x in text for x in ['purina', 'one', 'พูริน่า']): return 'Purina'
    if any(x in text for x in ['royal', 'canin', 'โรยัล']): return 'Royal Canin'
    if any(x in text for x in ['kaniva', 'คานิว่า']): return 'Kaniva'
    if any(x in text for x in ['me-o', 'มีโอ']): return 'Me-O'
    return 'Other'

# ฟังก์ชันจัดกลุ่มสายพันธุ์แมว
def standardize_breed(text):
    if pd.isna(text): return 'ไม่ระบุ'
    text = str(text).lower().strip()
    if any(x in text for x in ['ไทย', 'บ้าน', 'จร', 'mix']): return 'พันธุ์ไทย/ผสม'
    if any(x in text for x in ['persian', 'เปอร์เซีย']): return 'เปอร์เซีย'
    if any(x in text for x in ['british', 'บริติช']): return 'บริติช ช็อตแฮร์'
    if any(x in text for x in ['scottish', 'สก๊อตติช']): return 'สก๊อตติช โฟลด์'
    return 'อื่นๆ'

brand_col = 'ปัจจุบันคุณซื้ออาหารแมวสำเร็จรูปชนิดเม็ดแบรนด์ใด[โปรดพิมพ์ระบุ]'
breed_col = 'คุณเลี้ยงแมวพันธุ์ใด [โปรดพิมพ์ระบุ]'

df['Brand_Standardized'] = df[brand_col].apply(standardize_brand)
df['Breed_Standardized'] = df[breed_col].apply(standardize_breed)

# ---------------------------------------------------------
# 4. Logic Check (Straight-lining)
# ---------------------------------------------------------
# ตรวจสอบว่าให้คะแนนเท่ากันหมดทุก Option หรือไม่ (เช่น 5 หมด หรือ 1 หมด)
# เราจะเช็คเฉพาะแถวที่ตอบครบในคอลัมน์ Option
def is_straight_lining(row):
    valid_scores = row[option_cols].dropna()
    if len(valid_scores) > 1:
        # ถ้าค่า Max กับ Min เท่ากัน แสดงว่าตอบเหมือนกันหมด
        return valid_scores.nunique() == 1
    return False

df = df[~df.apply(is_straight_lining, axis=1)]

# ---------------------------------------------------------
# 5. Output & Save
# ---------------------------------------------------------
# สร้าง Target (จาก Target.py เดิม)
target_col_name = [col for col in df.columns if 'Option 6' in col][0]
def map_to_label(val):
    if val in ['เห็นด้วย', 'เห็นด้วยที่สุด']: return 1
    return 0

df['Purchase_Target'] = df[target_col_name].apply(map_to_label)

print(f"--- Data Cleaning Summary ---")
print(f"จำนวนข้อมูลเริ่มต้น: {initial_rows} แถว")
print(f"จำนวนข้อมูลหลัง Clean: {len(df)} แถว")
print(f"ข้อมูลถูกลบออกไป: {initial_rows - len(df)} แถว")

# บันทึกไฟล์
df.to_csv('Cleaned_CatFood_Data.csv', index=False, encoding='utf-8-sig')
print("\nบันทึกไฟล์ 'Cleaned_CatFood_Data.csv' เรียบร้อยแล้ว")