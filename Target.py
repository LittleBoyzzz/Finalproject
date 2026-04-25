import pandas as pd


df = pd.read_csv("BU Data from Survey Cases_final(5).csv", encoding='utf-8-sig')
#print(df.head())

# 2. ทำความสะอาดเบื้องต้น: เลือกเฉพาะผู้ที่ 'เคย' ซื้ออาหารแมว
df = df[df.iloc[:, 1] == 'เคย']

# 3. กำหนดตัวแปรเป้าหมาย (ใช้ Option 6 เป็นตัวอย่าง)
target_col = df[df.iloc[:, 48].notnull()].columns[0]  # คอลัมน์ที่มีข้อมูลไม่ว่างใน Option 6'

# 4. สร้าง Function สำหรับทำ Label (Binary Classification: 0 หรือ 1)
def map_to_label(val):
    # กลุ่มที่มีความตั้งใจซื้อสูง (Label = 1)
    if val in ['เห็นด้วย', 'เห็นด้วยที่สุด']:
        return 1
    # กลุ่มที่ไม่แน่ใจ หรือไม่มีความตั้งใจซื้อ (Label = 0)
    else:
        return 0

# 5. สร้างคอลัมน์ Target
df['Purchase_Target'] = df[target_col].apply(map_to_label)

# 6. ตรวจสอบความสมดุลของข้อมูล (Class Balance)
print("สรุปจำนวน Label:")
print(df['Purchase_Target'].value_counts())

# แสดงตัวอย่างข้อมูล
print("\nตัวอย่างผลลัพธ์:")
print(df[[target_col, 'Purchase_Target']].head())