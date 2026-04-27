import pandas as pd


df = pd.read_csv("BU Data from Survey Cases_final(5).csv", encoding='utf-8-sig', skiprows=1)


df = df[df.iloc[:, 1] == 'เคย']


target_col = 'Option 6 คุณคิดเห็นอย่างไรกับการออกแบบบรรจุภัณฑ์นี้  [รู้สึกอยากซื้อสินค้า]'


def map_to_label(val):
    if pd.isna(val): return 0
    if val == 'เห็นด้วย' or val == 'เห็นด้วยที่สุด':
        return 1
    else:
        return 0

df['Purchase_Target'] = df[target_col].apply(map_to_label)


print("สรุปจำนวน Label:")
print(df['Purchase_Target'].value_counts())

print("\nตรวจสอบข้อมูล 5 แถวแรก:")
print(df[[target_col, 'Purchase_Target']].head())