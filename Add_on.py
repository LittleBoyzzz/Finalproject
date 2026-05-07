import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.rcParams['font.family'] = 'Tahoma' 
plt.rcParams['axes.unicode_minus'] = False 

df_clean = pd.read_csv('Cleaned_Actual_Data_Full.csv')
df_train = pd.read_csv('Model_Ready_Train_Data_Selected.csv')


# กราฟที่ 1


top3_col = 'จากตัวเลือกทั้งหมด คุณชอบการออกแบบบรรจุภัณฑ์อาหารแมวสำเร็จรูปแบบใดมากที่สุด 3 อันดับแรก'
options_count = {f'Option {i}': 0 for i in range(1, 11)}

for row in df_clean[top3_col].dropna():
    for i in range(1, 11):
        if re.search(rf'Option {i}\b', row):
            options_count[f'Option {i}'] += 1

df_top3 = pd.DataFrame(list(options_count.items()), columns=['Option', 'Count']).sort_values('Count', ascending=False)

plt.figure(figsize=(10, 6))
colors1 = ['#2ca02c' if opt == 'Option 6' else '#d3d3d3' for opt in df_top3['Option']]
bars1 = plt.bar(df_top3['Option'], df_top3['Count'], color=colors1)

plt.title('กราฟที่ 1: ออปชั่น 6 เป็น Top 2 ที่คนชอบที่สุด (ทิ้งห่างออปชั่น 9, 10)', fontsize=14, fontweight='bold')
plt.ylabel('จำนวนครั้งที่ติด Top 3', fontsize=12)
plt.xlabel('รูปแบบบรรจุภัณฑ์', fontsize=12)
plt.xticks(rotation=45)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# กราฟที่ 2


corr_matrix = df_train.corr()
target_corr = corr_matrix['Purchase_Target_Encoded'].drop('Purchase_Target_Encoded').sort_values(ascending=True).tail(10)

plt.figure(figsize=(10, 6))
colors2 = ['#2ca02c' if 'Option 6' in col else '#d3d3d3' for col in target_corr.index]
bars2 = plt.barh(target_corr.index, target_corr.values, color=colors2)

plt.title('กราฟที่ 2: ออปชั่น 6 คือตัวแปรหลักที่ทำให้ลูกค้า "ตัดสินใจซื้อ" สูงที่สุด', fontsize=14, fontweight='bold')
plt.xlabel('ค่าความสัมพันธ์ (Correlation with Purchase Target)', fontsize=12)

for bar in bars2:
    xval = bar.get_width()
    plt.text(xval + 0.01, bar.get_y() + bar.get_height()/2, f'{xval:.2f}', ha='left', va='center')

plt.tight_layout()
plt.show()


# กราฟที่ 3


target_options = [2, 3, 6]
option_cols = [c for c in df_clean.columns if 'Option' in c and 'คุณคิดเห็น' in c]

results = {}
for i in target_options:
    cols = [c for c in option_cols if c.startswith(f'Option {i} ') or c.startswith(f'Option {i}ค') or c.startswith(f'Option {i}[')]
    scores = {}
    for c in cols:
        attr_match = re.search(r'\[(.*?)\]', c)
        if attr_match:
            short_name = attr_match.group(1).replace('รู้สึก', '').replace('รูู้สึก', '')
            scores[short_name] = df_clean[c].mean()
    results[f'Option {i}'] = scores

df_scores = pd.DataFrame(results).T
categories = df_scores.columns.tolist()

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

colors_dict = {'Option 2': '#1f77b4', 'Option 3': '#ff7f0e', 'Option 6': '#2ca02c'}

for opt in df_scores.index:
    values = df_scores.loc[opt].values.tolist()
    values += values[:1]
    
    if opt == 'Option 6':
        ax.plot(angles, values, color=colors_dict[opt], linewidth=3, label=f'{opt} (The Converter)')
        ax.fill(angles, values, color=colors_dict[opt], alpha=0.25)
    elif opt == 'Option 3':
        ax.plot(angles, values, color=colors_dict[opt], linewidth=2, linestyle='--', label=f'{opt} (Crowd Favorite)')
    else:
        ax.plot(angles, values, color=colors_dict[opt], linewidth=2, linestyle=':', label=opt)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
ax.set_ylim(2.8, 4.2)

plt.title('กราฟเจาะลึก: Option 6 vs ตัวท็อป (ทำไมคะแนนความชอบ Option 3 สูงกว่า แต่เราถึงเลือก 6?)', fontsize=14, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()
plt.show()


# กราฟที่ 4


opt3_buy = 'Option 3 คุณคิดเห็นอย่างไรกับการออกแบบบรรจุภัณฑ์นี้  [รู้สึกอยากซื้อสินค้า]'
opt6_buy = 'Option 6 คุณคิดเห็นอย่างไรกับการออกแบบบรรจุภัณฑ์นี้  [รู้สึกอยากซื้อสินค้า]'

grouped_data = df_clean.groupby('Purchase_Target')[[opt3_buy, opt6_buy]].mean()

labels = ['กลุ่มคนที่ไม่ซื้อสินค้า\n(Non-Buyers)', 'กลุ่มคนที่ซื้อสินค้าจริง!\n(Actual Buyers)']
opt3_scores = grouped_data[opt3_buy].values
opt6_scores = grouped_data[opt6_buy].values

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, opt3_scores, width, label='Option 3 (ขวัญใจมหาชน)', color='#ff7f0e')
rects2 = ax.bar(x + width/2, opt6_scores, width, label='Option 6 (ตัวปิดการขาย)', color='#2ca02c')

ax.set_ylabel('คะแนนเฉลี่ย "ความรู้สึกอยากซื้อ" (เต็ม 5)', fontsize=12)
ax.set_title('กราฟหมัดน็อค: Option 6 ชนะ Option 3 ขาดลอย ในกลุ่ม "ลูกค้าที่ยอมจ่ายเงินจริง"', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 5.0)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.axhline(y=4.0, color='gray', linestyle='--', alpha=0.5) 
plt.tight_layout()
plt.show()