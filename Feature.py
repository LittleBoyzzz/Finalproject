import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 0. ตั้งค่าฟอนต์ภาษาไทยให้ Matplotlib
# ==========================================
plt.rcParams['font.family'] = 'Tahoma' # หรือ 'Sarabun' ตามฟอนต์ที่มีในเครื่อง

# ==========================================
# 1. โหลดข้อมูลที่ได้จากข้อ 2 (Target)
# ==========================================
# สมมติว่าไฟล์ที่เตรียมเสร็จแล้วจากข้อ 2 ชื่อไฟล์นี้ (เปลี่ยนชื่อตามจริงได้เลย)
try:
    df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv('Cleaned_Actual_Data_Full.csv', encoding='cp874')

# ตัดคอลัมน์ที่ไม่จำเป็นต่อการหาความสัมพันธ์ออก (เช่น วันที่/เวลา)
# สมมติคอลัมน์แรกเป็น Timestamp
if 'Timestamp' in df.columns or df.columns[0] == 'Timestamp':
    df = df.iloc[:, 1:]

# ==========================================
# 2. เตรียมข้อมูล (Data Preprocessing สำหรับ Apriori)
# ==========================================
# Apriori Algorithm ต้องการข้อมูลในรูปแบบ One-Hot Encoding (True/False หรือ 1/0)
# เราจะแปลงข้อมูลคำตอบในแบบสอบถามให้เป็นคอลัมน์ย่อยๆ
df_encoded = pd.get_dummies(df.astype(str))
df_encoded = df_encoded.astype(bool) # แปลงค่าทั้งหมดเป็น Boolean ตามที่ mlxtend แนะนำ

# ==========================================
# 3. รัน Apriori Algorithm
# ==========================================
# min_support = 0.1 หมายถึง กฎนั้นๆ ต้องมีคนตอบเหมือนกันอย่างน้อย 10% ของคนทั้งหมด
# (คุณสามารถปรับค่า 0.1 ให้มากหรือน้อยลงได้ ตามปริมาณข้อมูล)
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# ==========================================
# 4. สร้างกฎความสัมพันธ์ (Association Rules)
# ==========================================
# min_threshold = 0.6 (Confidence) หมายถึง ความน่าจะเป็นที่จะเกิดผลลัพธ์ต้องเกิน 60%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# เรียงลำดับตามค่า Lift (ยิ่งค่า Lift มากกว่า 1 ยิ่งแปลว่ามีความสัมพันธ์กันมาก)
rules = rules.sort_values(by='lift', ascending=False)

print("\n=== ผลลัพธ์กฎความสัมพันธ์ (Association Rules) Top 10 ===")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# ==========================================
# 5. สร้างกราฟเครือข่าย (Network Graph)
# ==========================================
def draw_network_graph(rules, top_n=15):
    """
    ฟังก์ชันสำหรับวาด Network Graph จากกฎที่ได้
    ดึงมาแค่ top_n กฎแรก เพื่อไม่ให้กราฟดูรกจนเกินไป
    """
    # ตรวจสอบว่ามีกฎเกิดขึ้นหรือไม่
    if rules.empty:
        print("ไม่พบกฎความสัมพันธ์ที่ผ่านเกณฑ์ที่กำหนด ลองปรับ min_support หรือ min_threshold ให้ต่ำลง")
        return

    rules_to_show = rules.head(top_n)
    G = nx.DiGraph()
    
    for idx, row in rules_to_show.iterrows():
        # แปลงข้อมูล Set ให้เป็น String เพื่อแสดงบนกราฟ
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        
        # เพิ่ม Node และเชื่อมเส้นตามความสัมพันธ์
        G.add_edge(antecedent, consequent, weight=row['lift'])
    
    plt.figure(figsize=(14, 10))
    
    # รูปแบบการจัดวาง Node
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    # วาด Node และ Edge
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightgreen', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_family='Tahoma', font_size=10, font_weight='bold')
    
    plt.title(f'Network Graph แสดงกฎความสัมพันธ์ (Top {top_n} Rules)', fontsize=18)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# เรียกใช้ฟังก์ชันวาดกราฟ
draw_network_graph(rules)