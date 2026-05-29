"""初始化本地 SQLite —— 只导 AI 助手需要的表"""
import sqlite3
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "cloudemall.db")

SQL = """
CREATE TABLE IF NOT EXISTS pos_category (
    category_id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL DEFAULT 0,
    category_name TEXT
);

CREATE TABLE IF NOT EXISTS pos_product (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_sn TEXT NOT NULL,
    product_name TEXT NOT NULL,
    product_description TEXT NOT NULL,
    price REAL NOT NULL,
    product_category_id INTEGER NOT NULL,
    image_url TEXT,
    detail_url TEXT,
    stock INTEGER DEFAULT 100
);

INSERT INTO pos_category VALUES (1,0,'手机数码');
INSERT INTO pos_category VALUES (2,0,'电脑办公');
INSERT INTO pos_category VALUES (3,0,'生活百货');
INSERT INTO pos_category VALUES (4,0,'图书文具');
INSERT INTO pos_category VALUES (11,1,'手机通讯');
INSERT INTO pos_category VALUES (12,1,'手机配件');
INSERT INTO pos_category VALUES (21,2,'电脑整机');
INSERT INTO pos_category VALUES (22,2,'电脑外设');
INSERT INTO pos_category VALUES (31,3,'休闲零食');
INSERT INTO pos_category VALUES (32,3,'个人护理');
INSERT INTO pos_category VALUES (41,4,'教材教辅');
INSERT INTO pos_category VALUES (42,4,'办公文具');
INSERT INTO pos_category VALUES (111,11,'5G手机');
INSERT INTO pos_category VALUES (112,11,'游戏手机');
INSERT INTO pos_category VALUES (113,11,'拍照手机');
INSERT INTO pos_category VALUES (121,12,'手机壳');
INSERT INTO pos_category VALUES (122,12,'数据线/充电器');
INSERT INTO pos_category VALUES (123,12,'移动电源');
INSERT INTO pos_category VALUES (211,21,'轻薄本');
INSERT INTO pos_category VALUES (212,21,'游戏本');
INSERT INTO pos_category VALUES (213,21,'平板电脑');
INSERT INTO pos_category VALUES (221,22,'鼠标');
INSERT INTO pos_category VALUES (222,22,'键盘');
INSERT INTO pos_category VALUES (223,22,'显示器');
INSERT INTO pos_category VALUES (311,31,'坚果炒货');
INSERT INTO pos_category VALUES (312,31,'饼干蛋糕');
INSERT INTO pos_category VALUES (321,32,'洗发护发');
INSERT INTO pos_category VALUES (322,32,'纸品湿巾');
INSERT INTO pos_category VALUES (411,41,'考研复习');
INSERT INTO pos_category VALUES (412,41,'英语四六级');
INSERT INTO pos_category VALUES (421,42,'笔类');
INSERT INTO pos_category VALUES (422,42,'笔记本/手账');

INSERT INTO pos_product VALUES (1,'1001','黑耀系列钢笔礼盒','商品限购50件',169.0,421,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/62389fff-7a64-444a-aa66-78b1fd583e32.png','',100);
INSERT INTO pos_product VALUES (2,'1002','广博A5/96张皮面记事本 雅典黑 1本','耐磨防水，精选PU皮',12.9,422,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/440602d9-ba5f-42f8-9d9d-ba3300ff9aba.jpeg','',100);
INSERT INTO pos_product VALUES (5,'1005','甄沐海岸松无硅油氨基酸控油蓬松洗发水300ml','控油清爽',38.0,321,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/0a1918d9-a297-4e4f-b998-d7e58021d7df.jpeg','',100);
INSERT INTO pos_product VALUES (7,'1007','Mate X5','超轻薄四曲折叠',13499.0,111,'https://res6.vmallres.com/pimages//uomcdn/CN/pms/202309/gbom/6942103107320/800_800_959526DD397D0C873FCE80CE67C9A0BFmp.png','https://www.vmall.com/product/comdetail/index.html?prdId=10086281788718&sbomCode=2601010457506',99);
INSERT INTO pos_product VALUES (8,'1008','Pura 70 Pro','超聚光微距长焦',6499.0,113,'https://res2.vmallres.com/pimages//uomcdn/CN/pms/202404/gbom/6942103119071/800_800_AE94E48F4A6370D6E956B4E722588A5Amp.png','https://www.vmall.com/product/comdetail/index.html?prdId=10086821546239&sbomCode=2601010486504',100);
INSERT INTO pos_product VALUES (19,'P111001','小米14 Pro 钛金属版','骁龙8Gen3 徕卡光学镜头 120W秒充',4999.0,111,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/5959b1f8-3133-4ba4-b5ad-98f03065973f.jpg','',100);
INSERT INTO pos_product VALUES (20,'P111002','iPhone 15 Pro Max','钛金属边框 A17 Pro芯片',9999.0,111,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/d071f425-0c39-430a-a734-ee65a65c4c3f.jpg','',50);
INSERT INTO pos_product VALUES (21,'P111003','华为 Mate 60 Pro','卫星通话 昆仑玻璃 玄武架构',6999.0,111,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/2301f0bf-b2d9-4e1e-ba15-4387fd1cec0c.jpg','',30);
INSERT INTO pos_product VALUES (23,'P112001','红魔9 Pro','第三代骁龙8 ICE 13.0魔冷散热',4399.0,112,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/7c4aad19-33a4-469b-9765-2dc82065a302.png','',100);
INSERT INTO pos_product VALUES (24,'P112002','ROG 8 Pro','专业电竞手机 165Hz高刷',5299.0,112,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/9297d125-7707-43d7-90d8-8f882250a222.jpg','',60);
INSERT INTO pos_product VALUES (25,'P113001','vivo X100 Pro','蔡司APO超级长焦',4999.0,113,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/ae7c0ba1-8c1b-4335-99f7-26f32a8eff27.png','',100);
INSERT INTO pos_product VALUES (26,'P113002','OPPO Find X7 Ultra','双潜望四主摄 哈苏大师影像',5999.0,113,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/d7e6da45-0911-49f5-9833-5e364d28aaf0.jpeg','',80);
INSERT INTO pos_product VALUES (27,'P121001','iPhone 15 液态硅胶壳','官方同款 手感亲肤',29.9,121,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/3b4ba240-524a-4812-8ff5-89f5dd6bba87.jpeg','',500);
INSERT INTO pos_product VALUES (28,'P121002','华为 Mate60 素皮壳','商务风格',39.9,121,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/aa61eeb6-8ccb-479f-a30b-281b5e716fdb.jpeg','',300);
INSERT INTO pos_product VALUES (29,'P122001','Anker 苹果PD快充线','MFi认证 亲肤材质',49.0,122,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/2381a35b-387d-4942-9676-27cb9ce4d9ba.jpeg','',200);
INSERT INTO pos_product VALUES (30,'P122002','倍思 Type-C 数据线','100W快充 编织线身',19.9,122,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/da4514d1-c227-4817-aeea-b84bc3c4c930.jpeg','',500);
INSERT INTO pos_product VALUES (31,'P123001','小米移动电源3','20000mAh 大容量',79.0,123,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/f6afd2d3-d0be-48e8-9e89-9e7da87a09da.jpeg','',200);
INSERT INTO pos_product VALUES (32,'P123002','罗马仕 充电宝','20000mAh 小巧便携',59.0,123,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/02b34e38-5dc6-4c69-8a62-ce13eb263501.jpeg','',300);
INSERT INTO pos_product VALUES (33,'P211001','MacBook Air M2','M2芯片 极致轻薄',8999.0,211,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/02e6bdfa-9cf9-4954-92a4-bddb464e0545.jpeg','',20);
INSERT INTO pos_product VALUES (34,'P211002','联想小新Pro 14','i5-13500H 2.8K 120Hz',5499.0,211,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/f8d4fc5d-9d19-4e4c-9979-669304b15173.jpeg','',100);
INSERT INTO pos_product VALUES (35,'P211003','ThinkBook 14+','全能轻薄本',5299.0,211,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/8cb228cd-6f4e-432b-b866-cb18eb881e24.jpeg','',80);
INSERT INTO pos_product VALUES (36,'P212001','联想拯救者 R9000P','RTX4060 满血显卡',8499.0,212,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/a1220952-9944-4af2-9d5f-575774c32575.jpeg','',50);
INSERT INTO pos_product VALUES (37,'P212002','惠普暗影精灵9','i9-13900HX 强悍性能',7999.0,212,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/824be602-a1a5-457a-9f7f-abe50447943d.jpeg','',60);
INSERT INTO pos_product VALUES (38,'P213001','iPad Air 5','M1芯片 全面屏设计',4399.0,213,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/b884dbcb-0346-48db-b906-96c6dff04c0a.jpeg','',100);
INSERT INTO pos_product VALUES (39,'P213002','小米平板 6 Pro','骁龙8+ 11英寸2.8K',2499.0,213,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/be9ec805-35e3-460b-96a2-847a44a689d2.jpeg','',150);
INSERT INTO pos_product VALUES (40,'P221001','罗技 MX Master 3S','人体工学 静音滚轮',799.0,221,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/c326237e-4dd5-477e-914b-5d528a0fb297.jpeg','',50);
INSERT INTO pos_product VALUES (41,'P221002','雷蛇 毒蝰 V3','轻量化设计 30K传感器',399.0,221,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/b389efed-4d5e-4229-ae0f-c2b5d0d4930d.jpeg','',100);
INSERT INTO pos_product VALUES (42,'P222001','Keychron K2 机械键盘','蓝牙双模 红轴',498.0,222,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/04f9db81-d125-4898-a1e9-256bfd8cdb75.jpeg','',80);
INSERT INTO pos_product VALUES (43,'P222002','罗技 K380 蓝牙键盘','多设备切换 小巧便携',149.0,222,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/6627ef48-9bdb-4b82-a0b8-ccb6abcee7a5.jpeg','',200);
INSERT INTO pos_product VALUES (44,'P223001','戴尔 U2723QE','4K显示器 IPS Black',3499.0,223,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/4abded32-3f9c-448a-8256-42a32ed1732a.jpeg','',30);
INSERT INTO pos_product VALUES (45,'P223002','AOC 27英寸 2K','高性价比 75Hz',999.0,223,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/0b4f49de-246c-449b-a026-841bfda056a0.jpeg','',100);
INSERT INTO pos_product VALUES (46,'P311001','三只松鼠 每日坚果','30袋装 孕妇零食大礼包',139.0,311,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/9d29f843-c9ba-4f44-b7cc-5219691ac4b1.jpeg','',200);
INSERT INTO pos_product VALUES (47,'P311002','百草味 夏威夷果','奶油味 200g/袋',29.9,311,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/2d739828-98ea-4c75-a337-8df05c09661c.jpeg','',300);
INSERT INTO pos_product VALUES (48,'P312001','奥利奥 夹心饼干','原味+草莓味',12.5,312,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/93d67066-1bac-4c31-91c1-23cdff177ae2.jpeg','',500);
INSERT INTO pos_product VALUES (49,'P312002','好丽友 派','巧克力味 12枚入',19.9,312,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/cc8aa772-b6e2-43ae-9072-672ac17a5362.jpeg','',200);
INSERT INTO pos_product VALUES (50,'P321001','海飞丝 去屑洗发水','柠檬清爽型 750ml',39.9,321,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/3121ea0f-9993-4391-91ab-07ffa22ee04f.jpeg','',200);
INSERT INTO pos_product VALUES (51,'P321002','潘婷 护发素','3分钟奇迹奢护',29.9,321,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/386aa5bb-a810-4dab-ae59-9f89cd9f4e7f.jpeg','',150);
INSERT INTO pos_product VALUES (52,'P322001','维达 抽纸','4层140抽*24包',59.9,322,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/cd59d768-615b-4f52-b06f-ad7e914a4434.jpeg','',200);
INSERT INTO pos_product VALUES (53,'P322002','洁柔 Face面巾纸','古龙水香 4层加厚',19.9,322,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/3e9ec6ca-7fd9-4f2d-a9e5-fcea90c60488.jpeg','',300);
INSERT INTO pos_product VALUES (54,'P411001','肖秀荣 考研政治1000题','2026考研必备',45.0,411,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/1349d839-f9de-464d-aa03-3faa3bf0749d.jpeg','',300);
INSERT INTO pos_product VALUES (55,'P411002','张宇 高数18讲','基础强化一本通',58.0,411,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/0e0968a0-a87b-444c-888b-93bce25050c3.jpeg','',200);
INSERT INTO pos_product VALUES (56,'P412001','星火英语 四级真题','备考2026年6月',39.0,412,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/33054b13-55ed-4d00-ab2d-9c0d476f2cab.jpeg','',500);
INSERT INTO pos_product VALUES (57,'P412002','新东方 恋练有词','考研英语词汇',42.0,412,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/a1bc1fa5-b28f-4b20-b968-dcb4b476c8a0.jpeg','',300);
INSERT INTO pos_product VALUES (58,'P421001','得力 按动中性笔','0.5mm 黑色 12支/盒',9.9,421,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/e36305c2-ca57-43d0-86b2-67eb089fa294.jpeg','',1000);
INSERT INTO pos_product VALUES (59,'P421002','百乐 P500 签字笔','考试专用 0.5mm',7.5,421,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/740628df-2d14-4d1a-8f7c-a646f41b0cce.jpeg','',500);
INSERT INTO pos_product VALUES (60,'P422001','广博 错题本','B5大号 康奈尔笔记法',5.0,422,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/71bc0060-4c98-4ac4-9a2f-7429aaf3a724.jpeg','',200);
INSERT INTO pos_product VALUES (61,'P422002','国誉活页本','B5 26孔 超薄便携',15.0,422,'https://scau-mis-images.oss-cn-shenzhen.aliyuncs.com/5240d356-d03d-462d-b338-f050b6c21128.jpeg','',150);
"""

def build():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    conn.executescript(SQL)
    conn.commit()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pos_product")
    p = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM pos_category")
    c = cur.fetchone()[0]
    conn.close()
    print(f"Done: {p} products, {c} categories -> {DB_FILE}")

if __name__ == "__main__":
    build()
