import pymysql
import sqlite3
import pandas as pd
db = pymysql.connect(
    host="localhost",
    user="root", # 数据库用户名
    password="0210070029Xu", # 数据库密码
    database="logdatabase",
    port=3306,
    ssl={'ssl': {}}
)

# 创建游标对象
cursor = db.cursor()

# 读取CSV文件
csv_file_path = r'D:\DeskTop\logparser-master\data\HDFS\HDFS_2k.log_structured.csv'
df = pd.read_csv(csv_file_path)

add_log = ("INSERT INTO hdfs_structured "
               "(id, date, time, pid, level, component, content, eventid, EventTemplate) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")

try:
    for _, row in df.iterrows():
        log_data = (
            row['LineId'], row['Date'], row['Time'], row['Pid'], row['Level'],
            row['Component'], row['Content'], row['EventId'], row['EventTemplate']
        )
        cursor.execute(add_log, log_data)
    db.commit()
except Exception as e:
    print("An error occurred:", e)
    db.rollback()  # 回滚事务，撤销之前的操作

cursor.close()
db.close()