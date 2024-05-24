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


cursor.close()
db.close()