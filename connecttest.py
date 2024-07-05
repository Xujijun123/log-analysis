import pymysql
import sqlite3
import pandas as pd
try:
    db = pymysql.connect(
        host="localhost",
        user="root", # 数据库用户名
        password="2002119Li.", # 数据库密码
        database="logdatabase",
        port=3306,
        ssl={'ssl': {}}
    )
    print("数据库连接成功")
    # 创建游标对象
    cursor = db.cursor()


    cursor.close()
    db.close()

except pymysql.MySQLError as e:
    print(f"数据库连接失败，错误信息：{e}")