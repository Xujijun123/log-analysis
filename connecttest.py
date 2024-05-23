import pymysql

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

# 执行SQL查询
query = "SELECT * FROM systemadmins"
cursor.execute(query)

# 获取所有结果
results = cursor.fetchall()

# 打印结果
for row in results:
    print(row)


# 执行SQL查询
query = "SELECT * FROM operator"
cursor.execute(query)

# 获取所有结果
results = cursor.fetchall()

# 打印结果
for row in results:
    print(row)

# 关闭游标和数据库连接
cursor.close()
db.close()