import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('demo_result/HDFS_2k.log_structured.csv')

# 数据库和表名
database_name = 'logdatabase'
table_name = 'hdfs_structured'

# 构建清空表格的 SQL 语句
clear_statement = f"DELETE FROM {database_name}.{table_name};\n"

# 构建插入 SQL 语句
insert_statements = [clear_statement]
for index, row in df.iterrows():
    # 格式化日期和时间
    formatted_date = row['Date']
    formatted_time = row['Time']
    
    insert_statement = f"INSERT INTO {database_name}.{table_name} (LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate) VALUES ({row['LineId']}, '{formatted_date}', '{formatted_time}', {row['Pid']}, '{row['Level']}', '{row['Component']}', '{row['Content']}', '{row['EventId']}', '{row['EventTemplate']}');"
    insert_statements.append(insert_statement)

# 将插入语句写入到txt文件
with open('insert_statements.txt', 'w') as file:
    for statement in insert_statements:
        file.write(statement + '\n')

print("SQL插入语句已保存到 insert_statements.txt 文件中。")
