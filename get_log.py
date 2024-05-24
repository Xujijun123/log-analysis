from flask import Flask, render_template, request
import pandas as pd
import pymysql

app = Flask(__name__)

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '0210070029Xu',
    'database': 'logdatabase',
    'port': 3306,
    'ssl': {'ssl': {}}
}


def fetch_logs(page, per_page):
    try:
        # 连接数据库
        connection = pymysql.connect(**db_config)

        # 查询数据
        sql = 'SELECT * FROM hdfs_structured'
        df = pd.read_sql(sql, connection)

        # 关闭数据库连接
        connection.close()

        # 计算分页数据
        total = len(df)
        start = (page - 1) * per_page
        end = start + per_page
        logs = df.iloc[start:end]

        return logs, total

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, 0


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/logs')
def view_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    logs_df, total = fetch_logs(page, per_page)
    total_pages = (total + per_page - 1) // per_page
    return render_template('logs.html', logs=logs_df.to_html(), page=page, total_pages=total_pages)


if __name__ == '__main__':
    app.run(debug=True)
