from flask import Flask, render_template, request, redirect, url_for
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
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 这里可以添加验证用户名和密码的逻辑，例如查询数据库进行验证
        # 如果验证成功，可以设置用户登录状态，并重定向到操作员主页
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        sql = "SELECT * FROM operator WHERE username=%s AND password=%s"
        cursor.execute(sql, (username, password))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        if user:
            # 这里设置用户登录状态，可以使用 session 或者其他方式来保存登录状态
            return redirect(url_for('operator_mainpage'))
        else:
            # 如果验证失败，可以返回登录页面并显示错误消息
            error_message = 'Invalid username or password. Please try again.'
            return render_template('login.html', error_message=error_message)
    else:
        return render_template('login.html')
@app.route('/operator_mainpage')
def operator_mainpage():
    return render_template('operator_mainpage.html')

@app.route('/logs')
def view_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    logs_df, total = fetch_logs(page, per_page)
    total_pages = (total + per_page - 1) // per_page
    return render_template('logs.html', logs=logs_df.to_html(), page=page, total_pages=total_pages)


if __name__ == '__main__':
    app.run(debug=True)
