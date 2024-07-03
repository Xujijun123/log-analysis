import os
import subprocess

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from functools import wraps
app = Flask(__name__)
app.secret_key = '123'  # 用于加密session数据
app.config['SESSION_TYPE'] = 'filesystem'  # 会话类型为文件系统
app.config['UPLOAD_FOLDER'] = 'uploads/'
# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '0210070029Xu',
    'database': 'logdatabase',
    'port': 3306,
    'ssl': {'ssl': {}}
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login', error='请先登录'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/search_logs', methods=['GET'])
def search_logs():
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    fields = ['LineId', 'Date', 'Time', 'Pid', 'Level', 'Component', 'Content', 'EventId', 'EventTemplate']
    query = "SELECT * FROM hdfs_structured WHERE 1=1"
    params = []
    
    for field in fields:
        value = request.args.get(field)
        if value:
            if field in ['Date', 'Time', 'Level', 'Component', 'EventId', 'EventTemplate']:
                query += f" AND {field} = %s"
            elif field in ['LineId', 'Pid']:
                query += f" AND {field} = %s"
            else:
                query += f" AND {field} LIKE %s"
                value = f"%{value}%"  # 添加通配符以进行模糊匹配
            params.append(value)

    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # 为分页添加 LIMIT 和 OFFSET 子句
    limit_offset_clause = f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
    cursor.execute(query + limit_offset_clause, params)
    logs = cursor.fetchall()

    # 查询总记录数以计算总页数
    cursor.execute(f"SELECT COUNT(*) FROM ({query}) as count_table", params)
    total_logs = cursor.fetchone()[0]
    total_pages = (total_logs + per_page - 1) // per_page

    cursor.close()
    connection.close()

    # 转换 logs 为字典列表，以便在 Jinja 模板中使用
    columns = fields
    logs = [dict(zip(columns, log)) for log in logs]

    return render_template('search_logs.html', logs=logs, page=page, total_pages=total_pages)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        repassword = request.form['repassword']
        email = request.form['email']
        role = request.form['role']
        code = request.form['code'] if 'code' in request.form else None
        
        # 验证两次输入的密码是否一致
        if password != repassword:
            return redirect(url_for('register', error='两次密码不一致'))
        
        # 验证系统管理员的验证码
        code_list = ['123456','654321']
        if role == 'systemadmin' and code not in code_list:
            return redirect(url_for('register', error='验证码错误'))
        
        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()
            
            # 判断用户名和邮箱是否已被注册
            table = 'operator' if role == 'operator' else 'systemadmins'
            sql = f"SELECT * FROM {table} WHERE username = %s OR email = %s"
            cursor.execute(sql, (username, email))
            result = cursor.fetchone()
            if result:
                if result[1] == username:
                    return redirect(url_for('register', error='该用户名已被注册'))
                if result[3] == email:  # Assuming email is the 4th column in your table
                    return redirect(url_for('register', error='该邮箱已被注册'))
                return redirect(url_for('register'))
            
            # 向数据库中插入新用户
            if role == 'operator':
                sql = "INSERT INTO operator (username, password, email, 注销状况) VALUES (%s, %s, %s, %s)"
                cursor.execute(sql, (username, password, email, False))
            else:
                sql = "INSERT INTO systemadmins (username, password, email, code, 注销状况) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(sql, (username, password, email, code, False))
            connection.commit()

            return redirect(url_for('login', success='注册成功，请登录'))
        
        except pymysql.MySQLError as e:
            return redirect(url_for('register', error=f'数据库错误：{e}'))
        
        finally:
            cursor.close()
            connection.close()
    
    error = request.args.get('error')
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()
            
            # 根据角色选择相应的表
            table = 'operator' if role == 'operator' else 'systemadmins'
            sql = f"SELECT * FROM {table} WHERE username = %s AND password = %s"
            cursor.execute(sql, (username, password))
            user = cursor.fetchone()
            
            cursor.close()
            connection.close()
            
            if user:
                session['username'] = username  # 将用户名存储到session
                if role == 'operator':
                    return redirect(url_for('operator_mainpage'))
                else:
                    return redirect(url_for('admin_mainpage'))
            else:
                return redirect(url_for('login', error='用户名或密码错误'))
        
        except pymysql.MySQLError as e:
            return redirect(url_for('login', error=f'数据库错误：{e}'))
    
    error = request.args.get('error')
    success = request.args.get('success')
    return render_template('login.html', error=error, success=success)
    
@app.route('/operator_mainpage')
def operator_mainpage():
    return render_template('operator_mainpage.html')

@app.route('/admin_mainpage')
def admin_mainpage():
    return render_template('admin_mainpage.html')
def fetch_logs(page, per_page, log_type):
    try:
        # 连接数据库
        connection = pymysql.connect(**db_config)

        # 查询数据
        table_name = f'{log_type}_structured'
        sql = f'SELECT * FROM {table_name}'
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

@app.route('/logs')
def view_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    log_type = request.args.get('log_type', 'hdfs')  # 默认为HDFS日志
    logs_df, total = fetch_logs(page, per_page, log_type)
    total_pages = (total + per_page - 1) // per_page
    return render_template('logs.html', logs=logs_df.to_html(classes='log-table', index=False), page=page,
                           total_pages=total_pages, log_type=log_type)


@app.route('/log_analysis')
def log_analysis():
    return render_template('log_analysis.html')
@app.route('/log_warning')
def log_warning():
    return render_template('log_warning.html')



@app.route('/event_frequency')
def event_frequency():
    return render_template('event_frequency.html')

@app.route('/time_series')
def time_series():
    return render_template('time_series.html')

@app.route('/log_anomaly')
def log_anomaly():
    return render_template('log_anomaly.html')

@app.route('/upload_log_files', methods=['POST', 'GET'])
def upload_log_files():
    if request.method == 'POST':
        if 'file' not in request.files or 'anomalyfile' not in request.files:
            return redirect(request.url)
        log_file = request.files['file']
        anomaly_file = request.files['anomalyfile']

        if log_file.filename == '' or anomaly_file.filename == '':
            return redirect(request.url)

        if log_file and anomaly_file:
            log_path = os.path.join(app.config['UPLOAD_FOLDER'], log_file.filename)
            anomaly_path = os.path.join(app.config['UPLOAD_FOLDER'], anomaly_file.filename)

            log_file.save(log_path)
            anomaly_file.save(anomaly_path)

            #subprocess.run(['python', 'model/log-anomaly/parse/test_parser.py', log_path])

            structured_log_path = os.path.join(app.config['UPLOAD_FOLDER'], 'HDFS.log_structured.csv')

            #subprocess.run(['python', 'model/log-anomaly/process/test_processor.py', structured_log_path, anomaly_path])

            npy_file = os.path.join(app.config['UPLOAD_FOLDER'], 'x_test_tf-idf_v5.npy')
            csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'y_test_tf-idf_v5.csv')
            result=subprocess.run(['python', 'model/log-anomaly/model/test.py', npy_file, csv_file],
                           capture_output=True,text=True)
            # 将输出内容传递给模板

            anomalous_lines_path = os.path.join(app.config['UPLOAD_FOLDER'], 'anomalous_lines.csv')
            if os.path.exists(anomalous_lines_path):
                anomalous_lines = pd.read_csv(anomalous_lines_path)
                with open(log_path, 'r') as file:
                    log_lines = file.readlines()
                anomalous_log_lines = [log_lines[i] for i in anomalous_lines['LineId']]
                anomalous_lines_id = anomalous_lines['LineId'].tolist()
                # Combine line IDs and log lines into tuples
                anomalous_log_data = list(zip(anomalous_lines_id, anomalous_log_lines))
            else:
                anomalous_log_data = [("No anomalies detected", "")]
            return render_template('log_anomaly.html', output=result.stdout, anomalous_log_data=anomalous_log_data)

        return 'Failed to upload files'
    elif request.method == 'GET':
        # 处理GET请求的逻辑，比如返回上传表单页面
        return render_template('log_anomaly.html')
@app.route("/level_time")
def level_time():
    return render_template("level_time.html")

@app.route("/event_time")
def event_time():
    return render_template("event_time.html")

@app.route('/manage_operators/view', methods=['GET'])
def view_operators():
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        sql = "SELECT id, username, email FROM operator"
        cursor.execute(sql)
        operators = cursor.fetchall()
        cursor.close()
        connection.close()
        return render_template('view_operators.html', operators=operators)
    except pymysql.MySQLError as e:
        return render_template('admin_mainpage', error=f'数据库错误：{e}')
    
@app.route('/manage_operators/edit', methods=['GET', 'POST'])
def edit_operator():
    if request.method == 'POST':
        operator_id = request.form['operator_id']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()
            sql = "UPDATE operator SET username = %s, email = %s, password = %s WHERE id = %s"
            cursor.execute(sql, (username, email, password, operator_id))
            connection.commit()
            cursor.close()
            connection.close()
            return redirect(url_for('view_operators'))
        except pymysql.MySQLError as e:
            return render_template('edit_operator.html', error=f'数据库错误：{e}')
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        sql = "SELECT id, username, email FROM operator"
        cursor.execute(sql)
        operators = cursor.fetchall()
        cursor.close()
        connection.close()
        return render_template('edit_operator.html', operators=operators)
    except pymysql.MySQLError as e:
        return render_template('admin_mainpage', error=f'数据库错误：{e}')


@app.route('/manage_operators/add', methods=['GET', 'POST'])
def add_operator():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()
            
            # 检查用户名或邮箱是否已经存在
            sql = "SELECT * FROM operator WHERE username = %s OR email = %s"
            cursor.execute(sql, (username, email))
            result = cursor.fetchone()
            if result:
                cursor.close()
                connection.close()
                return redirect(url_for('add_operator', error='用户名或邮箱已存在'))
            
            # 插入新运维人员
            sql = "INSERT INTO operator (username, password, email, 注销状况) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, (username, password, email, False))
            connection.commit()
            cursor.close()
            connection.close()
            return redirect(url_for('view_operators'))
        except pymysql.MySQLError as e:
            return redirect(url_for('add_operator', error=f'数据库错误：{e}'))
    error = request.args.get('error')
    return render_template('add_operator.html', error=error)


@app.route('/manage_operators/delete', methods=['GET', 'POST'])
def delete_operator():
    if request.method == 'POST':
        operator_id = request.form['operator_id']
        try:
            connection = pymysql.connect(**db_config)
            cursor = connection.cursor()
            sql = "DELETE FROM operator WHERE id = %s"
            cursor.execute(sql, (operator_id,))
            connection.commit()
            cursor.close()
            connection.close()
            return redirect(url_for('view_operators'))
        except pymysql.MySQLError as e:
            return render_template('delete_operator.html', error=f'数据库错误：{e}')
    try:
        connection = pymysql.connect(**db_config)
        cursor = connection.cursor()
        sql = "SELECT id, username FROM operator"
        cursor.execute(sql)
        operators = cursor.fetchall()
        cursor.close()
        connection.close()
        return render_template('delete_operator.html', operators=operators)
    except pymysql.MySQLError as e:
        return render_template('admin_mainpage', error=f'数据库错误：{e}')



@app.route("/api/event_frequency", methods=["GET"])
def get_event_frequency():
    log_source = request.args.get("log_source", "hdfs")
    table_name = f"{log_source}_structured"
    try:
        query = f"""
        SELECT EventId, COUNT(*) as Frequency
        FROM {table_name}
        WHERE EventId IS NOT NULL
        GROUP BY EventId
        ORDER BY Frequency DESC
        """
        df = pd.read_sql(query, engine)

        event_frequency_data = df.to_dict(orient="records")
        return jsonify(event_frequency_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 创建数据库连接字符串
db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# 创建SQLAlchemy引擎
engine = create_engine(db_url)


@app.route("/api/logs_by_event/<event_id>", methods=["GET"])
def get_logs_by_event(event_id):
    log_source = request.args.get("log_source", "hdfs")
    table_name = f"{log_source}_structured"
    try:
        query = f"""
        SELECT LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate
        FROM {table_name}
        WHERE EventId = %(event_id)s
        """
        df = pd.read_sql(query, engine, params={"event_id": event_id})

        # Convert Timedelta to string
        df["Time"] = df["Time"].astype(str)

        logs_data = df.to_dict(orient="records")
        return jsonify(logs_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/time_series_data", methods=["GET"])
def get_time_series_data():
    log_source = request.args.get("log_source", "hdfs")
    table_name = f"{log_source}_structured"
    try:
        query = f"""
        SELECT DATE(Date) as date, COUNT(*) as count
        FROM {table_name}
        GROUP BY DATE(Date)
        ORDER BY DATE(Date)
        """

        df = pd.read_sql(query, engine)

        time_series_data = df.to_dict(orient="records")
        for entry in time_series_data:
            entry["date"] = entry["date"].strftime("%Y-%m-%d")

        return jsonify(time_series_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs_by_date/<date>", methods=["GET"])
def get_logs_by_date(date):
    log_source = request.args.get("log_source", "hdfs")
    table_name = f"{log_source}_structured"
    try:
        query = f"""
        SELECT HOUR(Time) as hour, COUNT(*) as count
        FROM {table_name}
        WHERE DATE(Date) = %(date)s
        GROUP BY HOUR(Time)
        ORDER BY HOUR(Time)
        """
        df = pd.read_sql(query, engine, params={"date": date})

        logs_by_date = df.to_dict(orient="records")
        for entry in logs_by_date:
            entry["hour"] = f"{entry['hour']:02}:00"

        return jsonify(logs_by_date)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/time_series_by_event", methods=["GET"])
def get_time_series_by_event():
    log_source = request.args.get("log_source", "hdfs")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    table_name = f"{log_source}_structured"
    print(f"Received request for time series data by event from log source: {log_source}, between {start_date} and {end_date}")
    
    try:
        query = f"""
        SELECT EventId, Date, COUNT(*) as value
        FROM {table_name}
        WHERE Date BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY EventId, Date
        ORDER BY EventId, Date
        """
        print(f"Executing query: {query}")
        df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
        print(f"Query executed successfully, retrieved {len(df)} records")

        # 打印查询结果
        print(df.head())

        event_groups = df.groupby('EventId')
        result = []
        for event_id, group in event_groups:
            event_data = {
                "event_id": event_id,
                "data": group[['Date', 'value']].rename(columns={'Date': 'timestamp'}).to_dict(orient='records')
            }
            result.append(event_data)
        
        print("Successfully processed event data for response")
        return jsonify(result)
    except Exception as e:
        print(f"Error while processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/time_series_by_level", methods=["GET"])
def get_time_series_by_level():
    log_source = request.args.get("log_source", "hdfs")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    table_name = f"{log_source}_structured"
    print(f"Received request for time series data by event from log source: {log_source}, between {start_date} and {end_date}")
    
    try:
        query = f"""
        SELECT Level, Date, COUNT(*) as value
        FROM {table_name}
        WHERE Date BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY Level, Date
        ORDER BY Level, Date
        """
        print(f"Executing query: {query}")
        df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
        print(f"Query executed successfully, retrieved {len(df)} records")

        # 打印查询结果
        print(df.head())

        event_groups = df.groupby('Level')
        result = []
        for level, group in event_groups:
            event_data = {
                "level": level,
                "data": group[['Date', 'value']].rename(columns={'Date': 'timestamp'}).to_dict(orient='records')
            }
            result.append(event_data)
        
        print("Successfully processed event data for response")
        return jsonify(result)
    except Exception as e:
        print(f"Error while processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
