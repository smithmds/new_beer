import pyodbc
server = '12.0.4100.bigslip01.windows.net'
database = 'Sensory'
username = 'datareader'
password = 'password'
driver= '{ODBC Driver 13 for SQL Server}'
port = None

if port is not None:
    connect_str = 'DRIVER={};PORT={};SERVER={};PORT={};DATABASE={};+UID={};PWD={}'.format(driver, port, server, port, database, username, password)
else:
    connect_str = 'DRIVER={};SERVER={};DATABASE={};+UID={};PWD={}'.format(driver, server, database, username, password)

    # cnxn = pyodbc.connect('DRIVER=' + driver + ';PORT=' + port + ';SERVER=' + server + ';PORT' + port + ';DATABASE=' + database+';UID=' + username+';PWD=' + password)

cnxn = pyodbc.connect(connect_str)
cursor = cnxn.cursor()
cursor.execute("SELECT TOP 10 * FROM dbo.CS_p50_AVL;")
row = cursor.fetchone()
while row:
    print (str(row[0]) + " " + str(row[1]))
    row = cursor.fetchone()
