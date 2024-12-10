import sqlite3

#fetch database
def fetch_all_data():
    conn = sqlite3.connect('rotation_results.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM rotation_data')
    
    # Fetch all rows from the query result
    rows = cursor.fetchall()

    # Print the fetched data
    for row in rows:
        print(row) 

    conn.close()

#clear current database content
def delete_all_data():
    conn = sqlite3.connect('rotation_results.db') 
    cursor = conn.cursor()

    cursor.execute('DELETE FROM rotation_data')

    conn.commit()
    conn.close()

def ask():
    response = input("Delete previous run? (y/n): ").strip().lower()
    if response == 'y':
        delete_all_data()
        print("data deleted")
    if response == 'n':
        print ("data not deleted")
    else: 
        print (" sure ") 
        
ask()
fetch_all_data()