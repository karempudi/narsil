import psycopg2 as pgdatabase
import argparse
import sys

"""
Useful for bundling all the database creation actions
and hold the information about the database
for a particular experiment, and some query funcitons
that can be called when you are updating the plots
in the main experiment window
"""
class exptDatabse(object):

    def __init__(self, dbname, user='postgres', password='postgres'
                tables=None):

        self.dbname = dbname
        self.dbuser = user
        self.dbpassword = password
        self.tables = tables

    def createDatabase(self):
        con = None
        try:
            con = pgdatabase.connect(user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()
            sys.stdout.write(f"")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()

        finally:
            if con:
                con.close()
    
    def createTable(self):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()
            sys.stdout.write(f"")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()
    
    def deleteDatabase(self):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()
            sys.stdout.write(f"")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()
        
    
    def deleteTables(self):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()
            sys.stdout.write(f"")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()

    def deleteOneTable(self):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()
            sys.stdout.write("")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()

   
    def queryDataForPlots(self, tableName):
        con = None
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute()

            sys.stdout.write("")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()

# If you ever want to use this in QProcess 
if __name__ == "__main__":
    print("Creating experiment database ...")
