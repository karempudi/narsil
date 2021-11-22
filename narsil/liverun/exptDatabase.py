import psycopg2 as pgdatabase
import argparse
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import Signal, Qt
from datetime import datetime

"""
Useful for bundling all the database creation actions
and hold the information about the database
for a particular experiment, and some query funcitons
that can be called when you are updating the plots
in the main experiment window
"""
class exptDatabase(object):

    def __init__(self, dbname=None, user='postgres', password='postgres',
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

            cur.execute("SELECT datname FROM pg_database")
            rows = cur.fetchall()
            exptDbExists = False

            for row in rows:
                if row[-1] == self.dbname:
                    exptDbExists = True

            if exptDbExists == True:
                sys.stdout.write("Datbase detected\n")
                dlg = QMessageBox()
                dlg.setWindowTitle("Please Confirm !!!")
                dlg.setText(f"Expt database exists!!!, delete it ?")
                dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                dlg.setIcon(QMessageBox.Question)
                button = dlg.exec()

                if button == QMessageBox.No:
                    sys.stdout.write("Overwriting database ...")
                    sys.stdout.flush()
                
                if button == QMessageBox.Yes:
                    cur.execute("DROP DATABASE " + str(self.dbname))
                    sys.stdout.write("Database deletion done\n")
            else:
                cur.execute("CREATE DATABASE "+ str(self.dbname))
                sys.stdout.write(f"Creating database {self.dbname}\n")

            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"Error in db creation process {e}\n")
            sys.stderr.flush()

        finally:
            if con:
                con.close()
    
    def createTables(self):

        # loop through the tables list and create one at a time, using 
        # their specific schema 
        con = None
        tablesToCreate = []
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            # get all existing tables, and remove them if the prompt is answered yes
            cur.execute("""SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'""")
            
            rows = cur.fetchall()

            for row in rows:
                if row[-1] in self.tables:
                    sys.stdout.write(f"Table: {row[-1]} exists in {self.dbname} database ...\n")
                    dlg = QMessageBox()
                    dlg.setWindowTitle("Please Confirm !!!")
                    dlg.setText(f"{row[-1]} exists  !! , delete it?")
                    dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    dlg.setIcon(QMessageBox.Question)
                    button = dlg.exec()
                    
                    if button == QMessageBox.No:
                        sys.stdout.write(f"{row[-1]} table will be appended ...\n")
                    elif button == QMessageBox.Yes:
                        cur.execute("DROP TABLE " + str(row[-1]))
                        sys.stdout.write(f"{row[-1]} table is be deleted ...\n")
                        # create it fresh later
                        tablesToCreate.append(row[-1])
                    sys.stdout.flush()
            
            if len(rows) == 0:
                tablesToCreate = self.tables

            # now create all the tables that need creation
            # table is usuall on of ['arrival', 'segmented', 'deadalive', 'growth']
            # add anything else and write a schema for it if you want later
            sys.stdout.write(f"{tablesToCreate} will soon be created\n")
            sys.stdout.flush()
            for table in tablesToCreate:

                if table == 'arrival':
                    cur.execute("""CREATE TABLE arrival
                            (id SERIAL PRIMARY KEY, time TIMESTAMP, position INT, timepoint INT)
                            """)
                    sys.stdout.write(f"Table {table} is created ...\n")
                elif table == 'segment':
                    cur.execute("""CREATE TABLE segment
                            (id SERIAl PRIMARY KEY, time TIMESTAMP, position INT, timepoint INT,
                            segmentedpath VARCHAR, rawpath VARCHAR, locations BYTEA, numchannels INT)
                            """)
                    sys.stdout.write(f"Table {table} is created ...\n")
                elif table == 'deadalive':
                    cur.execute("""CREATE TABLE deadalive
                            (id SERIAL PRIMARY KEY, time TIMESTAMP, position INT, timepoint INT,
                            channelno INT, status BYTEA)
                            """)
                    sys.stdout.write(f"Table {table} is created ...\n")
                elif table == 'growth':
                    cur.execute("""CREATE TABLE growth
                            (id SERIAL PRIMARY KEY, time TIMESTAMP, position INT, timepoint INT, 
                            channelno INT, areas BYTEA, lengths BYTEA, numobjects INT)
                            """)
                    sys.stdout.write(f"Table {table} is created ...\n")
                else:
                    sys.stdout.write(f"Table {table} has no schema, so not created ... \n")
                    sys.stdout.flush()

            sys.stdout.write(f"Tables successfully created ... \n")
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
            con = pgdatabase.connect(user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            cur.execute("SELECT datname FROM pg_database")
            rows = cur.fetchall()
            exptDbExists = False

            for row in rows:
                if row[-1] == self.dbname:
                    exptDbExists = True

            if exptDbExists == True:
                sys.stdout.write("Datbase detected\n")
                dlg = QMessageBox()
                dlg.setWindowTitle("Please Confirm !!!")
                dlg.setText(f"Expt database exists!!!, delete it ?")
                dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                dlg.setIcon(QMessageBox.Question)
                button = dlg.exec()

                if button == QMessageBox.No:
                    sys.stdout.write("Overwriting database ...")
                    sys.stdout.flush()
                
                if button == QMessageBox.Yes:
                    cur.execute("DROP DATABASE " + str(self.dbname))
                    sys.stdout.write("Database deletion done\n")

            else:
                sys.stdout.write("Database doesn't exist so skipping ... \n")
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

            # get all existing tables, and remove them if the prompt is answered yes
            cur.execute("""SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'""")
            
            rows = cur.fetchall()

            deletedTables = []
            for row in rows:
                dlg = QMessageBox()
                dlg.setWindowTitle("Please Confirm !!!")
                dlg.setText(f"{row[-1]} exists  !! , delete it?")
                dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                dlg.setIcon(QMessageBox.Question)
                button = dlg.exec()
                
                if button == QMessageBox.No:
                    sys.stdout.write(f"{row[-1]} table will be skipped deleting ...\n")
                elif button == QMessageBox.Yes:
                    cur.execute("DROP TABLE " + str(row[-1]))
                    deletedTables.append(row[-1])
                    sys.stdout.write(f"{row[-1]} table is deleted ...\n")
                    # create it fresh later
                sys.stdout.flush()
            sys.stdout.write(f"{deletedTables} tables are deleted \n")
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
        data = []
        try:
            con = pgdatabase.connect(database=self.dbname, user=self.dbuser, password=self.dbpassword)
            cur = con.cursor()
            con.autocommit = True

            if tableName == 'arrival':
                cur.execute("SELECT position, timepoint FROM arrival")
            elif tableName == 'segment':
                cur.execute("SELECT position, timepoint FROM segment")
            elif tableName == 'deadalive':
                pass

            data = cur.fetchall()
            
            sys.stdout.write("")
            sys.stdout.flush()
        
        except pgdatabase.DatabaseError as e:
            sys.stderr.write(f"")
            sys.stderr.flush()
        
        finally:
            if con:
                con.close()
        return data 

# If you ever want to use this in QProcess 
if __name__ == "__main__":
    print("Creating experiment database ...")
