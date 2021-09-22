## Functions to do live run live in this module

## UI directory

ui directory has the .ui files designed using Qt Designer 5.11.1.

These .ui file are converted to python classes to access the GUI elements
using PySide6 as the choice of python interface for Qt functionality.

To compile the .ui files to python classes use the following command

```
pyside-uic main.ui > ui_mainwindow.py
```

Each window on the screen has a .ui file associated with it and python class
to access the elements and functions to run when events happen in that window.

We use the classes for each of the window in the UI directory to build on top 
of them and hook all the handlers 

## Classes useful for the GUI

exptDatabase.py handles all the database queries, setup and schemas

exptProcesses.py handles the process creation and loading the nets and 
other details in the backend image processing of the GUI.

As usual run.py will have the classes and funcitons need to run the main
GUI loop. run.py will have the starting functions for the liverun module

datasets.py and utils.py have databundling capabilites and utils needed
in the GUI both in the front and backend

exptUI.py has all the classes that hook up the buttons with appropriate 
funcitonality in the UI using he ui_window.py files of each of the windows
in the UI directory.

## Database 

A Postgresql database is used to store the data used for plotting.
Additionally most analyzed data will be bundled and collected for further
analysis after the experiment run.

Generally one database is created for each experiment and, each of the 
database has one table for each process, to keep track of things going on
in the process, and each table has plots to monitor the status

