# ASR-Portfolio-Tracker
This repository contains a command-line interface (CLI) application to track a simple investment portfolio of the magnificent seven.
It consists out of the following components:
    - Model.py: the main engine, consisting out of a Stock class and a Portfolio class entaining corresponding relevant functions
    - Viewer.py: file entaining the desired data visualisation tools
    - DataCollector.py: here data from the Yahoo Finance API gets called to get historic prices of the magnificent seven. Data over 10 years is used from 01-01-2015 until      31-12-2024
    - Controller.py: this is the control room, from which the classes and functions from above files can be called

For the relevant packages and dependencies see requirements.txt (PDF Viewer, numpy, pandas, yfinance, seaborn)

