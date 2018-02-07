# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:32:51 2018

@author: A661242
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


if __name__ == '__main__':
    app.run(debug=True)