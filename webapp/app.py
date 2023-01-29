import flask
import pickle
import pandas as pd
import numpy as np


with open(f'webapp/model/malariaPrediction.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        location = flask.request.form['location']
        year = flask.request.form['year']
        falci = flask.request.form['falci']
        vivax = flask.request.form['vivax']
        ItnUse = flask.request.form['ItnUse']
        ItnUseRate = flask.request.form['ItnUseRate']
        ItnAccess = flask.request.form['ItnAccess']
        irs = flask.request.form['irs']
        treatment = flask.request.form['treatment']
        input_variables = np.array([[location, year,falci,vivax,ItnUse,ItnUseRate,ItnAccess,irs,treatment]])
        prediction = model.predict(input_variables)[0]
        return (flask.render_template('main.html', result=prediction))

if __name__ == '__main__':
    app.run()