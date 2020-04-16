import pymysql.cursors
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from flask import render_template, Flask, request, url_for, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from statistics import mean,median,stdev
import os

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def predict(nilai_kompetensi,nilai_behavior,engagement_ucapan,engagement_tinggal):
    conn = pymysql.connect(host='localhost',
                           user='root',
                           password='',
                           database='telkom')
    try:
        with conn.cursor() as cur:
            sql = "SELECT * FROM karyawan"
            cur.execute(sql)
            result = cur.fetchall()
            fro = pd.read_sql(sql, conn)
            pd.set_option('display.expand_frame_repr', False)
            fro.dropna(inplace=True)
            y = fro['performansi_individu']
            y.values.astype(float)
            y.head()

            X = fro.drop(["id","created_at","performansi_individu","updated_at"], axis=1)
            X.values.astype(float)
            X.head()

            regressor = SVR(kernel = 'linear')
            regressor.fit(X.values, y.values)

            yPred = regressor.predict(X.values)


            print ("Score :", regressor.score(X.values, yPred))
            print ("MSE : %.2f" % mean_squared_error(y.values, yPred))
            print("X: \n", X)
            print("Y: \n", y)
            print(len(result))

            # nilai_kompetensi = 2
            # nilai_behavior = 1.7
            # engagement_ucapan = 3
            # engagement_tinggal= 2.2


            Xnew = [[nilai_kompetensi, nilai_behavior, engagement_ucapan, engagement_tinggal]]
            ynew = regressor.predict(Xnew)
            inp = "INSERT INTO karyawan (nilai_kompetensi, nilai_behavior, engagement_ucapan, engagement_tinggal, performansi_individu) VALUES (%s, %s, %s, %s, %s)"
            cur.execute(inp, (nilai_kompetensi, nilai_behavior, engagement_ucapan, engagement_tinggal, ynew.item()))
            conn.commit()
            print("Predict ",ynew.item())
            return ynew.item()
    finally:
        conn.close()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
   return render_template("index.html")

@app.route('/graph', methods=['GET', 'POST'])

def formu():
    conn = pymysql.connect(host='localhost',
                           user='root',
                           password='',
                           database='telkom')
    # load performansi karyawan
    pkv = conn.cursor()
    pkv.execute("SELECT performansi_individu FROM karyawan")
    pkvrow = pkv.fetchall()
    pkvnum = list(sum(pkvrow, ()))
    pkmax = max(pkvnum)
    pkmin = min(pkvnum)
    pkm = mean(pkvnum)
    pkmean = np.around(pkm, decimals = 2)

    # load nilai kompetensi
    nkv = conn.cursor()
    nkv.execute("SELECT nilai_Kompetensi FROM karyawan")
    nkvrow = nkv.fetchall()
    nkvnum = list(sum(nkvrow, ()))
    nkmax = max(nkvnum)
    nkmin = min(nkvnum)
    nkm = mean(nkvnum)
    nkmean = np.around(nkm, decimals = 2)

    # load nilai behaviour
    nbv = conn.cursor()
    nbv.execute("SELECT nilai_Behavior FROM karyawan")
    nbvrow = nbv.fetchall()
    nbvnum = list(sum(nbvrow, ()))
    nbmax = max(nbvnum)
    nbmin = min(nbvnum)
    nbm = mean(nbvnum)
    nbmean = np.around(nbm, decimals = 2)

    # load engagement ucapan
    euv = conn.cursor()
    euv.execute("SELECT engagement_ucapan FROM karyawan")
    euvrow = euv.fetchall()
    euvnum = list(sum(euvrow, ()))
    eumax = max(euvnum)
    eumin = min(euvnum)
    eum = mean(euvnum)
    eumean = np.around(eum, decimals = 2)

    # load engagement tinggal
    etv = conn.cursor()
    etv.execute("SELECT engagement_tinggal FROM karyawan")
    etvrow = etv.fetchall()
    etvnum = list(sum(etvrow, ()))
    etmax = max(etvnum)
    etmin = min(etvnum)
    etm = mean(etvnum)
    etmean = np.around(etm, decimals = 2)

    if request.method == 'POST':
        nilai_kompetensi = request.form["nilai_kompetensi"]
        nilai_behavior = request.form["nilai_behavior"]
        engagement_ucapan = request.form["engagement_ucapan"]
        engagement_tinggal = request.form["engagement_tinggal"]
        pre = predict(nilai_kompetensi,nilai_behavior,engagement_ucapan,engagement_tinggal)
        return render_template("graph.html", name = round(pre),
                                kom = nilai_kompetensi,
                                beh = nilai_behavior,
                                uc = engagement_ucapan,
                                ti = engagement_tinggal,
                                pk = pkvnum,
                                pkmax = pkmax,
                                pkmin = pkmin,
                                pkmean = pkmean,
                                nk = nkvnum,
                                nkmax = nkmax,
                                nkmin = nkmin,
                                nkmean = nkmean,
                                nb = nbvnum,
                                nbmax = nbmax,
                                nbmin = nbmin,
                                nbmean = nbmean,
                                eu = euvnum,
                                eumax = eumax,
                                eumin = eumin,
                                eumean = eumean,
                                et = etvnum,
                                etmax = etmax,
                                etmin = etmin,
                                etmean = etmean
                               )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.secret_key = "secret key"

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
         if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
         file = request.files['file']
         if file.filename == '':
            flash('No file selected for uploading')
            return redirect('/')
         if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return redirect('/')
         else:
            flash('Allowed file types are csv and xlsx')
            return redirect('/')

@app.route('/upload')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
   app.secret_key = 'super secret key'
   app.config['SESSION_TYPE'] = 'filesystem'
   app.run(debug = True)