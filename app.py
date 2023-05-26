import gender as gender
#importing the libraries
import joblib
import os
import numpy as np
from flask import Flask,render_template,request,session,logging,url_for,redirect,flash,jsonify
from sqlalchemy import create_engine,text
from sqlalchemy.orm import scoped_session,sessionmaker
from sqlalchemy.sql import text
from passlib.hash import sha256_crypt
from twilio.rest import Client


engine= create_engine("mysql+pymysql://jairo:12345678@localhost/login")
                      #(mysql+pymysql://username:password@localhost/databasename)

db=scoped_session(sessionmaker(bind=engine))

# using flask framewok to deploy the model on the website
app = Flask(__name__)


# To return the  home page
@app.route("/")
def index():
    return render_template('home.html')
    # Registration
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        secure_password = sha256_crypt.encrypt(str(password))

        if password == confirm:

            query = text("INSERT INTO users(name,username,password) VALUES(:name,:username,:password)")
            db.execute(query, {"name": name, "username": username, "password": secure_password})

            db.commit()
            flash("Registration Successful you can Login", "success")
            return redirect(url_for('login'))
        else:
            flash("Password does not match!!", "danger")
            return render_template("register.html")
    return render_template("register.html")


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        #password_to_check = request.form.get("password")

        usernamedata = text("SELECT username FROM users WHERE username = :username")
        usernamedata = db.execute(usernamedata, {"username": username}).fetchone()

        passworddata = text("SELECT password FROM users WHERE  username = :username")
        passworddata = db.execute(passworddata, {"username": username}).fetchone()


        if usernamedata is None:
            flash("No username", "danger")
            render_template("login.html")
            # passworddata = db.execute(text("SELECT password FROM users WHERE password= :password"),
            #                           {"password": password}).fetchone()

        else:

            if password is None:
                flash("wrong password", "danger")
                render_template("login.html")
            else:
                if sha256_crypt.verify(password, passworddata[0]):
                    session["log"] = True

                    flash("You are now Logged In", "success")
                    #return redirect(url_for('predict'))
                    return render_template("predict.html")
                else:
                    flash("Incorrect password", "danger")
                    return render_template("login.html")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You are now logged out","Success")
    return redirect(url_for('login'))

# To receive the data entered in the student form and process it
@app.route('/predict', methods=['POST', 'GET'])
def result():
    studytime = float(request.form['studytime'])
    Medu = float(request.form['Medu'])
    Fedu = float(request.form['Fedu'])
    G1 = float(request.form['G1'])
    G2 = float(request.form['G2'])

    X = np.array([[studytime,Medu,Fedu, G1, G2]])


    model_path = r'C:\Users\Jairo\Desktop\Python\Student Performance Prediction System\models\lin_regre.sav'

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    score = float("{:.2f}".format(y_pred.item()))
    print(score)

    if 15 <= score <= 20:
        grade = "A"
    elif 12 <= score < 15:
        grade = "B"
    elif 9 <= score < 12:
        grade = "C"

    elif 6 <= score < 9:
        grade = "D"
    elif 3 <= score < 6:
        grade = "E"
    else:
        grade = "F"
    # TO give remarks according to the grade scored

    print(grade)
    if grade == "A":
        remark = "Excellent performance! Keep up the good work."
    elif grade == "B":
        remark = "Good job. Keep striving for improvement."
    elif grade == "C":
        remark = "Satisfactory performance. Room for improvement."
    elif grade == "D":
        remark = "Needs improvement. Put in more effort."
    elif grade == "E":
        remark = "Very weak performance. Significant improvement needed."
    else:
        remark = "Poor performance. Seek assistance."

   # To  return the result page with the score,grade and remarks
    return render_template('result.html', res=score, grade=grade,remarks=remark)
#Predict
@app.route("/prediction")
def prediction():
    return render_template("predict.html")
#Result
# @app.route("/result")
# def results():
#     return render_template("result.html")
# @app.route("/send_sms")
# def send_sms(student_name, grade):
# @app.route("/send_sms")
# def send_sms(grade):
#
#     # Your Twilio account sid and auth token
#     account_sid = "ACae8d5c83cdc12ef88647fb6925d54d02"
#     auth_token = "66984aca9974d1d4c7efa9c3b6feba0d"
#     client = Client(account_sid, auth_token)
#
#     message = f"Hello {username}, Your grade is: {grade}"
#
#     # Your Twilio phone number
#     from_number = "+254743192585"
#     # The student's phone number
#     to_number = "+254701436784"
#
#     client.messages.create(to=to_number, from_=from_number, body=message)
#
#     return render_template("notification.html")

if __name__ == '__main__':
    # to secure session data from attackers
    app.secret_key = "Jairo1234"
    app.run(debug=True, port=9457)
