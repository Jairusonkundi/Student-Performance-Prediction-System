import gender as gender
from flask import Flask, render_template, jsonify, request
import joblib
import os
import numpy as np

app = Flask(__name__)


# To see the  home page
@app.route("/")
def index():
    return render_template('home.html')


# To see the result
@app.route('/predict', methods=['POST', 'GET'])
def result():
    studytime = float(request.form['studytime'])
    Medu = float(request.form['Medu'])
    Fedu = float(request.form['Fedu'])
    G1 = float(request.form['G1'])
    G2 = float(request.form['G2'])

    X = np.array([[studytime, Medu, Fedu, G1, G2]])

    model_path = r'C:\Users\Jairo\Desktop\Python\Student Performance Prediction System\models\lin_reg.sav'

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

    return render_template('result.html', res=score, grade=grade,remarks=remark)


if __name__ == '__main__':
    app.run(debug=True, port=9457)
