from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    student_name = request.form['student_name']
    student_id = request.form['student_id']

    # A simple rule-based system to predict student performance
    if int (student_id) % 2 == 0:
        prediction = 'Excellent'
    else:
        prediction = 'Good'

    return render_template('results.html', student_name=student_name, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
