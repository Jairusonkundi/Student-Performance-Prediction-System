from flask import Flask
from twilio.rest import Client

app = Flask(__name__)

@app.route("/send_sms/<student_name>/<grade>")
def send_sms(student_name, grade):
    # Your Twilio account sid and auth token
    account_sid = "ACae8d5c83cdc12ef88647fb6925d54d02"
    auth_token = "66984aca9974d1d4c7efa9c3b6feba0d"
    client = Client(account_sid, auth_token)

    message = f"Hello {student_name}, Your grade is: {grade}"

    # Your Twilio phone number
    from_number = "+254743192585"
    # The student's phone number
    to_number = "+254701436784"

    client.messages.create(to=to_number, from_=from_number, body=message)

    return "SMS sent successfully!"

if __name__ == '__main__':
    app.run()
