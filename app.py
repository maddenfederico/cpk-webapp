from flask import Flask, render_template, request, redirect, url_for, session
from classifier import classify
import secrets
app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret
'''
result = {
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
    "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip "
    "ex ea commodo consequat.": {
        "Layer > Group Layers": .81,
        "Layer > Duplicate Group": .45
    },
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
    "dolore magna aliqua.": {
        "No Action": 1.0,
        "Layer > Duplicate Group": .45
    }
}
'''


@app.route("/", methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        session['text'] = request.form['textbox']
        return redirect(url_for('results'))
        
    return render_template("main.html")


@app.route("/results")
def results():
    result = classify(session.get('text', 'There was some weird issue where the results page gets returned before session is updated'))
    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
