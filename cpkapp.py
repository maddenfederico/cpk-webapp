from flask import Flask, render_template

app = Flask(__name__)

sentences = {
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


@app.route("/")
def hello():
    return render_template("main.html")


@app.route("/results")
def results():
    return render_template("results.html", sentences=sentences)


if __name__ == "__main__":
    app.run(debug=True)
