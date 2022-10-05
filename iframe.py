from flask import Flask, render_template

app = Flask(__name__, static_url_path="", static_folder="resources/static",
            template_folder="resources/templates")

@app.route('/')
def home():
    return 'Home Page.'

@app.route('/ml')
def mlpage():
    return render_template('ml_iframe.html')

if __name__ == '__main__':
    app.run(debug=True)