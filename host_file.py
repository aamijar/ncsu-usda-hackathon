from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
     return render_template('index.html')

@app.route('/home')
def home():
     os.system("python3 /home/pi/mmcontrol2.py -hm")
     return render_template('home.html')

@app.route('/next')
def next():
     os.system("python3 /home/pi/mmcontrol2.py -d 200")
     return render_template('home.html')

@app.route("/reset")
def re():
     os.system("python3 /home/pi/master_control.py")
     return render_template('home.html')

if __name__ == '__main__':
     app.run(debug=True,host='10.138.198.250')
