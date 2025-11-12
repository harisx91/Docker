from flask import Flask

app = Flask(__name__)

color = "red"

@app.route('/')
def hello_world():
    return '<h1>New Change!</h1>'

def backgroud_color():
    print(color)

#Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
