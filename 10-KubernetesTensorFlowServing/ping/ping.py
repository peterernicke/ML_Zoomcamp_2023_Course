from flask import Flask

# creates an app with a name
app = Flask('ping')

# route specify at which address the function will live
@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)