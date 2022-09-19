from app import create_app

# A.k.a. dashapp.py

flask_server = create_app()

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80, debug=True)
    flask_server.run(host='0.0.0.0', debug=True)

