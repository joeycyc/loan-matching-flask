# Loan Matching 2.0 - Flask Application

## Usage
To run the application on Windows, execute `start.bat`.

## Enhancement procedures
### To add new dashboard
1. Add new dashboard_xx.py to app/dash/
2. Add render_dashboard_xx() in app/dash/routes.py
3. Add new link in app/templates/index.html
4. Add following line in create_app() in app/\_\_init\_\_.py
    - from app.dash.dashboard_xx import add_dashboard as add_dashboard_xx
    - flask_server = add_dashboard_xx(flask_server)

## Reference:
1. [Integrate Plotly Dash Into Your Flask App](https://hackersandslackers.com/plotly-dash-with-flask/)
2. How to embed a Dash app into an existing Flask app
    - [Tutorial](https://medium.com/@olegkomarov_77860/how-to-embed-a-dash-app-into-an-existing-flask-app-ea05d7a2210b)
    - [Github](https://github.com/okomarov/dash_on_flask)
3. Embed Multiple Dash Apps in Flask with Microsoft Authentication
    - [Tutorial](https://towardsdatascience.com/embed-multiple-dash-apps-in-flask-with-microsoft-authenticatio-44b734f74532)
    - [Github](https://github.com/shkiefer/dash_in_flask_msal/tree/basic)

