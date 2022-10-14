# Loan Matching 2.0 - Flask Application

## Objective
To visualize the back-to-back support of loan facilities on development project (DP) funding needs.

## Usage
### Run application
- On Windows, execute `.\start.bat`
- On UNIX type OS, execute `wsgi.py`

### GUI
View [here](https://github.com/joeycyc/loan-matching-flask/blob/master/github/intro.gif) if you cannot see the GIF.
![](https://github.com/joeycyc/loan-matching-flask/blob/master/github/intro.gif)

## Program Logic
Please view [Implementation Details](https://github.com/joeycyc/loan-matching-flask/blob/master/github/implementation_details.pdf) for the For the underlying logic of loan matching.

## Operation Procedures
### Installation
1. Clone the project folder
2. Set up Python virtual environment (Python >= 3.8) and install requirements
3. Place the data files `project_data.xlsx`, `bts_data.xlsx`, `project_data_template.xlsx`, `bts_data_template.xlsx` in `app/dash/data/input/`

### Enhancement
#### To add new dashboard
1. Add new `dashboard_xx.py` and `dash_config_xx.yaml` to `app/dash/`
2. Add `render_dashboard_xx()` in `app/dash/routes.py`
3. Add new link in `app/templates/index.html`
4. Add following line in `create_app()` in `app/__init__.py`
    - `from app.dash.dashboard_xx import add_dashboard as add_dashboard_xx`
    - `flask_server = add_dashboard_xx(flask_server)`

## Reference:
1. [Integrate Plotly Dash Into Your Flask App](https://hackersandslackers.com/plotly-dash-with-flask/)
2. How to embed a Dash app into an existing Flask app
    - [Tutorial](https://medium.com/@olegkomarov_77860/how-to-embed-a-dash-app-into-an-existing-flask-app-ea05d7a2210b)
    - [Github](https://github.com/okomarov/dash_on_flask)
3. Embed Multiple Dash Apps in Flask with Microsoft Authentication
    - [Tutorial](https://towardsdatascience.com/embed-multiple-dash-apps-in-flask-with-microsoft-authenticatio-44b734f74532)
    - [Github](https://github.com/shkiefer/dash_in_flask_msal/tree/basic)

