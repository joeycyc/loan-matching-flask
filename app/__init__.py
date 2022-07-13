from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_app():
    """Initialize the core application."""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object('config.DevConfig')

    # Initialize Plugins
    db.init_app(app)

    with app.app_context():
        # Include routes
        from . import routes

        # Import Dash application
        from .dash.dashboard_01 import init_dashboard
        app = init_dashboard(app)

        return app
