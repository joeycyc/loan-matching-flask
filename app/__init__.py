from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    """Initialize the core application."""
    flask_server = Flask(__name__, instance_relative_config=False)
    flask_server.config.from_object('config.DevConfig')

    # Register extensions
    db.init_app(flask_server)

    with flask_server.app_context():
        # Register blueprints
        from .routes import server_bp
        flask_server.register_blueprint(server_bp)
        from .dash.routes import dash_bp
        flask_server.register_blueprint(dash_bp)

        # Process dash apps
        from app.dash.dashboard_01 import add_dashboard as add_dashboard_01
        flask_server = add_dashboard_01(flask_server)
        # from app.dash.dashboard_01b import add_dashboard as add_dashboard_01b
        # flask_server = add_dashboard_01b(flask_server)
        # from app.dash.dashboard_01c import add_dashboard as add_dashboard_01c
        # flask_server = add_dashboard_01c(flask_server)
        from app.dash.dashboard_01d import add_dashboard as add_dashboard_01d
        flask_server = add_dashboard_01d(flask_server)
        from app.dash.dashboard_04 import add_dashboard as add_dashboard_04
        flask_server = add_dashboard_04(flask_server)
        from app.dash.dashboard_05 import add_dashboard as add_dashboard_05
        flask_server = add_dashboard_05(flask_server)

        return flask_server



