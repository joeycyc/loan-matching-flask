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

        # TODO: blueprint to be added
        from .dash.routes import dash_bp
        flask_server.register_blueprint(dash_bp)

        # Process dash apps
        # TODO: add_dashboard to be added
        from app.dash.dashboard_01 import add_dashboard as add_dashboard_01
        flask_server = add_dashboard_01(flask_server)
        from app.dash.dashboard_02 import add_dashboard as add_dashboard_02
        flask_server = add_dashboard_02(flask_server)
        from app.dash.dashboard_03 import add_dashboard as add_dashboard_03
        flask_server = add_dashboard_03(flask_server)

        return flask_server



