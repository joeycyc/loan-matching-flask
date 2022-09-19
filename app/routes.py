from flask import Blueprint, render_template
# from flask import current_app

# A.k.a. app/webapp.py

server_bp = Blueprint('main', __name__)


@server_bp.route('/')
def render_index():
    """Landing page."""
    return render_template('index.html')
