from flask import render_template
from flask import current_app


@current_app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')