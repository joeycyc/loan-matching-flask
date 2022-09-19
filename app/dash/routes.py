from flask import Blueprint, render_template
from app.dash import dashboard_01 as dashboard_01_obj
from app.dash import dashboard_02 as dashboard_02_obj

dash_bp = Blueprint('dashboard', __name__)


@dash_bp.route('/dashboard_01')
def render_dashboard_01():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_02')
def render_dashboard_02():
    """Landing page."""
    return render_template('base.html')
