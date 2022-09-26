from flask import Blueprint, render_template
from app.dash import dashboard_01 as dashboard_01_obj
from app.dash import dashboard_01b as dashboard_02_obj
from app.dash import dashboard_01c as dashboard_03_obj

dash_bp = Blueprint('dashboard', __name__)


@dash_bp.route('/dashboard_01')
def render_dashboard_01():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_01b')
def render_dashboard_01b():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_01c')
def render_dashboard_01c():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_01d')
def render_dashboard_01d():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_04')
def render_dashboard_04():
    """Landing page."""
    return render_template('base.html')


@dash_bp.route('/dashboard_05')
def render_dashboard_05():
    """Landing page."""
    return render_template('base.html')
