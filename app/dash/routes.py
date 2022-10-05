from flask import Blueprint, render_template

dash_bp = Blueprint('dashboard', __name__)


@dash_bp.route('/dashboard_01')
def render_dashboard_01():
    return render_template('base.html')


@dash_bp.route('/dashboard_01b')
def render_dashboard_01b():
    return render_template('base.html')


@dash_bp.route('/dashboard_01c')
def render_dashboard_01c():
    return render_template('base.html')


@dash_bp.route('/dashboard_01d')
def render_dashboard_01d():
    return render_template('base.html')


@dash_bp.route('/dashboard_04')
def render_dashboard_04():
    return render_template('base.html')


@dash_bp.route('/dashboard_05')
def render_dashboard_05():
    return render_template('base.html')


@dash_bp.route('/upload_file')
def render_upload_file():
    return render_template('base.html')
