"""
Dashboard for matching 60% land cost with corporate loan (term & revolver)
With Uncommitted Revolver replacement + Committed Revolver ceiling
Based on dashboard_01d, added Committed Revolver ceiling
"""
from .utils import *
from .loan_matching import LoanMatching
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx
import pickle
import codecs

DASHBOARD_ID = 'dashboard_04'
URL_BASE = '/dashboard_04/'
DASH_CONFIG_FILEPATH = './app/dash/dash_config_04.yaml'


def get_gantt_3(matching_object, stage_to_display, projects_to_display,
                bar_height=30, *args, **kwargs):
    """Visualize data in gantt chart - version 3: 4 subplots for shortfall+matched, leftover, total shortfall, equity,
     color = amount
    Args:
    - stage_to_display: str, 'stage (0|1|2|...)'
    - projects_to_display: list of strings of project names
    - bar_height: int, bar height
    """
    master_output_vis = matching_object.MASTER_OUTPUT.copy()
    master_output_vis['trace_value_str'] = master_output_vis['trace_value'].apply(lambda x: '%.3f' % x)

    gantts = ['gantt_main', 'gantt_fac', 'gantt_ttl_shortfall', 'gantt_equity', 'gantt_for_acq']
    gantts_titles = ['Project needs and matched entries (HK$B)',
                     'Loan facilities available (leftover) (HK$B)',
                     'Total project needs (HK$B)',
                     'Equity (HK$B)',
                     'Revolver set aside for acquisition (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': master_output_vis['stage'] == stage_to_display,
                             'project': master_output_vis['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_main'] = (master_output_vis['output_type'].isin(['shortfall', 'matched'])) & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (master_output_vis['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (master_output_vis['loan_type'] != 'Equity')
    conditions_to_display['gantt_ttl_shortfall'] = (master_output_vis['output_type'] == 'ttl_shortfall') & \
        conditions_to_display['stage']
    conditions_to_display['gantt_equity'] = (master_output_vis['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (master_output_vis['loan_type'] == 'Equity')
    conditions_to_display['gantt_for_acq'] = (master_output_vis['output_type'] == 'for_acquisition') & \
        conditions_to_display['stage']

    subplot_data = {g: master_output_vis[conditions_to_display[g]] for g in gantts}

    # Dynamically define the heights of subplots given with the number of items
    # Calculate the number of items, at least 5
    subplot_num_items_to_display = {g: max(len(subplot_data[g]['y_label'].unique()), 5) for g in gantts}
    # Each bar 30px
    subplot_heights = {g: subplot_num_items_to_display[g] * bar_height for g in gantts}
    plot_height = sum(subplot_heights.values())
    subplot_height_shares = [subplot_heights[g] / plot_height for g in gantts]

    ## spacing between 2 subplots
    subplot_vertical_spacing = 0.05

    # marker_colorscale
    # marker_colorscales = {k: v for k, v in zip(gantts, ['Greens', 'Reds', 'Purples'])}

    # Manually construct Bar graph_object
    fig = make_subplots(rows=len(gantts), cols=1,
                        shared_xaxes=True,
                        row_heights=subplot_height_shares,
                        vertical_spacing=subplot_vertical_spacing,
                        subplot_titles=gantts_titles)
    for g_idx, g in enumerate(gantts):
        plot_df = subplot_data[g].copy()
        # Skip to next loop if dataframe is empty
        if plot_df.shape[0] == 0:
            fig.add_trace(go.Bar(x=plot_df[['from_date', 'to_date']].apply(lambda x: dt.date(1970, 1, 1) + (x[1] - x[0]), axis=1),
                                 y=plot_df['y_label'],
                                 orientation='h',
                                 base=plot_df['from_date']),
                          row=g_idx + 1, col=1)
            continue
        # General treatment for all subplot
        plot_df['net_margin'] = plot_df['net_margin'].fillna('0')
        # Special treatment per subplot
        if g == 'gantt_main':  # Group by project then output_type (shortfall -> matched)
            plot_df.sort_values(by=['stage', 'project_id', 'output_type_num', 'solo_jv_rank',
                                    'loan_sub_type_rank', 'loan_id', 'loan_facility_id'],
                                ascending=[True, True, False, True,
                                           True, True, True],
                                inplace=True)
            plot_df['y_label'] = plot_df[['output_type', 'y_label']].apply(
                lambda x: x[1] if x[0] == 'matched' else x[1]+' - Project needs (shortfall)', axis=1)
            plot_df.reset_index(drop=True, inplace=True)
        elif g == 'gantt_equity':
            plot_df['output_type'] = 'equity'

        # Set marker colors and other visualization attributes
        plot_df['marker_color'] = plot_df['output_type'].astype(str) + '-' + \
                                  plot_df['loan_sub_type'].astype(str).replace('nan', '') + '-' + \
                                  plot_df['match_type'].astype(str).replace('nan', '') + '-' + \
                                  plot_df['committed'].astype(str).replace('nan', '')
        plot_df['marker_color'].replace({
            '.*shortfall.*': 'indigo',
            'matched-T.*': 'yellow',
            'matched-R-normal.*': 'orange',
            'matched-R-replacement.*': 'deepskyblue',
            'matched-R-reserved.*': 'orange',
            'matched-Equity.*': 'hotpink',
            'leftover-T.*': 'yellow',
            'leftover-R.*-C': 'orange',
            'leftover-R.*-U': 'deepskyblue',
            'equity-Equity.*': 'hotpink',
            'for_acquisition-R-for_acquisition-C': 'orange',
            'for_acquisition-R-for_acquisition-U': 'deepskyblue',
        }, regex=True, inplace=True)
        # Past colors: '#ddd255', '#f29340', 'pink', 'firebrick', 'seagreen', 'mediumslateblue', 'teal'
        plot_df['marker_line_width'] = \
            plot_df['match_type'].apply(lambda x: 1 if x == 'reserved' else 0)
        plot_df['marker_line_color'] = \
            plot_df['match_type'].apply(lambda x: 'orange' if x == 'reserved' else '#444')
        plot_df['marker_opacity'] = \
            plot_df['match_type'].apply(lambda x: 0 if x == 'reserved' else 1)
        plot_df['marker_pattern_shape'] = \
            plot_df['match_type'].apply(lambda x: '/' if x == 'reserved' else '')
        plot_df['marker_pattern_solidity'] = \
            plot_df['match_type'].apply(lambda x: 0.1 if x == 'reserved' else 0.3)

        # Rows are displayed from the bottom to the top
        plot_df = plot_df.iloc[::-1]

        fig.add_trace(go.Bar(x=plot_df[['from_date', 'to_date']].apply(lambda x: dt.date(1970, 1, 1) + (x[1] - x[0]), axis=1),
                             y=plot_df['y_label'],
                             orientation='h',
                             base=plot_df['from_date'],
                             marker_color=plot_df['marker_color'],
                             # marker_line_width=plot_df['marker_line_width'],
                             # marker_line_color=plot_df['marker_line_color'],
                             # marker_opacity=plot_df['marker_opacity'],
                             marker_pattern_shape=plot_df['marker_pattern_shape'],
                             marker_pattern_solidity=plot_df['marker_pattern_solidity'],
                             # marker_colorscale=marker_colorscales[g],
                             text=plot_df['trace_value_str'],
                             customdata=np.stack((plot_df['to_date'], plot_df['net_margin']), axis=-1),
                             # customdata=plot_df['to_date'],
                             hovertemplate='%{y}<br>'
                                           '%{base: %Y-%B-%a %d} to %{customdata[0]: %Y-%B-%a %d}<br>'
                                           'Amount in $B: %{text}<br>'
                                           'Net margin: %{customdata[1]:.3f}%'),
                      row=g_idx+1, col=1)

    fig.update_layout(height=plot_height)
    fig.update_yaxes(tickfont_size=12)
    fig.update_xaxes(type='date',
                     ticktext=matching_object.DATE_TICK_TEXTS,
                     tickvals=matching_object.DATE_TICK_VALUES)
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    return fig


def get_gantt_diff_bar_width_3(matching_object, stage_to_display, projects_to_display,
                               bar_height=30, trace_height_range=(0.25, 1.0), *args, **kwargs):
    """Visualize data in gantt chart - similar to get_gantt_3, but different bar widths according to amount
    Args:
    - stage_to_display: str, 'stage (0|1|2|...)'
    - projects_to_display: list of strings of project names
    - bar_height: int, bar height
    - trace_height_range: from 0 to 1
    """
    master_output_vis = matching_object.MASTER_OUTPUT.copy()
    master_output_vis['trace_value_str'] = master_output_vis['trace_value'].apply(lambda x: '%.3f' % x)
    # Trace width, apply log transformation to trace_value log2(x+4)*10
    # master_output_vis['trace_height'] = master_output_vis['trace_value'].apply(lambda x: int(np.log2(x + 4) * 10))

    gantts = ['gantt_main', 'gantt_fac', 'gantt_ttl_shortfall', 'gantt_equity', 'gantt_for_acq']
    gantts_titles = ['Project needs and matched entries (HK$B)',
                     'Loan facilities available (leftover) (HK$B)',
                     'Total project needs (HK$B)',
                     'Equity (HK$B)',
                     'Revolver set aside for acquisition (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': master_output_vis['stage'] == stage_to_display,
                             'project': master_output_vis['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_main'] = (master_output_vis['output_type'].isin(['shortfall', 'matched'])) & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (master_output_vis['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (master_output_vis['loan_type'] != 'Equity')
    conditions_to_display['gantt_ttl_shortfall'] = (master_output_vis['output_type'] == 'ttl_shortfall') & \
        conditions_to_display['stage']
    conditions_to_display['gantt_equity'] = (master_output_vis['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (master_output_vis['loan_type'] == 'Equity')
    conditions_to_display['gantt_for_acq'] = (master_output_vis['output_type'] == 'for_acquisition') & \
        conditions_to_display['stage']

    subplot_data = {g: master_output_vis[conditions_to_display[g]] for g in gantts}

    # Dynamically define the heights of subplots given with the number of items
    # Calculate the sum of max amount ($B) per y_label, value is 0 if subplot_data is empty dataframe
    subplot_summaxamts = {
        **{g: int(np.ceil(subplot_data[g][['trace_value', 'y_label']].groupby('y_label').max('trace_value').sum()))
           for g in gantts if subplot_data[g].shape[0] > 0},
        **{g: 0 for g in gantts if subplot_data[g].shape[0] == 0}
    }
    # Calculate the no. of rows in each subplot
    subplot_num_rows = {g: len(subplot_data[g]['y_label'].unique()) for g in gantts}
    # Set subplot heights
    # Each bar height at least 20px
    # subplot_heights = {g: max(subplot_num_rows[g] * 20, subplot_summaxamts[g] * 2) for g in gantts}
    subplot_heights = dict()
    for g in gantts:
        if g in ['gantt_main', 'gantt_fac']:
            # Each bar height at least 30px
            subplot_heights[g] = max(subplot_summaxamts[g] * bar_height / 10, subplot_num_rows[g] * bar_height)
        elif g in ['gantt_ttl_shortfall', 'gantt_equity']:
            # Bound to [80px, 360px]
            subplot_heights[g] = max(min(subplot_summaxamts[g] * bar_height / 10, 360), 80)
        else:
            subplot_heights[g] = 100
    plot_height = sum(subplot_heights.values())
    subplot_height_shares = [subplot_heights[g] / plot_height for g in gantts]

    ## spacing between 2 subplots
    subplot_vertical_spacing = 0.05

    # marker_colorscale
    # marker_colorscales = {k: v for k, v in zip(gantts, ['Greens', 'Reds', 'Purples'])}

    # Manually construct Bar graph_object
    fig = make_subplots(rows=len(gantts), cols=1,
                        shared_xaxes=True,
                        row_heights=subplot_height_shares,
                        vertical_spacing=subplot_vertical_spacing,
                        subplot_titles=gantts_titles)
    for g_idx, g in enumerate(gantts):
        plot_df = subplot_data[g].copy()
        # Skip to next loop if dataframe is empty
        if plot_df.shape[0] == 0:
            # TODO: How to show empty subplot?
            # fig.add_trace(
                # go.Bar(x=plot_df[['from_date', 'to_date']].apply(lambda x: dt.date(1970, 1, 1) + (x[1] - x[0]), axis=1),
                #        y=plot_df['y_label'],
                #        orientation='h',
                #        base=plot_df['from_date']),
                # row=g_idx + 1, col=1)
            continue
        # General treatment for all subplot
        plot_df['net_margin'] = plot_df['net_margin'].fillna('0')
        # Special treatment per subplot
        if g == 'gantt_main':  # Group by project then output_type (shortfall -> matched)
            plot_df.sort_values(by=['stage', 'project_id', 'output_type_num', 'solo_jv_rank',
                                    'loan_sub_type_rank', 'loan_id', 'loan_facility_id'],
                                ascending=[True, True, False, True,
                                           True, True, True],
                                inplace=True)
            plot_df['y_label'] = plot_df[['output_type', 'y_label']].apply(
                lambda x: x[1] if x[0] == 'matched' else x[1]+' - Project needs (shortfall)', axis=1)
            plot_df.reset_index(drop=True, inplace=True)
        elif g == 'gantt_equity':
            plot_df['output_type'] = 'equity'

        # Set marker colors and other visualization attributes
        plot_df['marker_color'] = plot_df['output_type'].astype(str) + '-' + \
                                  plot_df['loan_sub_type'].astype(str).replace('nan', '') + '-' + \
                                  plot_df['match_type'].astype(str).replace('nan', '') + '-' + \
                                  plot_df['committed'].astype(str).replace('nan', '')
        plot_df['marker_color'].replace({
            '.*shortfall.*': 'indigo',
            'matched-T.*': 'yellow',
            'matched-R-normal.*': 'orange',
            'matched-R-replacement.*': 'deepskyblue',
            'matched-R-reserved.*': 'orange',
            'matched-Equity.*': 'hotpink',
            'leftover-T.*': 'yellow',
            'leftover-R.*-C': 'orange',
            'leftover-R.*-U': 'deepskyblue',
            'equity-Equity.*': 'hotpink',
            'for_acquisition-R-for_acquisition-C': 'orange',
            'for_acquisition-R-for_acquisition-U': 'deepskyblue',
        }, regex=True, inplace=True)
        # Past colors: '#ddd255', '#f29340', 'pink', 'firebrick', 'seagreen', 'mediumslateblue', 'teal'
        plot_df['marker_line_width'] = \
            plot_df['match_type'].apply(lambda x: 1 if x == 'reserved' else 0)
        plot_df['marker_line_color'] = \
            plot_df['match_type'].apply(lambda x: 'orange' if x == 'reserved' else '#444')
        plot_df['marker_opacity'] = \
            plot_df['match_type'].apply(lambda x: 0 if x == 'reserved' else 1)
        plot_df['marker_pattern_shape'] = \
            plot_df['match_type'].apply(lambda x: '/' if x == 'reserved' else '')
        plot_df['marker_pattern_solidity'] = \
            plot_df['match_type'].apply(lambda x: 0.1 if x == 'reserved' else 0.3)

        # Set trace_height - relative height from given range
        # trace_height_range = (0.25, 1.0)
        trace_value_max = max(max(plot_df['trace_value']), 0.001)
        # plot_df['trace_height'] = plot_df['trace_value'] / max(plot_df['trace_value'])
        # plot_df['trace_height'] = plot_df['trace_height'].apply(lambda x: max(x, 0.5))
        plot_df['trace_height'] = plot_df['trace_value'].apply(
            lambda x: x / trace_value_max * (trace_height_range[1]-trace_height_range[0]) + trace_height_range[0])

        # Rows are displayed from the bottom to the top
        plot_df = plot_df.iloc[::-1]

        fig.add_trace(go.Bar(x=plot_df[['from_date', 'to_date']].apply(lambda x: dt.date(1970, 1, 1) + (x[1] - x[0]), axis=1),
                             y=plot_df['y_label'],
                             width=plot_df['trace_height'],
                             orientation='h',
                             base=plot_df['from_date'],
                             marker_color=plot_df['marker_color'],
                             # marker_line_width=plot_df['marker_line_width'],
                             # marker_line_color=plot_df['marker_line_color'],
                             # marker_opacity=plot_df['marker_opacity'],
                             marker_pattern_shape=plot_df['marker_pattern_shape'],
                             marker_pattern_solidity=plot_df['marker_pattern_solidity'],
                             # marker_colorscale=marker_colorscales[g],
                             text=plot_df['trace_value_str'],
                             constraintext='none',
                             customdata=np.stack((plot_df['to_date'], plot_df['net_margin']), axis=-1),
                             # customdata=plot_df['to_date'],
                             hovertemplate='%{y}<br>'
                                           '%{base: %Y-%B-%a %d} to %{customdata[0]: %Y-%B-%a %d}<br>'
                                           'Amount in $B: %{text}<br>'
                                           'Net margin: %{customdata[1]:.3f}%'),
                      row=g_idx+1, col=1)

    fig.update_layout(height=plot_height)
    # fig.update_traces(textfont_size=12)
    fig.update_yaxes(tickfont_size=12)
    fig.update_xaxes(type='date',
                     ticktext=matching_object.DATE_TICK_TEXTS,
                     tickvals=matching_object.DATE_TICK_VALUES)
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    return fig


def add_dashboard(server):
    """Create Plotly Dash dashboard."""

    '''Create Dash app'''
    dash_app = Dash(
        server=server,
        url_base_pathname=URL_BASE,
        suppress_callback_exceptions=True,
        external_stylesheets=['/static/style.css'],
        title='Matching 60% land costs with corporate loans'
    )

    '''Dash app layout'''
    def serve_layout():
        # Initial matching with default config, and extract the required data to be placed in initial Dash layout
        # Load default config
        with open(DASH_CONFIG_FILEPATH, 'r') as cf:
            dash_config = yaml.safe_load(cf)

        # Run matching
        init_matching_object = LoanMatching(dash_config)

        # Extract required data to be placed in initial Dash layout
        all_stage_dicts = list(init_matching_object.STAGED_OUTPUTS_METADATA['stages'].values())
        all_stages = [stage_dict['value'] for stage_dict in all_stage_dicts]

        all_projects = init_matching_object.MASTER_OUTPUT['project_name'].dropna().unique()

        uc_repl_opts_ = [
            {'label': 'UC evergreen',
             'value': 'input_uc_evergreen'},
            {'label': 'UC need to fully cover Committed Revolver',
             'value': 'input_uc_full_cover'},
            {'label': 'Check replacement by net margin diff. x area (by net margin diff. only if not selected)',
             'value': 'input_uc_check_saving_by_area'},
        ]
        uc_repl_default_values_ = [
            [x['value'] for x in uc_repl_opts_][i]
            for i, y in enumerate([
                init_matching_object.UC_EVERGREEN,
                init_matching_object.UC_FULL_COVER,
                init_matching_object.UC_CHECK_SAVING_BY_AREA
            ])
            if y
        ]  # ['input_uc_evergreen', 'input_uc_check_saving_by_area']

        ttl_cr_amt = sum([v['facility_amount_inB']
                          for v in init_matching_object.STAGED_OUTPUTS['stage 0']['fac'].values()
                          if (v['loan_sub_type'] == 'R') and (v['committed'] == 'C') and
                          (v['target_prepayment_date_idx'] >= 0)])
        ttl_cr_amt_str = '%.3f' % ttl_cr_amt

        init_matching_object_pkl_str = codecs.encode(pickle.dumps(init_matching_object), "base64").decode()

        # Structure the layout
        layout = \
            html.Div([
                html.H3('Matching 60% land costs with corporate loans'),

                # Panels
                html.Div([
                    # (1) Left: Matching parameters selection panel
                    html.Div([
                        html.Div([
                            html.H4('Matching parameters'),
                        ]),
                        html.Div([
                            # Target prepayment date delta (tpp_date_delta_ymd)
                            html.Label('Target prepayment date delta: ', style={'font-weight': 'bold'}),
                            dcc.Input(id='tpp-date-delta-year', type='number', placeholder='Year',
                                      value=init_matching_object.TPP_DATE_DELTA_YMD[0], min=-99, max=0, step=1,
                                      style={'width': '3vw'}),
                            html.Label(' years '),
                            dcc.Input(id='tpp-date-delta-month', type='number', placeholder='Month',
                                      value=init_matching_object.TPP_DATE_DELTA_YMD[1], min=-11, max=0, step=1,
                                      style={'width': '3vw'}),
                            html.Label(' months '),
                            dcc.Input(id='tpp-date-delta-day', type='number', placeholder='Day',
                                      value=init_matching_object.TPP_DATE_DELTA_YMD[2], min=-31, max=0, step=1,
                                      style={'width': '3vw'}),
                            html.Label(' days '),
                            html.Br(),
                        ]),
                        html.Div([
                            # Equity
                            html.Label('Equity: ', style={'font-weight': 'bold'}),
                            html.Label('HK$'),
                            dcc.Input(id='input-equity-amt', type='number', placeholder='Equity',
                                      value=init_matching_object.DASH_CONFIG['equity']['amt_in_billion'],
                                      style={'width': '4vw'}),
                            html.Label('B'),
                            html.Br(),
                        ]),
                        html.Div([
                            # Uncommitted Revolver options
                            html.Label('Uncommitted Revolver (UC) replacement options: ', style={'font-weight': 'bold'}),
                            dcc.Checklist(id='input-uc-options',
                                          options=uc_repl_opts_,
                                          value=uc_repl_default_values_,
                                          labelStyle={'display': 'block'}),
                        ]),
                        html.Div([
                            # Set aside committed revolver for acquisition
                            html.Label('Set aside committed revolver: ', style={'font-weight': 'bold'}),
                            html.Label('(total amount is HK$' + ttl_cr_amt_str + 'B)', style={'font-size': '12px'}),
                            html.Br(),

                            html.Label('Set aside max. HK$', style={'vertical-align': 'middle'}),
                            dcc.Input(id='cr-ceiling', type='number',
                                      value=init_matching_object.REVOLVER_CEILING, min=0, max=99999,
                                      style={'width': '4vw', 'vertical-align': 'middle'}),
                            html.Label('B revolver with ', style={'vertical-align': 'middle'}),
                            dcc.Dropdown(id='cr-ceiling-to-stay',
                                         options=[{'value': 'max_cost', 'label': 'highest cost'},
                                                  {'value': 'min_cost', 'label': 'lowest cost'},
                                                  {'value': 'max_area', 'label': 'largest area'},
                                                  {'value': 'min_area', 'label': 'smallest area'},
                                                  {'value': 'max_amount', 'label': 'highest loan amount'},
                                                  {'value': 'min_amount', 'label': 'lowest loan amount'},
                                                  {'value': 'max_period', 'label': 'longest period'},
                                                  {'value': 'min_period', 'label': 'shortest period'},
                                                  {'value': 'max_net_margin', 'label': 'highest net margin'},
                                                  {'value': 'min_net_margin', 'label': 'lowest net margin'}],
                                         value=init_matching_object.REVOLVER_TO_STAY,
                                         clearable=False,
                                         style={'width': '10vw',
                                                'display': 'inline-block', 'vertical-align': 'middle'}),
                            html.Label(' for ', style={'vertical-align': 'middle'}),
                            dcc.Dropdown(id='cr-ceiling-for',
                                         options=[{'value': 'loan_matching', 'label': 'loan matching'},
                                                  {'value': 'acquisition', 'label': 'acquisition'}],
                                         value=init_matching_object.REVOLVER_CEILING_FOR,
                                         clearable=False,
                                         style={'width': '10vw',
                                                'display': 'inline-block', 'vertical-align': 'middle'}),
                            html.Br(),


                        ]),
                        html.Div([
                            # Button to rerun matching
                            html.Button('Rerun matching', id='btn-rerun-matching', className='button'),
                            # Button to refresh figure
                            # html.Button('Refresh chart', id='btn-refresh-chart', className='button'),
                            # Button to show/ hide matching scheme
                            html.Button('Show/ hide matching scheme', id='btn-show-scheme', className='button'),
                        ]),
                    ], className='column left-column card'),
                    # (2) Right: Interactive visualization panel
                    html.Div([
                        html.Div([
                            html.H4('Interactive visualization'),
                        ]),
                        html.Div([
                            html.Label('Stage: ', style={'font-weight': 'bold'}),
                            dcc.Dropdown(id='filter-stage',
                                         options=all_stage_dicts,
                                         value=all_stages[0],
                                         clearable=False,
                                         style={'width': '36vw'}),
                        ]),
                        html.Div([
                            html.Label('Projects: ', style={'font-weight': 'bold'}),
                            dcc.Dropdown(id='filter-projects',
                                         options=all_projects,
                                         value=all_projects,
                                         multi=True,
                                         clearable=False),
                        ]),
                        html.Div([
                            html.Label('Chart type: ', style={'font-weight': 'bold'}),
                            dcc.RadioItems(
                                id='chart-type',
                                options=[{'label': '1. Simple Gantt chart', 'value': 13},
                                         {'label': '2. Gantt chart with diff. bar height', 'value': 23}],
                                value=13,
                                inline=True,
                                style={'display': 'inline-block'},
                            ),
                        ]),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Label('Bar height', style={'font-weight': 'bold'}, className='side-by-side'),
                                    dcc.Slider(20, 100, value=30, included=True, marks=None, id='bar-height'),
                                ], className='column'),
                                html.Div([
                                    html.Label('Bar height range ',
                                               style={'font-weight': 'bold'}, className='side-by-side'),
                                    html.Label('(for chart type #2 only)'),
                                    dcc.RangeSlider(0, 1.5, step=0.05, value=[0.25, 1], marks=None,
                                                    id='bar-height-range'),
                                ], className='column'),
                            ], className='row'),
                        ]),
                    ], className='column right-column card'),
                ], className='row'),

                # (3) Bottom/ Toggle: Matching scheme
                html.Div([
                    html.Div([
                        html.H4('Current matching scheme'),
                        html.Div(id='selected-input-display'),
                    ], className='full-span-column card'),
                ], className='row', id='matching-scheme-info-box'),

                # (4) Graph
                html.Div(dcc.Graph(id='figure-output'), className='card'),

                # (5) Store
                dcc.Store(id='current-matching-object', data={'current_matching_object': init_matching_object_pkl_str}),
            ])

        return layout

    dash_app.layout = serve_layout

    '''Callbacks'''
    @dash_app.callback(
        Output('selected-input-display', 'children'),
        Output('figure-output', 'figure'),
        Output('current-matching-object', 'data'),
        Input('current-matching-object', 'data'),
        Input('filter-stage', 'value'),
        Input('filter-projects', 'value'),
        Input('chart-type', 'value'),
        Input('bar-height', 'value'),
        Input('bar-height-range', 'value'),
        Input('btn-rerun-matching', 'n_clicks'),
        # Input('btn-refresh-chart', 'n_clicks'),
        State('tpp-date-delta-year', 'value'),
        State('tpp-date-delta-month', 'value'),
        State('tpp-date-delta-day', 'value'),
        State('input-equity-amt', 'value'),
        State('input-uc-options', 'value'),
        State('cr-ceiling', 'value'),
        State('cr-ceiling-to-stay', 'value'),
        State('cr-ceiling-for', 'value'),
    )
    def update_plot(*args):
        triggered_id = ctx.triggered_id
        current_matching_object_json, \
            stage_to_display, projects_to_display, chart_type, bar_height, bar_height_range, _, \
            tpp_date_delta_year, tpp_date_delta_month, tpp_date_delta_day, \
            input_equity_amt, input_uc_options,\
            cr_ceiling, cr_ceiling_to_stay, cr_ceiling_for = args

        # Load matching_object
        matching_object_pkl_str = current_matching_object_json['current_matching_object']
        matching_object = pickle.loads(codecs.decode(matching_object_pkl_str.encode(), "base64"))

        # Set default values if None and assign back to matching object
        tpp_date_delta_year = tpp_date_delta_year if tpp_date_delta_year is not None else -1
        tpp_date_delta_month = tpp_date_delta_month if tpp_date_delta_year is not None else 0
        tpp_date_delta_day = tpp_date_delta_day if tpp_date_delta_day is not None else 0
        matching_object.TPP_DATE_DELTA_YMD = (tpp_date_delta_year, tpp_date_delta_month, tpp_date_delta_day)

        equity_amt = input_equity_amt if input_equity_amt is not None else 40.2
        matching_object.DASH_CONFIG['equity']['amt_in_billion'] = equity_amt

        uc_evergreen = 'input_uc_evergreen' in input_uc_options
        matching_object.UC_EVERGREEN = uc_evergreen

        uc_full_cover = 'input_uc_full_cover' in input_uc_options
        matching_object.UC_FULL_COVER = uc_full_cover

        uc_saving_by_area = 'input_uc_check_saving_by_area' in input_uc_options
        matching_object.UC_CHECK_SAVING_BY_AREA = uc_saving_by_area

        cr_ceiling = cr_ceiling if cr_ceiling is not None else 99999.0
        matching_object.REVOLVER_CEILING = cr_ceiling

        cr_ceiling_for = cr_ceiling_for if cr_ceiling_for is not None else 'loan_matching'
        matching_object.REVOLVER_CEILING_FOR = cr_ceiling_for

        cr_ceiling_to_stay = cr_ceiling_to_stay if cr_ceiling_to_stay is not None else 'max_cost'
        matching_object.REVOLVER_TO_STAY = cr_ceiling_to_stay

        # If clicked the rerun matching button, reload data and rerun matching and update master data
        if triggered_id == 'btn-rerun-matching':
            matching_object.load_data()
            matching_object.preprocess_data()
            matching_object.init_working_data()
            matching_object.matching_main_proc(scheme=matching_object.DASH_CONFIG['matching_scheme']['scheme_id'])
        else:
            pass

        # Matching scheme display
        ms_display = list()
        # li 1
        ms_display.append(
            html.Li(matching_object.STAGED_OUTPUTS_METADATA.get('short_description', '[description missing]'))
        )
        # li 2
        ms_display.append(
            html.Li(f'Target prepayment date for Term and Committed Revolver = Expiry date '
                    f'{tpp_date_delta_year} years {tpp_date_delta_month} months {tpp_date_delta_day} days')
        )
        # li 3
        if uc_evergreen:
            ms_display.append(html.Li('UC evergreen (no expiry date) assumed'))
        else:
            ms_display.append(html.Li('UC evergreen (no expiry date) not assumed'))
        # li 4
        ms_display_text_uc_rep = 'For UC replacement, '
        if uc_full_cover:
            ms_display_text_uc_rep += 'UC has to fully cover the matched Committed Revolver entry, '
        else:
            ms_display_text_uc_rep += 'UC may partially cover the matched Committed Revolver entry, '
        if uc_saving_by_area:
            ms_display_text_uc_rep += 'and saving is calculated by net_margin difference x overlapping area'
        else:
            ms_display_text_uc_rep += 'and saving is calculated by net_margin difference only'
        ms_display.append(html.Li(ms_display_text_uc_rep))
        # li 5
        ms_display_text_cr_ceiling = 'Set aside '
        if cr_ceiling == 99999:
            ms_display_text_cr_ceiling += 'all revolver with '
        else:
            ms_display_text_cr_ceiling += ('HK$' + str(cr_ceiling) + 'B revolver with ')
        ms_display_text_cr_ceiling += {
            'max_cost': 'highest cost',
            'min_cost': 'lowest cost',
            'max_area': 'largest area',
            'min_area': 'smallest area',
            'max_amount': 'highest loan amount',
            'min_amount': 'lowest loan amount',
            'max_period': 'longest period',
            'min_period': 'shortest period',
            'max_net_margin': 'highest net margin',
            'min_net_margin': 'lowest net margin'
        }[cr_ceiling_to_stay]
        ms_display_text_cr_ceiling += (' for '+ cr_ceiling_for.replace('_', ' '))
        ms_display.append(html.Li(ms_display_text_cr_ceiling))

        # Render chart
        # Set default as 13
        if chart_type not in [13, 23]:
            chart_type = 13

        if chart_type == 11:
            # DEPRECATED
            # fig = get_gantt_1(stage_to_display, projects_to_display)
            pass
        elif chart_type == 12:
            # DEPRECATED
            # fig = get_gantt_2(stage_to_display, projects_to_display)
            pass
        elif chart_type == 13:
            fig = get_gantt_3(matching_object, stage_to_display, projects_to_display,
                              bar_height=bar_height)
        elif chart_type == 23:
            fig = get_gantt_diff_bar_width_3(matching_object, stage_to_display, projects_to_display,
                                             bar_height=bar_height, trace_height_range=tuple(bar_height_range))
        else:
            pass

        # End of callback, encode the final matching object and update to dcc.Store
        matching_object_pkl_str = codecs.encode(pickle.dumps(matching_object), "base64").decode()

        return html.Ol(ms_display), fig, {'current_matching_object': matching_object_pkl_str}

    @dash_app.callback(
        Output('matching-scheme-info-box', 'style'),
        Input('btn-show-scheme', 'n_clicks')
    )
    def toggle_matching_scheme_info_box(n_clicks):
        if n_clicks is None or n_clicks % 2 == 0:
            return {'display': 'none'}
        else:
            return {'display': 'block'}

    return server


