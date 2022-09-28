import base64
import datetime as dt
import io
import numpy as np
import pandas as pd
import re
from dash import Dash, dcc, html, dash_table, Input, Output, State
from pandas_schema import Column, Schema
from pandas_schema.validation import LeadingWhitespaceValidation, TrailingWhitespaceValidation, IsDistinctValidation, \
    IsDtypeValidation, CanConvertValidation, MatchesPatternValidation, InRangeValidation, InListValidation

URL_BASE = '/upload_file/'
TODAY = dt.datetime.today()
INPUT_DATA_DIR = './app/dash/data/input/'
PROJECT_DATA_FILENAME = 'project_data.xlsx'
FACILITY_DATA_FILENAME = 'facility_data.xlsx'


def add_dash_app(server):
    """Create Plotly Dash dashboard."""

    dash_app = Dash(
            server=server,
            url_base_pathname=URL_BASE,
            suppress_callback_exceptions=True,
            external_stylesheets=['/static/style.css']
        )

    '''Dash app layout'''
    dash_app.layout = html.Div([
        # Project
        html.Label('Upload loan data file(s)', style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Acceptable file formats: Excel with worksheets "projects", CSV'),
        dcc.Upload(
            id='upload-excel-project',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Not allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='data-uploaded-project'),
        html.Hr(),  # horizontal line

        # Loan
        html.Label('Upload project data file', style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Acceptable file formats:'),
        html.Ul(
            html.Li('1 Excel file with worksheet name "tbl_company_grp", "tbl_company", '
                    '"tbl_lender", "tbl_loan", and "tbl_loan_facility"'),
            html.Li('4-5 CSV files with file names containing "company_grp" (optional), "company", '
                    '"lender", "loan", "loan_facility"'),
        ),
        dcc.Upload(
            id='upload-excel-loan',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Div(id='data-uploaded-loan'),
    ])

    '''Dash app callbacks'''
    @dash_app.callback(Output('data-uploaded-project', 'children'),
                       Input('upload-excel-project', 'contents'),
                       State('upload-excel-project', 'filename'),
                       State('upload-excel-project', 'last_modified'))
    def process_project_file_uploaded(content, filename, last_modified_date):
        """Parse project file content, upload to file server and return the HTML content for display"""
        if content is not None:
            # content_type = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64'
            # content_string = 'UEsDBBQABgAIAAAAIQDKjBQWmwEAALkIAAATAAgCW0NvbnRlbnRfVHlwZXNdLnhtbCCiBAIooAA...'
            # decoded = b'PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00\xca\x8c\x14\x16\x9b\x01\x00\x00...'
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)

            # 1. Data validation, any error found in this stage will abort the upload
            # 1.1. File
            try:
                if 'xls' in filename:
                    try:
                        project_df = pd.read_excel(io.BytesIO(decoded), sheet_name='projects')
                    except Exception as e:
                        print(e)
                        return html.Div(["Worksheet 'projects' not found."])
                elif 'csv' in filename:
                    try:
                        project_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    except Exception as e:
                        print(e)
                        return html.Div(["Failed to load CSV file."])
                else:
                    return html.Div(['The uploaded file is not Excel/ CSV'])
            except Exception as e:
                print(e)
                return html.Div(['Unknown error with the file uploaded.'])

            # 1.2. Data fields
            required_fields = ['project_id', 'project_name', 'start_date', 'end_date', 'land_cost_60_pct_inB',
                               'solo_jv']

            # 1.2.1. Check if required fields are all present
            if not set(project_df.columns).issuperset(set(required_fields)):
                missing_fields = set(required_fields).difference(set(project_df.columns))
                return html.Div([f'Column(s) {", ".join(missing_fields)} not found.'])

            # 1.2.2. Value check
            # Missing value check
            value_error_strs = []
            for colname in ['project_id', 'project_name', 'start_date', 'end_date']:
                if not all(project_df[colname].notnull()):
                    value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

            active_project_df = project_df[(project_df['start_date'] <= TODAY) & (project_df['end_date'] >= TODAY)]
            for colname in ['land_cost_60_pct_inB', 'solo_jv']:
                if not all(active_project_df[colname].notnull()):
                    value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

            # Valid value check
            valid_value_errors = Schema([
                Column('solo_jv', [InListValidation(['Solo', 'JV'])]),
            ]).validate(active_project_df[['solo_jv']])

            for error in valid_value_errors:
                value_error_strs.append(f'column: "{error.column}", row: {error.row + 1}, '
                                        f'value: "{error.value}" - {error.message}')

            if len(value_error_strs) > 0:
                return html.Div(['<br />'.join(value_error_strs)])

            # 2. Data cleanness check, error found in this stage will NOT abort the upload
            data_warning_str = ''
            # 2.1. Schema check, including data type check
            project_df_schema = Schema([
                Column('project_id', [IsDtypeValidation(np.int64), IsDistinctValidation()]),
                Column('project_name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                Column('start_date', [IsDtypeValidation(np.datetime64)]),
                Column('end_date', [IsDtypeValidation(np.datetime64)]),
                Column('land_cost_60_pct_inB', [IsDtypeValidation(np.float64)]),
                Column('solo_jv', [InListValidation(['Solo', 'JV'])]),
            ])
            # InRangeValidation(0, 120)
            # MatchesPatternValidation(r'\d{4}[A-Z]{4}')

            schema_check_warnings = project_df_schema.validate(project_df[required_fields])

            schema_check_warning_strs = [
                f'column: "{w.column}", row: {w.row + 1}, value: "{w.value}" - {w.message}'
                for w in schema_check_warnings
            ]
            if len(schema_check_warning_strs) > 0:
                data_warning_str += 'WARNING: <br />' + '<br />'.join(schema_check_warning_strs)

            # 3. Pre-process and upload file
            dt_to_date_fields = ['start_date', 'end_date']
            for field in dt_to_date_fields:
                project_df[field] = project_df[field].dt.date
            with pd.ExcelWriter(INPUT_DATA_DIR + PROJECT_DATA_FILENAME) as writer:
                project_df.to_excel(writer, sheet_name='projects', index=False, freeze_panes=(1, 0))

            return html.Div([
                html.Div('File uploaded: ' + filename),
                html.Pre(data_warning_str),
                html.Div('Preview - "projects"'),
                dash_table.DataTable(
                    project_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in project_df.columns]
                ),
                # html.H6(dt.datetime.fromtimestamp(last_modified_date)),
                # For debugging, display the raw content provided by the web browser
                # html.Pre(content[0:200] + '...', style={
                #     'whiteSpace': 'pre-wrap',
                #     'wordBreak': 'break-all'
                # })
            ])

    @dash_app.callback(Output('data-uploaded-loan', 'children'),
                       Input('upload-excel-loan', 'contents'),
                       State('upload-excel-loan', 'filename'))
    def process_loan_file_uploaded(contents, filenames):
        """Parse loan file content, upload to file server and return the HTML content for display"""
        if contents is not None:
            # content_type = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64'
            # content_string = 'UEsDBBQABgAIAAAAIQDKjBQWmwEAALkIAAATAAgCW0NvbnRlbnRfVHlwZXNdLnhtbCCiBAIooAA...'
            # decoded = b'PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00\xca\x8c\x14\x16\x9b\x01\x00\x00...'

            warning_strs = []

            # 1. Data validation
            # 1.1. Uploaded files validation, error found in this stage will abort the upload
            # Check if either 1 Excel or 4-5 CSV files are uploaded
            file_count_failed = True
            file_input_type = None
            if len(contents) == 1 and 'xls' in filenames[0]:
                file_count_failed = False
                file_input_type = 'Excel'
            if 4 <= len(contents) <= 5 and all(['csv' in filename for filename in filenames]):
                file_count_failed = False
                file_input_type = 'CSVs'
            if file_count_failed:
                return html.Div(['Upload either 1 Excel or 4-5 CSV files.'])

            # Extract raw data tables, if a table is not found, extract the previous version of data table;
            # If the previous version is also unavailable, abort the file upload.
            company_grp_df = pd.DataFrame()
            company_df = pd.DataFrame()
            lender_df = pd.DataFrame()
            loan_df = pd.DataFrame()
            facility_df = pd.DataFrame()
            df_names = ['company_grp_df', 'company_df', 'lender_df', 'loan_df', 'facility_df']
            ws_names = ['tbl_company_grp', 'tbl_company', 'tbl_lender', 'tbl_loan', 'tbl_loan_facility']
            if file_input_type == 'Excel':
                content_type, content_string = contents[0].split(',')
                decoded = base64.b64decode(content_string)
                for ws_name, df_name in zip(ws_names, df_names):
                    try:
                        exec(f'{df_name} = pd.read_excel(io.BytesIO(decoded), sheet_name="{ws_name}")')
                    except Exception as e:
                        print(e)
                        warning_strs.append(f'worksheet "{ws_name}" not found, use the previous data instead.')
                    finally:
                        try:
                            exec(f'{df_name} = pd.read_excel(INPUT_DATA_DIR + FACILITY_DATA_FILENAME, '
                                 f'sheet_name="{ws_name}")')
                        except Exception as e:
                            print(e)
                            return html.Div([f'worksheet "{ws_name}" not found, '
                                             f'and the previous data ({FACILITY_DATA_FILENAME}) is unavailable.'])

            elif file_input_type == 'CSVs':
                for content, filename in zip(contents, filenames):
                    content_type, content_string = content.split(',')
                    decoded = base64.b64decode(content_string)
                    df_name, ws_name = '', ''
                    # Guess the DataFrame to load data into, based on the CSV filename
                    if re.findall(r'company_grp', filename):
                        df_name, ws_name = 'company_grp_df', 'tbl_company_grp'
                    elif re.findall(r'company', filename):
                        df_name, ws_name = 'company_df', 'tbl_company'
                    elif re.findall(r'lender', filename):
                        df_name, ws_name = 'lender_df', 'tbl_lender'
                    elif re.findall(r'facility', filename):
                        df_name, ws_name = 'facility_df', 'tbl_loan_facility'
                    elif re.findall(r'loan', filename):
                        df_name, ws_name = 'loan_df', 'tbl_loan'
                    else:
                        pass
                    try:
                        exec(f"{df_name} = pd.read_csv(io.StringIO(decoded.decode('utf-8')))")
                    except Exception as e:
                        print(e)
                        warning_strs.append(f'Failed to load "{filename}", use the previous data instead.')
                    finally:
                        try:
                            exec(f'{df_name} = pd.read_excel(INPUT_DATA_DIR + FACILITY_DATA_FILENAME, '
                                 f'sheet_name="{ws_name}")')
                        except Exception as e:
                            print(e)
                            return html.Div([f'Failed to load "{filename}", '
                                             f'and the previous data ({FACILITY_DATA_FILENAME}) is unavailable.'])
                    # Check if any DataFrame is empty, and load previous data if yes
                    for ws_name, df_name in zip(ws_names, df_names):
                        is_empty = False
                        exec(f'is_empty = {df_name}.empty')
                        if is_empty:
                            try:
                                exec(f'{df_name} = pd.read_excel(INPUT_DATA_DIR + FACILITY_DATA_FILENAME, '
                                     f'sheet_name="{ws_name}")')
                            except Exception as e:
                                print(e)
                                return html.Div([f'{df_name} data is not uploaded, '
                                                 f'and the previous data ({FACILITY_DATA_FILENAME}) is unavailable.'])
            else:
                # Impossible to reach here
                pass

            # At this point, the 5 DataFrame should have data, continue to data validation/ checking

            # 1.2. Data fields
            required_fields = {
                'company_grp_df',
                'company_df',
                'lender_df',
                'loan_df',
                'facility_df'
            }


            data_warning_str = 'WARNING: <br />' + '<br />'.join(warning_strs)

            # TODO: Up to here

            if contents is not None:
                # 0. Initialization
                # Check if either one Excel
                # Set general variables



                for content, filename in zip(contents, filenames):
                    content_type, content_string = content.split(',')
                    decoded = base64.b64decode(content_string)

                    # 1. Data validation, any error found in this stage will abort the upload


                    # 1.2. Data fields
                    required_fields = ['loan_id', 'loan_name', 'start_date', 'end_date', 'land_cost_60_pct_inB',
                                       'solo_jv']

                    # 1.2.1. Check if required fields are all present
                    if not set(loan_df.columns).issuperset(set(required_fields)):
                        missing_fields = set(required_fields).difference(set(loan_df.columns))
                        return html.Div([f'Column(s) {", ".join(missing_fields)} not found.'])

                    # 1.2.2. Value check
                    # Missing value check
                    value_error_strs = []
                    for colname in ['loan_id', 'loan_name', 'start_date', 'end_date']:
                        if not all(loan_df[colname].notnull()):
                            value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

                    active_loan_df = loan_df[(loan_df['start_date'] <= TODAY) & (loan_df['end_date'] >= TODAY)]
                    for colname in ['land_cost_60_pct_inB', 'solo_jv']:
                        if not all(active_loan_df[colname].notnull()):
                            value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

                    # Valid value check
                    valid_value_errors = Schema([
                        Column('solo_jv', [InListValidation(['Solo', 'JV'])]),
                    ]).validate(active_loan_df[['solo_jv']])

                    for error in valid_value_errors:
                        value_error_strs.append(f'column: "{error.column}", row: {error.row + 1}, '
                                                f'value: "{error.value}" - {error.message}')

                    if len(value_error_strs) > 0:
                        return html.Div(['<br />'.join(value_error_strs)])

                    # 2. Data cleanness check, error found in this stage will NOT abort the upload
                    data_warning_str = ''
                    # 2.1. Schema check, including data type check
                    loan_df_schema = Schema([
                        Column('loan_id', [IsDtypeValidation(np.int64), IsDistinctValidation()]),
                        Column('loan_name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                        Column('start_date', [IsDtypeValidation(np.datetime64)]),
                        Column('end_date', [IsDtypeValidation(np.datetime64)]),
                        Column('land_cost_60_pct_inB', [IsDtypeValidation(np.float64)]),
                        Column('solo_jv', [InListValidation(['Solo', 'JV'])]),
                    ])
                    # InRangeValidation(0, 120)
                    # MatchesPatternValidation(r'\d{4}[A-Z]{4}')

                    schema_check_warnings = loan_df_schema.validate(loan_df[required_fields])

                    schema_check_warning_strs = [
                        f'column: "{w.column}", row: {w.row + 1}, value: "{w.value}" - {w.message}'
                        for w in schema_check_warnings
                    ]
                    if len(schema_check_warning_strs) > 0:
                        data_warning_str += 'WARNING: ' + '<br />'.join(schema_check_warning_strs)

                    # 3. Pre-process and upload file
                    dt_to_date_fields = ['start_date', 'end_date']
                    for field in dt_to_date_fields:
                        loan_df[field] = loan_df[field].dt.date
                    with pd.ExcelWriter(INPUT_DATA_DIR + PROJECT_DATA_FILENAME) as writer:
                        loan_df.to_excel(writer, sheet_name='loans', index=False, freeze_panes=(1, 0))

                    return html.Div([
                        html.Div('File uploaded: ' + filename),
                        html.Pre(data_warning_str),
                        html.Div('Preview - "loans"'),
                        dash_table.DataTable(
                            loan_df.to_dict('records'),
                            [{'name': i, 'id': i} for i in loan_df.columns]
                        ),
                        # html.H6(dt.datetime.fromtimestamp(last_modified_date)),
                        # For debugging, display the raw content provided by the web browser
                        # html.Pre(content[0:200] + '...', style={
                        #     'whiteSpace': 'pre-wrap',
                        #     'wordBreak': 'break-all'
                        # })
                    ])

    return server


# self.FACILITIES_DF = self.FACILITIES_DF[[
#     'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche',
#     'currency_id', 'facility_amount', 'available_period_from',
#     'available_period_to',
#     'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
#     'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
#     'facility_date', 'expiry_date', 'withdrawn',
#     'short_code', 'name', 'short_form',
#     'company_name', 'company_short_name', 'type',
#     'margin',
#     'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
#     'comm_fee_margin_below', 'all_in_price', 'net_margin',
#     'upfront_fee'
# ]].copy()

