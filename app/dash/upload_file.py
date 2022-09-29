import base64
import datetime as dt
import io, shutil
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
BACKUP_DATA_DIR = './app/dash/data/input/backup/'
PROJECT_DATA_FILENAME = 'project_data.xlsx'
BTS_DATA_FILENAME = 'bts_data.xlsx'


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
        html.Hr(),  # horizontal line

        # Loan
        html.Label('Upload project data file', style={'font-weight': 'bold'}),
        html.Br(),
        html.Label('Acceptable file formats:'),
        html.Ul([
            html.Li('[Recommended] 4-5 CSV files with file names containing "company_grp" (optional), "company", '
                    '"lender", "loan", "loan_facility"'),
            html.Li('1 Excel file with worksheet names (must be exactly) "tbl_company_grp", "tbl_company", '
                    '"tbl_lender", "tbl_loan", and "tbl_loan_facility"'),
        ]),
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
        html.Hr(),  # horizontal line

        # Data preview
        html.Label('Data preview', style={'font-weight': 'bold'}),
        html.Div(id='data-uploaded-project'),
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
            # 1.1. Imported file
            try:
                if 'xls' in filename:
                    try:
                        project_df = pd.read_excel(io.BytesIO(decoded), sheet_name='projects')
                    except Exception as e:
                        print(e)
                        return html.Pre(["ERROR: Worksheet 'projects' not found."])
                elif 'csv' in filename:
                    try:
                        project_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    except Exception as e:
                        print(e)
                        return html.Pre(["ERROR: Failed to load CSV file."])
                else:
                    return html.Pre(['ERROR: The uploaded file is not Excel/ CSV'])
            except Exception as e:
                print(e)
                return html.Pre(['ERROR: Unknown error with the file uploaded.'])

            # 1.2. Data fields
            # 1.2.1. Check if required fields are all present
            required_fields = ['project_id', 'project_name', 'start_date', 'end_date',
                               'land_cost_60_pct_inB', 'solo_jv']
            if not set(project_df.columns).issuperset(set(required_fields)):
                missing_fields = set(required_fields).difference(set(project_df.columns))
                return html.Pre([f'ERROR: Column(s) {", ".join(missing_fields)} not found.'])

            # 1.2.2. Value check
            value_error_strs = []
            # Missing value check
            for colname in ['project_id', 'project_name', 'start_date', 'end_date']:
                if not all(project_df[colname].notnull()):
                    value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

            active_project_df = project_df[(project_df['start_date'] <= TODAY) & (project_df['end_date'] >= TODAY)]
            for colname in ['land_cost_60_pct_inB', 'solo_jv']:
                if not all(active_project_df[colname].notnull()):
                    value_error_strs.append(f'There are missing value(s) in column: "{colname}"')

            # Valid value check
            valid_value_schema = Schema([
                Column('project_id', [IsDistinctValidation()]),
                Column('solo_jv', [InListValidation(['Solo', 'JV'])]),
            ])
            valid_value_errors = valid_value_schema.validate(active_project_df[['project_id', 'solo_jv']])

            for error in valid_value_errors:
                value_error_strs.append(f'column: "{error.column}", row: {error.row + 1}, '
                                        f'value: "{error.value}" - {error.message}')

            # Abort upload
            if len(value_error_strs) > 0:
                return html.Pre(['ERROR: \n' + '\n'.join(value_error_strs)])

            # 2. Data cleanness check, warning found in this stage will NOT abort the upload
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

            clean_value_warnings = project_df_schema.validate(project_df[required_fields])

            clean_value_warning_strs = [
                f'column: "{w.column}", row: {w.row + 1}, value: "{w.value}" - {w.message}'
                for w in clean_value_warnings
            ]

            data_warning_str = ''
            if len(clean_value_warning_strs) > 0:
                data_warning_str = 'WARNING: \n' + '\n'.join(clean_value_warning_strs)

            # 3. Pre-process and upload file
            for field in ['start_date', 'end_date']:
                try:
                    project_df[field] = project_df[field].dt.date
                except Exception as e:
                    print(e)
                    try:
                        project_df[field] = project_df[field].apply(
                            lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date())
                    except Exception as e:
                        print(e)
                        data_warning_str += f'\nTable: "project" - Unknown date format in column "{field}"'
            with pd.ExcelWriter(INPUT_DATA_DIR + PROJECT_DATA_FILENAME) as writer:
                project_df.to_excel(writer, sheet_name='projects', index=False, freeze_panes=(1, 0))

            # Save a backup
            now = dt.datetime.now().strftime("%Y%m%d_%H%M")
            shutil.copyfile(
                INPUT_DATA_DIR + PROJECT_DATA_FILENAME,
                BACKUP_DATA_DIR + PROJECT_DATA_FILENAME.replace('.xlsx', '') + '_' + now + '.xlsx'
            )

            # 4. Final return
            return html.Div([
                html.Div('File uploaded: ' + filename),
                html.Pre(data_warning_str),
                html.Div('Preview - "project"'),
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
            uploaded_filenames = []

            # 1. Data validation
            # 1.1. Imported files validation, error found in this stage will abort the upload,
            #  warning found in this stage will not abort the upload but warning messages will be given
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
                return html.Pre(['ERROR: Upload either 1 Excel or 4-5 CSV files.'])

            # Extract raw data tables, if a table is not found, extract the previous version of data table
            # Initialize empty DataFrames
            dfs = {
                'company_grp_df': pd.DataFrame(),
                'company_df': pd.DataFrame(),
                'lender_df': pd.DataFrame(),
                'loan_df': pd.DataFrame(),
                'facility_df': pd.DataFrame()
            }
            df_names = list(dfs.keys())
            ws_names = ['tbl_company_grp', 'tbl_company', 'tbl_lender', 'tbl_loan', 'tbl_loan_facility']

            if file_input_type == 'Excel':
                content_type, content_string = contents[0].split(',')
                decoded = base64.b64decode(content_string)

                # Fill up the 5 DataFrames in top-down manner
                for ws_name, df_name in zip(ws_names, df_names):
                    try:  # Find the worksheet for data extraction
                        dfs[df_name] = pd.read_excel(io.BytesIO(decoded), sheet_name=ws_name)
                    except Exception as e:
                        print(e)
                        warning_strs.append(f'worksheet "{ws_name}" not found, use the previous data instead.')
                        try:  # Worksheet not found/ data extraction failed, find the previous data
                            dfs[df_name] = pd.read_excel(INPUT_DATA_DIR + BTS_DATA_FILENAME, sheet_name=ws_name)
                        except Exception as e:  # The previous data is also unavailable, abort upload
                            print(e)
                            return html.Pre([f'ERROR: Worksheet "{ws_name}" not found, '
                                             f'and the previous data ({BTS_DATA_FILENAME}) is unavailable.'])
                uploaded_filenames.append(filenames[0])

            elif file_input_type == 'CSVs':
                # Fill up DataFrames in bottom-up manner, e.g., depending on how much data is imported
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

                    try:  # Try data extraction
                        dfs[df_name] = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                        uploaded_filenames.append(filename)
                    except Exception as e:
                        print(e)
                        warning_strs.append(f'Failed to load "{filename}", use the previous data instead.')
                        try:  # Data extraction failed, find the previous data
                            dfs[df_name] = pd.read_excel(INPUT_DATA_DIR + BTS_DATA_FILENAME, sheet_name=ws_name)
                        except Exception as e:  # The previous data is also unavailable, abort upload
                            print(e)
                            return html.Pre([f'ERROR: Failed to load "{filename}", '
                                             f'and the previous data ({BTS_DATA_FILENAME}) is unavailable.'])
                    # Fill in the missing DataFrame: if any DataFrame is empty, then load the previous data
                    for ws_name, df_name in zip(ws_names, df_names):
                        if dfs[df_name].empty:
                            try:
                                dfs[df_name] = pd.read_excel(INPUT_DATA_DIR + BTS_DATA_FILENAME, sheet_name=ws_name)
                            except Exception as e:
                                print(e)
                                return html.Pre([f'ERROR: {df_name.replace("_df", "")} data is not uploaded, '
                                                 f'and the previous data ({BTS_DATA_FILENAME}) is unavailable.'])
            else:
                # Impossible to reach here
                pass

            # At this point, the 5 DataFrame should have data, continue to data validation/ checking
            # print(company_grp_df, company_df, lender_df, loan_df, facility_df)

            # 1.2. Data fields
            # 1.2.1. Check if required fields are all present
            required_fields_dict = {
                'company_grp_df': [],
                'company_df': ['sys_company_id', 'company_name', 'company_short_name', 'type'],
                'lender_df': ['sys_lender_id', 'short_code', 'name', 'short_form'],
                'loan_df': [
                    'sys_loan_id', 'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
                    'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
                    'facility_date', 'expiry_date', 'withdrawn', 'upfront_fee'
                ],
                'facility_df': [
                    'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche', 'currency_id',
                    'facility_amount', 'available_period_from', 'available_period_to',
                    'margin', 'comm_fee_amount', 'comm_fee_margin',
                    'comm_fee_margin_over', 'comm_fee_margin_below', 'all_in_price', 'net_margin'
                ]
            }
            for df_name, required_fields in required_fields_dict.items():
                df = dfs[df_name]
                if not set(df.columns).issuperset(set(required_fields)):
                    missing_fields = set(required_fields).difference(set(df.columns))
                    return html.Pre([
                        f'ERROR: Table {df_name.replace("_df", "")} - column(s) {", ".join(missing_fields)} not found.'
                    ])

            # 1.2.2. Value check
            value_error_strs = []
            # Missing value check
            no_missing_data_fields_dict = {
                'company_df': ['sys_company_id', 'company_name', 'company_short_name'],
                'lender_df': ['sys_lender_id', 'short_code', 'name', 'short_form'],
                'loan_df': [
                    'sys_loan_id', 'reference_id', 'borrower_id', 'lender_id',
                    'committed', 'loan_type', 'facility_date', 'expiry_date', 'withdrawn'
                ],
                'facility_df': [
                    'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'facility_amount',
                    'available_period_from', 'available_period_to', 'margin', 'net_margin'
                ]
            }
            for df_name, no_missing_data_fields in no_missing_data_fields_dict.items():
                tbl_name = df_name.replace("_df", "")
                df = dfs[df_name]
                for colname in no_missing_data_fields:
                    if not all(df[colname].notnull()):
                        value_error_strs.append(f'Table "{tbl_name}" - '
                                                f'there are missing value(s) in column: "{colname}"')

            # Valid value check
            valid_value_data_fields_dict = {
                'company_df': ['sys_company_id'],
                'lender_df': ['sys_lender_id'],
                'loan_df': ['sys_loan_id', 'borrower_id', 'lender_id'],
                'facility_df': ['sys_loan_facility_id', 'loan_id']
            }
            possible_borrower_ids = list(dfs['company_df']['sys_company_id'].unique())  # company/ project company IDs
            possible_lender_ids = list(dfs['lender_df']['sys_lender_id'].unique())
            possible_loan_ids = list(dfs['loan_df']['sys_loan_id'].unique())
            valid_value_schemas_dict = {
                'company_df': Schema([Column('sys_company_id', [IsDistinctValidation()])]),
                'lender_df': Schema([Column('sys_lender_id', [IsDistinctValidation()])]),
                'loan_df': Schema([
                    Column('sys_loan_id', [IsDistinctValidation()]),
                    Column('borrower_id', [InListValidation(possible_borrower_ids)]),
                    Column('lender_id', [InListValidation(possible_lender_ids)]),
                ]),
                'facility_df': Schema([
                    Column('sys_loan_facility_id', [IsDistinctValidation()]),
                    Column('loan_id', [InListValidation(possible_loan_ids)]),
                ]),
            }
            for df_name, valid_value_schema in valid_value_schemas_dict.items():
                tbl_name = df_name.replace("_df", "")
                df = dfs[df_name]
                df_subset = df[valid_value_data_fields_dict[df_name]]
                valid_value_errors = valid_value_schema.validate(df_subset)
                for error in valid_value_errors:
                    value_error_strs.append(f'Table: "{tbl_name}", column: "{error.column}", row: {error.row + 1}, '
                                            f'value: "{error.value}" - {error.message}')

            # Abort upload
            if len(value_error_strs) > 0:
                return html.Pre(['ERROR: \n' + '\n'.join(value_error_strs)])

            # 2. Data cleanness check, warning found in this stage will NOT abort the upload
            # 2.1. Schema check, including data type check
            clean_value_data_fields_dict = {
                'company_df': ['sys_company_id', 'company_name', 'company_short_name'],
                'lender_df': ['sys_lender_id', 'short_code', 'name', 'short_form'],
                'loan_df': [
                    'sys_loan_id', 'reference_id', 'borrower_id', 'lender_id',
                    'committed', 'loan_type', 'facility_date', 'expiry_date', 'withdrawn'
                ],
                'facility_df': [
                    'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'facility_amount',
                    'available_period_from', 'available_period_to', 'margin', 'net_margin'
                ]
            }
            clean_value_schemas_dict = {
                'company_df': Schema([
                    Column('sys_company_id', [IsDtypeValidation(np.int64)]),
                    Column('company_name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                    Column('company_short_name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                ]),
                'lender_df': Schema([
                    Column('sys_lender_id', [IsDtypeValidation(np.int64)]),
                    Column('short_code', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                    Column('name', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                    Column('short_form', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                ]),
                'loan_df': Schema([
                    Column('sys_loan_id', [IsDtypeValidation(np.int64)]),
                    Column('reference_id', [LeadingWhitespaceValidation(), TrailingWhitespaceValidation()]),
                    Column('borrower_id', [IsDtypeValidation(np.int64)]),
                    Column('lender_id', [IsDtypeValidation(np.int64)]),
                    Column('committed', [InListValidation(['C', 'U'])]),
                    Column('loan_type', [InListValidation(['C', 'P'])]),
                    Column('facility_date', [IsDtypeValidation(np.datetime64)]),
                    Column('expiry_date', [IsDtypeValidation(np.datetime64)]),
                    Column('withdrawn', [InListValidation(['Y', 'N'])]),
                ]),
                'facility_df': Schema([
                    Column('sys_loan_facility_id', [IsDtypeValidation(np.int64)]),
                    Column('loan_id', [IsDtypeValidation(np.int64)]),
                    Column('loan_sub_type', [InListValidation(['T', 'R'])]),
                    Column('facility_amount', [IsDtypeValidation(np.int64)]),
                    Column('available_period_from', [IsDtypeValidation(np.datetime64)]),
                    Column('available_period_to', [IsDtypeValidation(np.datetime64)]),
                    Column('margin', [IsDtypeValidation(np.float64)]),
                    Column('net_margin', [IsDtypeValidation(np.float64)]),
                ]),
            }
            for df_name, clean_value_schema in clean_value_schemas_dict.items():
                tbl_name = df_name.replace("_df", "")
                df = dfs[df_name]
                df_subset = df[clean_value_data_fields_dict[df_name]]
                clean_value_warnings = clean_value_schema.validate(df_subset)
                for w in clean_value_warnings:
                    warning_strs.append(f'Table: "{tbl_name}", column: "{w.column}", row: {w.row + 1}, '
                                        f'value: "{w.value}" - {w.message}')

            data_warning_str = ''
            if len(warning_strs) > 0:
                data_warning_str = 'WARNING: \n' + '\n'.join(warning_strs)

            # 3. Pre-process and upload file
            date_fields_dict = {
                'loan_df': ['facility_date', 'expiry_date'],
                'facility_df': ['available_period_from', 'available_period_to']
            }
            for df_name, fields in date_fields_dict.items():
                tbl_name = df_name.replace('_df', '')
                for field in fields:
                    try:
                        dfs[df_name][field] = dfs[df_name][field].dt.date
                    except Exception as e:
                        print(e)
                        try:
                            dfs[df_name][field] = dfs[df_name][field].apply(
                                lambda x: dt.datetime.strptime(x, '%Y-%m-%d').date())
                        except Exception as e:
                            print(e)
                            data_warning_str += f'\nTable: "{tbl_name}" - Unknown date format in column "{field}"'

            with pd.ExcelWriter(INPUT_DATA_DIR + BTS_DATA_FILENAME) as writer:
                for df_name, ws_name in zip(df_names, ws_names):
                    dfs[df_name].to_excel(writer, sheet_name=ws_name, index=False, freeze_panes=(1, 0))

            # Save a backup
            now = dt.datetime.now().strftime("%Y%m%d_%H%M")
            shutil.copyfile(
                INPUT_DATA_DIR + BTS_DATA_FILENAME,
                BACKUP_DATA_DIR + BTS_DATA_FILENAME.replace('.xlsx', '') + '_' + now + '.xlsx'
            )

            # 4. Final return
            dash_tables = {
                tbl_name: dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns])
                for (tbl_name, df) in [(df_name.replace('_df', ''), dfs[df_name]) for df_name in df_names]
            }

            return html.Div([
                html.Div('File(s) uploaded: ' + ', '.join(uploaded_filenames)),
                html.Pre(data_warning_str),
                html.Div([
                    html.Label('Preview - "facility"'),
                    dash_tables['facility'],
                ]),
                html.Br(),
                html.Div([
                    html.Label('Preview - "loan"'),
                    dash_tables['loan'],
                ]),
                html.Br(),
                html.Div([
                    html.Label('Preview - "lender"'),
                    dash_tables['lender'],
                ]),
                html.Br(),
                html.Div([
                    html.Label('Preview - "company"'),
                    dash_tables['company'],
                ]),
                html.Br(),
                html.Div([
                    html.Label('Preview - "company_grp"'),
                    dash_tables['company_grp'],
                ]),
            ])

    return server

