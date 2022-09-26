"""
Dashboard for matching 60% land cost with corporate loan (term & revolver)
With Uncommitted Revolver replacement
Same as dashboard_01.py
"""
from .utils import *
import yaml
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx


DASHBOARD_ID = 'dashboard_01b'
URL_BASE = '/dashboard_01b/'
DASH_CONFIG_FILEPATH = './app/dash/dash_config_01.yaml'


'''
--------------------
(0) Read Config Data and Define Global Variables
--------------------
'''
with open(DASH_CONFIG_FILEPATH, 'r') as cf:
    DASH_CONFIG = yaml.safe_load(cf)

INPUT_DATA_DIR = DASH_CONFIG['data']['input']['dir']
OUTPUT_DATA_DIR = DASH_CONFIG['data']['output']['dir']

EXPORT_INTERIM_OUTPUT_CSV = DASH_CONFIG['data']['output']['to_export']['interim_output_csv']
EXPORT_MASTER_OUTPUT_CSV = DASH_CONFIG['data']['output']['to_export']['master_output_csv']

# Dates
TODAY = dt.date.today()
if DASH_CONFIG['dates']['day_zero']['is_today']:
    DAY_ZERO = TODAY  # Day 0 of matching
else:
    DAY_ZERO = DASH_CONFIG['dates']['day_zero']['date']
MIN_DATE = TODAY
MAX_DATE = TODAY

DATE_TICK_TEXTS = []
DATE_TICK_VALUES = []

# Containing the rawest form of data
FACILITIES_DF = pd.DataFrame()
PROJECTS_DF = pd.DataFrame()

# Working data objects, which can be initialized, and mutated during matching
WKING_FACILITIES_DF = pd.DataFrame()
WKING_PROJECTS_DF = pd.DataFrame()
WKING_FACILITIES_DICT = dict()
WKING_PROJECTS_DICT = dict()
MATCHED_ENTRIES = list()

# Output data
STAGED_OUTPUTS = dict()  # Output per stage in dict format
STAGED_OUTPUTS_METADATA = dict()  # Description of the matching scheme
MASTER_OUTPUT = pd.DataFrame()  # Master output in dataframe format, can be exported to CSV/ Excel
MASTER_OUTPUT_VIS = pd.DataFrame()  # Master output for visualization

# Columns selected for output dataframes
SELECTED_FAC_COLUMNS = ['loan_facility_id', 'loan_id', 'reference_id', 'facility_name',
                        'lender_short_code', 'lender_short_form',
                        'loan_type', 'committed', 'loan_sub_type', 'tranche',
                        'available_period_from', 'available_period_to',
                        'facility_date', 'expiry_date', 'target_prepayment_date',
                        'facility_amount_inB', 'facility_amount',
                        'margin',
                        'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
                        'comm_fee_margin_below', 'all_in_price', 'net_margin',
                        'upfront_fee']
SELECTED_PROJ_COLUMNS = ['project_id', 'project_name', 'solo_jv',
                         'start_date', 'end_date',
                         'land_cost_60_pct_inB']

# Compatibility of matching scheme and visualization
COMPATIBILITY_S2V = {
    'scheme_1': ['get_gantt_1', 'get_gantt_2', 'get_gantt_3'],
    'scheme_2': ['get_gantt_1', 'get_gantt_2', 'get_gantt_3']
}
COMPATIBILITY_V2S = dict()
for k, vs in COMPATIBILITY_S2V.items():
    for v in vs:
        COMPATIBILITY_V2S[v] = COMPATIBILITY_V2S.get(v, []) + [k]

# Other global variables
DEFAULT_TPP_DATE_DELTA_YMD = (-1, 0, 0)
UC_FULL_COVER = DASH_CONFIG['matching_scheme']['full_cover']
UC_CHECK_SAVING_BY_AREA = DASH_CONFIG['matching_scheme']['check_saving_by_area']


'''
--------------------
(I) Load and Pre-process Data
--------------------
'''


def load_data():
    """Load data to FACILITIES_DF and PROJECTS_DF."""
    global INPUT_DATA_DIR, OUTPUT_DATA_DIR
    global FACILITIES_DF, PROJECTS_DF
    global TODAY, DAY_ZERO, MIN_DATE, MAX_DATE, DATE_TICK_TEXTS, DATE_TICK_VALUES

    '''(1) Load Loan Data'''
    loan_input_data_template = DASH_CONFIG['data']['input']['loan_data_template']['selected']

    if loan_input_data_template == 'template_l1':
        # Loan data from BTS
        loan_input_data_metadata = DASH_CONFIG['data']['input']['loan_data_template'][loan_input_data_template]

        bts_master_tbl_excel = loan_input_data_metadata['excels'][1]
        raw_df_company_grp = pd.read_excel(INPUT_DATA_DIR+bts_master_tbl_excel['name'],
                                           sheet_name=bts_master_tbl_excel['worksheets'][0])
        raw_df_company = pd.read_excel(INPUT_DATA_DIR+bts_master_tbl_excel['name'],
                                       sheet_name=bts_master_tbl_excel['worksheets'][1])
        raw_df_lender = pd.read_excel(INPUT_DATA_DIR+bts_master_tbl_excel['name'],
                                      sheet_name=bts_master_tbl_excel['worksheets'][2])

        bts_loan_profile_excel = loan_input_data_metadata['excels'][2]
        raw_df_loan = pd.read_excel(INPUT_DATA_DIR+bts_loan_profile_excel['name'],
                                    sheet_name=bts_loan_profile_excel['worksheets'][0])
        raw_df_facility = pd.read_excel(INPUT_DATA_DIR+bts_loan_profile_excel['name'],
                                        sheet_name=bts_loan_profile_excel['worksheets'][1])

        # Removing unwanted rows
        raw_df_company = raw_df_company[~raw_df_company['sys_company_id'].isna()]

        # Prepare single table for loan
        FACILITIES_DF = raw_df_facility.merge(raw_df_loan, left_on='loan_id', right_on='sys_loan_id',
                                              how='left', suffixes=('_fac', '_loan'))\
                                       .merge(raw_df_lender, left_on='lender_id', right_on='sys_lender_id',
                                              how='left', suffixes=('', '_lender'))\
                                       .merge(raw_df_company, left_on='borrower_id', right_on='sys_company_id',
                                              how='left', suffixes=('', '_com'))
        FACILITIES_DF = FACILITIES_DF[['sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche',
                                       'currency_id', 'facility_amount', 'available_period_from', 'available_period_to',
                                       'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
                                       'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
                                       'facility_date', 'expiry_date', 'withdrawn',
                                       'short_code', 'name', 'short_form',
                                       'company_name', 'company_short_name', 'type',
                                       'margin',
                                       'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
                                       'comm_fee_margin_below', 'all_in_price', 'net_margin',
                                       'upfront_fee']].copy()
        FACILITIES_DF.rename(columns={'sys_loan_facility_id': 'loan_facility_id',
                                      'short_code': 'lender_short_code',
                                      'short_form': 'lender_short_form',
                                      'name': 'lender_name',
                                      'type': 'company_type'}, inplace=True)
        # FACILITIES_DF.replace('NULL', None, inplace=True)
    elif loan_input_data_template == 'template_l2':
        # Loan data from BTS
        loan_input_data_metadata = DASH_CONFIG['data']['input']['loan_data_template'][loan_input_data_template]

        bts_tbl_excel = loan_input_data_metadata['excels'][1]
        raw_df_company_grp = pd.read_excel(INPUT_DATA_DIR + bts_tbl_excel['name'],
                                           sheet_name=bts_tbl_excel['worksheets'][0])
        raw_df_company = pd.read_excel(INPUT_DATA_DIR + bts_tbl_excel['name'],
                                       sheet_name=bts_tbl_excel['worksheets'][1])
        raw_df_lender = pd.read_excel(INPUT_DATA_DIR + bts_tbl_excel['name'],
                                      sheet_name=bts_tbl_excel['worksheets'][2])
        raw_df_loan = pd.read_excel(INPUT_DATA_DIR + bts_tbl_excel['name'],
                                    sheet_name=bts_tbl_excel['worksheets'][3])
        raw_df_facility = pd.read_excel(INPUT_DATA_DIR + bts_tbl_excel['name'],
                                        sheet_name=bts_tbl_excel['worksheets'][4])

        # Removing unwanted rows
        raw_df_company = raw_df_company[~raw_df_company['sys_company_id'].isna()]

        # Prepare single table for loan
        FACILITIES_DF = raw_df_facility.merge(raw_df_loan, left_on='loan_id', right_on='sys_loan_id',
                                              how='left', suffixes=('_fac', '_loan')) \
                                       .merge(raw_df_lender, left_on='lender_id', right_on='sys_lender_id',
                                              how='left', suffixes=('', '_lender')) \
                                       .merge(raw_df_company, left_on='borrower_id', right_on='sys_company_id',
                                              how='left', suffixes=('', '_com'))
        FACILITIES_DF = FACILITIES_DF[['sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche',
                                       'currency_id', 'facility_amount', 'available_period_from', 'available_period_to',
                                       'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
                                       'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
                                       'facility_date', 'expiry_date', 'withdrawn',
                                       'short_code', 'name', 'short_form',
                                       'company_name', 'company_short_name', 'type',
                                       'margin',
                                       'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
                                       'comm_fee_margin_below', 'all_in_price', 'net_margin',
                                       'upfront_fee']].copy()
        FACILITIES_DF.rename(columns={'sys_loan_facility_id': 'loan_facility_id',
                                      'short_code': 'lender_short_code',
                                      'short_form': 'lender_short_form',
                                      'name': 'lender_name',
                                      'type': 'company_type'}, inplace=True)

    '''(2) Load Project Data'''
    # Project data
    # Project ID and Project Name
    project_input_data_template = DASH_CONFIG['data']['input']['project_data_template']['selected']

    if project_input_data_template == 'template_p1':
        project_input_data_metadata = DASH_CONFIG['data']['input']['project_data_template'][project_input_data_template]
        project_tbl_excel = project_input_data_metadata['excels'][1]
        PROJECTS_DF = pd.read_excel(INPUT_DATA_DIR+project_tbl_excel['name'], sheet_name=project_tbl_excel['worksheets'][0])

        # Remove rows with NAs
        # PROJECTS_DF = PROJECTS_DF.dropna()

    '''(3) Update Global Variables'''
    # MIN_DATE not used for calculation, but DAY_ZERO
    MIN_DATE = min(*FACILITIES_DF['available_period_from'], *PROJECTS_DF['start_date']).to_pydatetime().date()
    MAX_DATE = max(*FACILITIES_DF['expiry_date'], *PROJECTS_DF['end_date']).to_pydatetime().date()
    DATE_TICK_TEXTS, DATE_TICK_VALUES = get_date_tick_lists(DAY_ZERO, MAX_DATE, short_form=True)

    return


def preprocess_data(tpp_date_delta_ymd=(-1, 0, 0), equity_amt_in_b=[40.2], uc_evergreen=True):
    """Pre-process data to update FACILITIES_DF and PROJECTS_DF.
    Args:
        - tpp_date_delta_ymd: The time delta in year, month, day between loan expiry date and target prepayment date
        - equity_amount_in_b: List of equity amount in HK$B
        - uc_evergreen: bool, whether the Uncommitted Revolver (UC-RTN) is assumed to have no maturity date
    """
    global FACILITIES_DF, PROJECTS_DF
    global TODAY, DAY_ZERO, MIN_DATE, MAX_DATE

    '''(1) Pre-process Loan Data'''
    # Fill NA
    str_fields = ['lender_short_code', 'loan_type', 'project_loan_share', 'committed', 'loan_sub_type', 'tranche']
    FACILITIES_DF[str_fields] = FACILITIES_DF[str_fields].fillna('')

    # Concatenate info as identifier, e.g. '5Y$1.0b-CCB-CommittedTerm(123)'
    FACILITIES_DF['facility_name'] = \
        (FACILITIES_DF['expiry_date'] - FACILITIES_DF['facility_date']).apply(
            lambda x: str(round(x.days/365*2)/2).replace('.0', '')) + 'Y' + \
        '$' + round(FACILITIES_DF['facility_amount']/1E9, 1).astype(str) + 'b' + '-' + \
        FACILITIES_DF['lender_short_code'] + '-' + \
        FACILITIES_DF['committed'].replace({'C': 'Committed', 'U': 'Uncommitted'}) + \
        FACILITIES_DF['loan_sub_type'].replace({'T': 'Term', 'R': 'RTN'}) + \
        '(' + FACILITIES_DF['loan_facility_id'].astype(str) + ')'

    # Target prepayment date, default delta against expiry_date is year=-1, month=0, day=0
    if uc_evergreen:
        tpd_lst = []
        for row in FACILITIES_DF[['committed', 'loan_sub_type', 'expiry_date']].to_dict('records'):
            if row['committed'] == 'U' and row['loan_sub_type'] == 'R':
                tpd_lst.append(MAX_DATE)
            elif row['loan_sub_type'] in ['T', 'R']:
                tpd_lst.append(offset_date(row['expiry_date'], *tpp_date_delta_ymd))
            else:
                tpd_lst.append(row['expiry_date'])
        FACILITIES_DF['target_prepayment_date'] = tpd_lst
    else:
        FACILITIES_DF['target_prepayment_date'] = \
            FACILITIES_DF[['loan_sub_type', 'expiry_date']].apply(
            lambda x: offset_date(x[1], *tpp_date_delta_ymd) if x[0] in ['T', 'R'] else x[1], axis=1)

    # Facility amount in HK$B
    FACILITIES_DF['facility_amount_inB'] = FACILITIES_DF['facility_amount'] / 1E9

    # Dummy data: Equity, add equity entries one by one
    for i, amt_in_b in enumerate(equity_amt_in_b):
        if len(equity_amt_in_b) == 1:
            facility_name = 'W-Equity'
        else:
            facility_name = 'W-Equity-' + str(i+1)
        equity = {
            'loan_facility_id': 99990 + i + 1,
            'loan_id': 99990 + i + 1,
            'loan_sub_type': 'Equity',
            'currency_id': 1,
            'facility_amount': amt_in_b * 1E9,
            'available_period_from': DAY_ZERO,
            'available_period_to': MAX_DATE,
            'loan_type': 'Equity',
            'facility_date': DAY_ZERO,
            'expiry_date': MAX_DATE,
            'withdrawn': 'N',
            'lender_short_code': 'W',
            'lender_name': 'Wheelock',
            'lender_short_form': 'W',
            'facility_name': facility_name,
            'target_prepayment_date': MAX_DATE,
            'facility_amount_inB': amt_in_b
        }

        FACILITIES_DF = pd.concat([FACILITIES_DF, pd.DataFrame({k: [v] for k, v in equity.items()})],
                                  ignore_index=True, sort=False)

    # Change data type of date
    fac_date_fields = ['available_period_from', 'available_period_to',
                       'facility_date', 'expiry_date', 'target_prepayment_date']
    for field in fac_date_fields:
        FACILITIES_DF[field] = FACILITIES_DF[field].apply(std_date)
        FACILITIES_DF[field+'_idx'] = FACILITIES_DF[field].apply(lambda x: date2idx(x, DAY_ZERO))

    '''(2) Pre-process Project Data'''
    # Change data type of date
    proj_date_fields = ['start_date', 'end_date']
    for field in proj_date_fields:
        PROJECTS_DF[field] = PROJECTS_DF[field].apply(std_date)
        PROJECTS_DF[field+'_idx'] = PROJECTS_DF[field].apply(lambda x: date2idx(x, DAY_ZERO))

    return


'''
--------------------
(II) Matching
--------------------
'''


def init_working_data():
    """Initialize WKING_FACILITIES_DF, WKING_PROJECTS_DF,
    WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES, and
    STAGED_OUTPUTS before matching."""
    global FACILITIES_DF, PROJECTS_DF
    global TODAY, DAY_ZERO, MIN_DATE, MAX_DATE
    global WKING_FACILITIES_DF, WKING_PROJECTS_DF
    global WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES
    global STAGED_OUTPUTS, STAGED_OUTPUTS_METADATA, MASTER_OUTPUT, MASTER_OUTPUT_VIS

    # Filter unwanted records
    # 1. Project Loan: facilties.loan_type = 'P',
    # 2. Loan is cancelled: facilities.withdrawn == 'Y',
    # 3. Past loan/ project: loan.target_prepayment_date < DAY_ZERO project.end_date < DAY_ZERO
    WKING_FACILITIES_DF = FACILITIES_DF[
        (FACILITIES_DF['loan_type'] != 'P') &
        (FACILITIES_DF['withdrawn'] != 'Y') &
        (FACILITIES_DF['target_prepayment_date'] >= DAY_ZERO)
    ]
    WKING_PROJECTS_DF = PROJECTS_DF[PROJECTS_DF['end_date'] >= DAY_ZERO]

    # Find boundaries
    max_date_idx = max(*WKING_FACILITIES_DF['target_prepayment_date_idx'],
                       *WKING_PROJECTS_DF['end_date_idx'])

    # Initialize working data {loan_facility_id: np.array([vector])}
    wking_facilities_list = WKING_FACILITIES_DF.to_dict('records')
    for record in wking_facilities_list:
        start = max(0, record['available_period_from_idx'])
        end = record['target_prepayment_date_idx']
        vec = np.zeros(max_date_idx + 1)
        vec[start:end + 1] = record['facility_amount_inB']
        record['vector'] = vec
    # Convert to dict with loan_facility_id as index
    WKING_FACILITIES_DICT = {fac['loan_facility_id']: fac for fac in wking_facilities_list}

    # Initialize working data {project_id: np.array([vector])}
    wking_projects_list = WKING_PROJECTS_DF.to_dict('records')
    for record in wking_projects_list:
        start = max(0, record['start_date_idx'])
        end = record['end_date_idx']
        vec = np.zeros(max_date_idx + 1)
        vec[start:end + 1] = record['land_cost_60_pct_inB']
        record['vector'] = vec
    # Convert to dict with project_id as index
    WKING_PROJECTS_DICT = {proj['project_id']: proj for proj in wking_projects_list}

    # Initialize working data [{'loan_facility_id': int, 'project_id': int, 'vectors': [vector]}]
    MATCHED_ENTRIES = list()

    # Initialize staged output
    STAGED_OUTPUTS = dict()
    STAGED_OUTPUTS_METADATA = dict()
    MASTER_OUTPUT = pd.DataFrame()
    MASTER_OUTPUT_VIS = pd.DataFrame()

    # 3 key working objects for matching: WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES
    #  and their updated states per each stage of matching will be stored in STAGED_OUTPUTS
    return


def matching_by_area(fac_idxs: list, proj_idxs: list, match_type: str = 'normal', **kwargs):
    """A procedure to perform matching given
    Args:
        - fac_idxs: list of facility ids
        - proj_idxs: list of project ids
        - match_type: a free text remark of the type of matched entry, e.g. 'normal', 'reserved', 'replacement'
    return: void
    """
    global WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES

    if len(fac_idxs) == 0 or len(proj_idxs) == 0:
        return
    while True:
        # Initialization
        best_overlapping = np.array([])
        max_overlapping_area = 0
        # best_match_entry = {'loan_facility_id': int, 'project_id': int, 'vector': np.array, 'match_type': str}
        best_match_entry = dict()
        best_match_fac_idx = -1
        best_match_proj_idx = -1
        for fac_idx, proj_idx in itertools.product(fac_idxs, proj_idxs):
            overlapping = np.minimum(WKING_FACILITIES_DICT[fac_idx]['vector'],
                                     WKING_PROJECTS_DICT[proj_idx]['vector'])
            overlapping_area = sum(overlapping)
            if overlapping_area > max_overlapping_area: # Better match found
                best_overlapping = overlapping
                max_overlapping_area = overlapping_area
                best_match_entry = {'loan_facility_id': fac_idx,
                                    'project_id': proj_idx,
                                    'vector': overlapping,
                                    'match_type': match_type}
                best_match_fac_idx = fac_idx
                best_match_proj_idx = proj_idx
            else:
                pass
        if max_overlapping_area == 0:
            break
        else: # max_overlapping_area > 0 -> There is a match
            MATCHED_ENTRIES.append(best_match_entry)
            # Update values in master
            WKING_FACILITIES_DICT[best_match_fac_idx]['vector'] -= best_overlapping
            WKING_PROJECTS_DICT[best_match_proj_idx]['vector'] -= best_overlapping
    return


def replace_matched_entries(full_cover: bool = True, check_saving_by_area: bool = True):
    """A procedure for replacing matched matched Committed Revolver (C-RTN) entries with Uncommitted Revolver (UC-RTN)
    Args:
    - full_cover: bool,
        if true, replacement could be made only when all elem in UC-RTN vector >= that in C-RTN matched entry's vector
        if false, partial replacement is allowed
    - check_saving_by_area: bool,
        if true, saving is calculated by net_margin difference x overlapping area,
        if false, consider saving by largest net_margin difference, then by overlapping area
    return: void
    """
    global WKING_FACILITIES_DICT, MATCHED_ENTRIES

    # loan_facility_id of Uncommitted Revolver (UC-RTN)
    ucrtn_idxs = [fac['loan_facility_id'] for fac in list(WKING_FACILITIES_DICT.values())
                  if (fac['loan_sub_type'] == 'R') and (fac['committed'] == 'U')]
    # The position nos. in MATCHED_ENTRIES for matched Committed Revolver (C-RTN)
    crtn_matched_entry_poss = [i for i, me in enumerate(MATCHED_ENTRIES)
                               if ((WKING_FACILITIES_DICT[me['loan_facility_id']]['loan_sub_type'] == 'R') and
                                   (WKING_FACILITIES_DICT[me['loan_facility_id']]['committed'] == 'C'))]
    # Mapping of replacement UC-RTN entry's (fac_id, proj_id) -> Position no. in MATCHED_ENTRIES
    #   for locating the right pos for vector updating
    ucrtn_matched_entry_poss_dict = dict()
    # Mapping of reserved (replaced) C-RTN entry's (fac_id, proj_id) -> Position no. in MATCHED_ENTRIES
    #   for locating the right pos for vector updating
    crtn_reserved_entry_poss_dict = dict()

    if (len(ucrtn_idxs) > 0) and (len(crtn_matched_entry_poss) > 0):
        # Check each UC-RTN to see if it can replace certain matched C-RTNs
        for ucrtn_idx in ucrtn_idxs:
            while True:
                # Initialization of best replacement info #
                ucrtn_info = WKING_FACILITIES_DICT[ucrtn_idx]  # Get the latest UC-RTN info
                best_replacement_vector = np.array([])
                max_cost_saving = 0
                best_overlapping_area = 0  # The overlapping area of best match, not necessarily the max
                crtn_matched_entry_to_reserve_pos = -1  # Position no. in MATCHED_ENTRIES, NOT loan_facility_id
                crtn_matched_entry_to_reserve_fac_idx = -1  # loan_facility_id of C-RTN matched entry
                crtn_matched_entry_to_reserve_proj_idx = -1  # project_id of C-RTN matched entry
                # End of initialization #

                # Loop over the matched C-RTNs to check if any replacement could be made #
                for crtn_matched_entry_pos in crtn_matched_entry_poss:
                    crtn_matched_entry = MATCHED_ENTRIES[crtn_matched_entry_pos]
                    wking_crtn_fac_idx = crtn_matched_entry['loan_facility_id']
                    wking_proj_idx = crtn_matched_entry['project_id']
                    crtn_info = WKING_FACILITIES_DICT[wking_crtn_fac_idx]
                    # Checking #
                    # Criteria:
                    # i) All elements in UC-RTN vector >= that in C-RTN matched entry's vector
                    # ii) net_margin of UC-RTN < that in C-RTN matched entry
                    # iii) Largest cost saving between UC-RTN and C-RTN matched entry, where
                    #  cost saving = net_margin difference x overlapping area if check_saving_by_area is True
                    #  cost saving = net_margin difference if check_saving_by_area is False
                    margin_diff = crtn_info['net_margin'] - ucrtn_info['net_margin']  # The gain
                    overlapping_vector = np.minimum(crtn_matched_entry['vector'], ucrtn_info['vector'])
                    overlapping_area = sum(overlapping_vector)
                    # Computation of cost saving
                    if check_saving_by_area:
                        cost_saving = overlapping_area * margin_diff
                    else:
                        cost_saving = margin_diff
                    # Checking of (i)
                    if full_cover:
                        # UC-RTN fully cover matched C-RTN entry
                        pass_criteria_1 = all(ucrtn_info['vector'] >= crtn_matched_entry['vector'])
                    else:
                        # At least some overlapping
                        pass_criteria_1 = (overlapping_area > 0)
                    # Checking of (iii) implies (ii), b/c if margin_diff is non-positive,
                    #  cost_saving must be <= 0 <= max_cost_saving
                    if check_saving_by_area:
                        pass_criteria_23 = cost_saving > max_cost_saving
                    else:
                        pass_criteria_23 = (overlapping_area > 0) and \
                                           ((cost_saving > max_cost_saving) or
                                            ((cost_saving == max_cost_saving != 0) and
                                             (overlapping_area > best_overlapping_area)))
                    # End of checking #
                    # Updating best replacement info if criteria are met #
                    if pass_criteria_1 and pass_criteria_23:
                        best_replacement_vector = overlapping_vector.copy()
                        max_cost_saving = cost_saving
                        crtn_matched_entry_to_reserve_pos = crtn_matched_entry_pos
                        crtn_matched_entry_to_reserve_fac_idx = wking_crtn_fac_idx
                        crtn_matched_entry_to_reserve_proj_idx = wking_proj_idx
                    else:
                        pass
                    # End of updating best replacement info #

                # End of one loop of the matched C-RTNs #

                # Check if there is any replacement, if yes, update the master data #
                if crtn_matched_entry_to_reserve_pos == -1:  # No replacement could be made
                    break
                else:  # A replacement could be made, this is what we are looking for
                    # Append (if new)/ Update (if exist) 'replacement' matched entry for UC-RTN
                    if (ucrtn_idx, crtn_matched_entry_to_reserve_proj_idx) not in ucrtn_matched_entry_poss_dict:
                        # New matched entry, append
                        ucrtn_matched_entry_poss_dict[(ucrtn_idx, crtn_matched_entry_to_reserve_proj_idx)] = \
                            len(MATCHED_ENTRIES)
                        MATCHED_ENTRIES.append({'loan_facility_id': ucrtn_idx,
                                                'project_id': crtn_matched_entry_to_reserve_proj_idx,
                                                'vector': best_replacement_vector.copy(),
                                                'match_type': 'replacement'})
                    else:  # Matched entry already exists, update by adding vector values
                        pos = ucrtn_matched_entry_poss_dict[(ucrtn_idx, crtn_matched_entry_to_reserve_proj_idx)]
                        MATCHED_ENTRIES[pos]['vector'] += best_replacement_vector
                    # Update UC-RTN vector values in master
                    WKING_FACILITIES_DICT[ucrtn_idx]['vector'] -= best_replacement_vector

                    # Append (if new)/ Update (if exist) 'replaced' matched C-RTN entry
                    if (crtn_matched_entry_to_reserve_fac_idx, crtn_matched_entry_to_reserve_proj_idx) \
                            not in crtn_reserved_entry_poss_dict:
                        # New reserved entry, append
                        crtn_reserved_entry_poss_dict[(crtn_matched_entry_to_reserve_fac_idx,
                                                       crtn_matched_entry_to_reserve_proj_idx)] = len(MATCHED_ENTRIES)
                        MATCHED_ENTRIES.append({'loan_facility_id': crtn_matched_entry_to_reserve_fac_idx,
                                                'project_id': crtn_matched_entry_to_reserve_proj_idx,
                                                'vector': best_replacement_vector.copy(),
                                                'match_type': 'reserved'})
                    else:  # Replaced entry already exists, update by adding vector values
                        pos = crtn_reserved_entry_poss_dict[(crtn_matched_entry_to_reserve_fac_idx,
                                                             crtn_matched_entry_to_reserve_proj_idx)]
                        MATCHED_ENTRIES[pos]['vector'] += best_replacement_vector
                    # Update matched C-RTN entry in master
                    MATCHED_ENTRIES[crtn_matched_entry_to_reserve_pos]['vector'] -= best_replacement_vector

                # End of Check if there is any replacement, if yes, update the master data #

        # Remove matched entries if sum of all vector elements == 0
        MATCHED_ENTRIES = [matched_entry for matched_entry in MATCHED_ENTRIES if sum(matched_entry['vector']) != 0]

    else:  # No UC-RTN or matched C-RTN
        pass

    return


def matching_main_proc(scheme: int = 1):
    """Main matching procedures to produce staged outputs.
    Args:
        - scheme: int, the matching scheme id
    """
    global WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES
    global STAGED_OUTPUTS, STAGED_OUTPUTS_METADATA, MASTER_OUTPUT
    global SELECTED_FAC_COLUMNS, SELECTED_PROJ_COLUMNS
    global UC_FULL_COVER, UC_CHECK_SAVING_BY_AREA

    '''(1) Matching per stage'''
    # ===== Matching Scheme 1 ===== #
    if scheme == 1:
        # === Scheme 1: matching procedure === #
        # Stage 0: Initial
        # Stage 1: Term Facilities vs. Solo then JV
        # Stage 2: Revolver vs. Solo then JV
        # Stage 3: Equity vs. Solo then JV

        # Output to STAGED_OUTPUTS
        for ln, loan_sub_type in enumerate(['<Initial>', 'T', 'R', 'Equity']):
            for sjn, solo_jv in enumerate(['Solo', 'JV']):
                # Sub-set the matching batch
                fac_idxs = [fac['loan_facility_id'] for fac in list(WKING_FACILITIES_DICT.values())
                            if fac['loan_sub_type'] == loan_sub_type]
                proj_idxs = [proj['project_id'] for proj in list(WKING_PROJECTS_DICT.values())
                             if proj['solo_jv'] == solo_jv]
                # Matching by area
                matching_by_area(fac_idxs, proj_idxs)
            # Total shortfall
            ttl_shortfall_vec = np.sum(np.stack([v['vector'] for v in WKING_PROJECTS_DICT.values()]), axis=0)
            # Output per stages: Initial -> Term -> Revolver -> Equity 5+6
            STAGED_OUTPUTS['stage ' + str(ln)] = {'fac': copy.deepcopy(WKING_FACILITIES_DICT),
                                                  'proj': copy.deepcopy(WKING_PROJECTS_DICT),
                                                  'matched': copy.deepcopy(MATCHED_ENTRIES),
                                                  'ttl_shortfall_vec': ttl_shortfall_vec}

        # === Scheme 1: matching scheme metadata === #
        STAGED_OUTPUTS_METADATA = {
            '_id': 1,
            'name': 'Simple T-R-E',
            'description': 'Stage 0: Initial; '
                           'Stage 1: Term Facilities vs. Solo then JV; '
                           'Stage 2: Revolver vs. Solo then JV; '
                           'Stage 3: Equity vs. Solo then JV',
            'short_description': 'Term -> Revolver -> Equity, Solo -> JV per stage',
            'stages': {
                0: {'value': 'stage 0', 'label': 'Stage 0: Initial'},
                1: {'value': 'stage 1', 'label': 'Stage 1: Term'},
                2: {'value': 'stage 2', 'label': 'Stage 2: Term+RTN'},
                3: {'value': 'stage 3', 'label': 'Stage 3: Term+RTN+Equity'}
            }
        }
    # ===== End of Matching Scheme #1 ===== #

    # ===== Matching Scheme 2 ===== #
    elif scheme == 2:
        # === Scheme 2: matching procedure === #
        # Stage 0: Initial
        # Stage 1: Term Facilities vs. Solo then JV
        # Stage 2: Committed Revolver vs. Solo then JV
        # Stage 2a: Uncommitted Revolver to replace Committed Revolver
        # Stage 3: Equity vs. Solo then JV

        # Output to STAGED_OUTPUTS
        # == Stage 0, 1 & 2 == #
        for ln, loan_sub_type in enumerate(['<Initial>', 'T', 'R']):
            for sjn, solo_jv in enumerate(['Solo', 'JV']):
                # Sub-set the matching batch
                proj_idxs = [proj['project_id'] for proj in list(WKING_PROJECTS_DICT.values())
                             if proj['solo_jv'] == solo_jv]
                if loan_sub_type == 'R':  # For Stage 2, only select Committed Revolver
                    fac_idxs = [fac['loan_facility_id'] for fac in list(WKING_FACILITIES_DICT.values())
                                if (fac['loan_sub_type'] == loan_sub_type) and (fac['committed'] == 'C')]
                else:  # For Stage 0 and 1
                    fac_idxs = [fac['loan_facility_id'] for fac in list(WKING_FACILITIES_DICT.values())
                                if fac['loan_sub_type'] == loan_sub_type]
                # Matching by area
                matching_by_area(fac_idxs, proj_idxs)
            # Total shortfall
            ttl_shortfall_vec = np.sum(np.stack([v['vector'] for v in WKING_PROJECTS_DICT.values()]), axis=0)
            # Output per stages: Initial -> Term -> Revolver -> Equity 5+6
            STAGED_OUTPUTS['stage ' + str(ln)] = {'fac': copy.deepcopy(WKING_FACILITIES_DICT),
                                                  'proj': copy.deepcopy(WKING_PROJECTS_DICT),
                                                  'matched': copy.deepcopy(MATCHED_ENTRIES),
                                                  'ttl_shortfall_vec': ttl_shortfall_vec}

        # == Stage 2a: Replace matched Committed Revolver (C-RTN) with Uncommitted Revolver (UC-RTN) == #
        # For STAGED_OUTPUTS, 'proj' remains the same as that in Stage 2,
        # update 'fac' for Committed Revolver and
        # 'matched': MATCHED_ENTRIES (list of dicts), with each dict's format as
        #  {'loan_facility_id': fac_idx, 'project_id': proj_idx, 'vector': overlapping, 'match_type': match_type}

        replace_matched_entries(full_cover=UC_FULL_COVER, check_saving_by_area=UC_CHECK_SAVING_BY_AREA)

        # Total shortfall
        ttl_shortfall_vec = np.sum(np.stack([v['vector'] for v in WKING_PROJECTS_DICT.values()]), axis=0)
        # Output for Stage 2a
        STAGED_OUTPUTS['stage 2a'] = {'fac': copy.deepcopy(WKING_FACILITIES_DICT),
                                      'proj': copy.deepcopy(WKING_PROJECTS_DICT),
                                      'matched': copy.deepcopy(MATCHED_ENTRIES),
                                      'ttl_shortfall_vec': ttl_shortfall_vec}

        # == Stage 3: Match equity == #
        for sjn, solo_jv in enumerate(['Solo', 'JV']):
            # Sub-set the matching batch
            proj_idxs = [proj['project_id'] for proj in list(WKING_PROJECTS_DICT.values())
                         if proj['solo_jv'] == solo_jv]
            fac_idxs = [fac['loan_facility_id'] for fac in list(WKING_FACILITIES_DICT.values())
                        if fac['loan_sub_type'] == 'Equity']
            # Matching by area
            matching_by_area(fac_idxs, proj_idxs)
        # Total shortfall
        ttl_shortfall_vec = np.sum(np.stack([v['vector'] for v in WKING_PROJECTS_DICT.values()]), axis=0)
        # Output for Stage 3
        STAGED_OUTPUTS['stage 3'] = {'fac': copy.deepcopy(WKING_FACILITIES_DICT),
                                     'proj': copy.deepcopy(WKING_PROJECTS_DICT),
                                     'matched': copy.deepcopy(MATCHED_ENTRIES),
                                     'ttl_shortfall_vec': ttl_shortfall_vec}

        # === Scheme 2: matching scheme metadata === #
        fc_ = 'UC full cover' if UC_FULL_COVER else 'UC partial cover'
        csa_ = 'saving = net margin diff x area' if UC_CHECK_SAVING_BY_AREA else 'saving = net margin diff only'
        stage2a_suffix = ' (' + fc_ + ', ' + csa_ + ')'
        STAGED_OUTPUTS_METADATA = {
            '_id': 2,
            'name': 'T-R-E plus UC-RTN replacement',
            'description': 'Stage 0: Initial; '
                           'Stage 1: Term Facilities vs. Solo then JV; '
                           'Stage 2: Committed Revolver vs. Solo then JV; '
                           # 'Stage 2a: Uncommitted Revolver to replace Committed Revolver' + stage2a_suffix + '; '
                           'Stage 2a: Uncommitted Revolver to replace Committed Revolver; '
                           'Stage 3: Equity vs. Solo then JV',
            'short_description': 'Term -> Committed Revolver -> UC replacement -> Equity, Solo -> JV per stage',
            'stages': {
                0: {'value': 'stage 0', 'label': 'Stage 0: Initial'},
                1: {'value': 'stage 1', 'label': 'Stage 1: Term'},
                2: {'value': 'stage 2', 'label': 'Stage 2: Term + Committed RTN'},
                3: {'value': 'stage 2a',
                    # 'label': 'Stage 2a: Term + Committed RTN + Uncommitted RTN Replacement' + stage2a_suffix
                    'label': 'Stage 2a: Term + Committed RTN + Uncommitted RTN Replacement'},
                4: {'value': 'stage 3', 'label': 'Stage 3: Term + Committed RTN + Uncommitted RTN Replacement + Equity'}
            }
        }
    # ===== End of Matching Scheme #2 ===== #


    '''(2) Tidy up result'''
    MASTER_OUTPUT = pd.DataFrame()
    # (2.1) Loop over stages and store to centralized dfs
    for stage, stage_state in STAGED_OUTPUTS.items():
        # (2.1.1) Matched entries
        # Convert to dataframe
        matched_entries_dict = {
            'loan_facility_id': [],
            'project_id': [],
            'match_type': [],
            'from_date_idx': [],
            'to_date_idx': [],
            'matched_amt_inB': [],
            'trace_value': []
        }
        for entry in stage_state['matched']:
            for f, t, a in vec2rects(entry['vector']):
                matched_entries_dict['loan_facility_id'].append(entry['loan_facility_id'])
                matched_entries_dict['project_id'].append(entry['project_id'])
                matched_entries_dict['match_type'].append(entry['match_type'])
                matched_entries_dict['from_date_idx'].append(f)
                matched_entries_dict['to_date_idx'].append(t)
                matched_entries_dict['matched_amt_inB'].append(a)
                matched_entries_dict['trace_value'].append(a)
        matched_entries_df = pd.DataFrame(matched_entries_dict)
        # Convert back to date columns
        matched_entries_df['from_date'] = matched_entries_df['from_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        matched_entries_df['to_date'] = matched_entries_df['to_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        # Add indicator columns
        matched_entries_df['stage'] = stage
        matched_entries_df['output_type'] = 'matched'
        matched_entries_df['output_type_num'] = 1
        # Augment dataframe
        matched_entries_df = matched_entries_df.sort_values(['project_id', 'loan_facility_id']).reset_index(drop=True)
        matched_entries_df = matched_entries_df.merge(PROJECTS_DF[SELECTED_PROJ_COLUMNS], on='project_id', how='left') \
            .merge(FACILITIES_DF[SELECTED_FAC_COLUMNS], on='loan_facility_id', how='left')
        matched_entries_df['projectXfacility'] = matched_entries_df['project_name'] + ' | ' + \
                                                 matched_entries_df['facility_name'] + \
                                                 ' (' + matched_entries_df['match_type'] + ')'
        matched_entries_df['projectXfacility'] = matched_entries_df['projectXfacility'].str.replace(' (normal)', '',
                                                                                                    regex=False)
        matched_entries_df['y_label'] = matched_entries_df['projectXfacility']

        # (2.1.2) Shortfall
        # Convert to dataframe
        shortfalls_dict = {
            'project_id': [],
            'from_date_idx': [],
            'to_date_idx': [],
            'shortfall_amt_inB': [],
            'trace_value': []
        }
        for k, v in stage_state['proj'].items():
            for f, t, a in vec2rects(v['vector'],
                                     preserve_zero=True,
                                     preserve_zero_st_idx=max(0, v['start_date_idx']),
                                     preserve_zero_end_idx=v['end_date_idx']):
                # Special for project: Show zero even the amount is filled
                shortfalls_dict['project_id'].append(k)
                shortfalls_dict['from_date_idx'].append(f)
                shortfalls_dict['to_date_idx'].append(t)
                shortfalls_dict['shortfall_amt_inB'].append(a)
                shortfalls_dict['trace_value'].append(a)
            '''
            if len(vec2rects(v['vector'])) == 0:  # Special for project: If all the project requirement is filled
                shortfalls_dict['project_id'].append(k)
                shortfalls_dict['from_date_idx'].append(
                    max(v['start_date_idx'], 0))  # Now the min date for display is 0
                shortfalls_dict['to_date_idx'].append(v['end_date_idx'])
                shortfalls_dict['shortfall_amt_inB'].append(0)
                shortfalls_dict['trace_value'].append(0)
            else:  # Special for project: there is shortfall for the project
                for f, t, a in vec2rects(v['vector']):
                    shortfalls_dict['project_id'].append(k)
                    shortfalls_dict['from_date_idx'].append(f)
                    shortfalls_dict['to_date_idx'].append(t)
                    shortfalls_dict['shortfall_amt_inB'].append(a)
                    shortfalls_dict['trace_value'].append(a)
            '''
        shortfalls_df = pd.DataFrame(shortfalls_dict)
        # Convert back to date columns
        shortfalls_df['from_date'] = shortfalls_df['from_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        shortfalls_df['to_date'] = shortfalls_df['to_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        # Add indicator columns
        shortfalls_df['stage'] = stage
        shortfalls_df['output_type'] = 'shortfall'
        shortfalls_df['output_type_num'] = 2
        # Augment dataframe
        shortfalls_df = shortfalls_df.sort_values(['project_id']).reset_index(drop=True)
        shortfalls_df = shortfalls_df.merge(PROJECTS_DF[SELECTED_PROJ_COLUMNS], on='project_id', how='left')
        shortfalls_df['y_label'] = shortfalls_df['project_name']

        # (2.1.3) Leftover
        # Convert to dataframe
        leftovers_dict = {
            'loan_facility_id': [],
            'from_date_idx': [],
            'to_date_idx': [],
            'leftover_amt_inB': [],
            'trace_value': []
        }
        for k, v in stage_state['fac'].items():
            for f, t, a in vec2rects(v['vector']):
                leftovers_dict['loan_facility_id'].append(k)
                leftovers_dict['from_date_idx'].append(f)
                leftovers_dict['to_date_idx'].append(t)
                leftovers_dict['leftover_amt_inB'].append(a)
                leftovers_dict['trace_value'].append(a)
        leftovers_df = pd.DataFrame(leftovers_dict)
        # Convert back to date columns
        leftovers_df['from_date'] = leftovers_df['from_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        leftovers_df['to_date'] = leftovers_df['to_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        # Add indicator columns
        leftovers_df['stage'] = stage
        leftovers_df['output_type'] = 'leftover'
        leftovers_df['output_type_num'] = 3
        # Augment dataframe
        leftovers_df = leftovers_df.sort_values(['loan_facility_id']).reset_index(drop=True)
        leftovers_df = leftovers_df.merge(FACILITIES_DF[SELECTED_FAC_COLUMNS], on='loan_facility_id', how='left')
        leftovers_df['y_label'] = leftovers_df['facility_name']

        # (2.1.4) Total shortfall
        # Convert to dataframe
        ttl_shortfalls_dict = {
            'from_date_idx': [],
            'to_date_idx': [],
            'shortfall_amt_inB': [],
            'trace_value': []
        }
        pzei_ = len(stage_state['ttl_shortfall_vec']) - 1
        for f, t, a in vec2rects(stage_state['ttl_shortfall_vec'],
                                 preserve_zero=True,
                                 preserve_zero_st_idx=0,
                                 preserve_zero_end_idx=pzei_):
            ttl_shortfalls_dict['from_date_idx'].append(f)
            ttl_shortfalls_dict['to_date_idx'].append(t)
            ttl_shortfalls_dict['shortfall_amt_inB'].append(a)
            ttl_shortfalls_dict['trace_value'].append(a)
        ttl_shortfall_df = pd.DataFrame(ttl_shortfalls_dict)
        # Convert back to date columns
        ttl_shortfall_df['from_date'] = ttl_shortfall_df['from_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        ttl_shortfall_df['to_date'] = ttl_shortfall_df['to_date_idx'].apply(lambda x: idx2date(x, DAY_ZERO))
        # Add indicator columns
        ttl_shortfall_df['stage'] = stage
        ttl_shortfall_df['output_type'] = 'ttl_shortfall'
        ttl_shortfall_df['output_type_num'] = 4
        # Augment dataframe
        ttl_shortfall_df['y_label'] = 'Total Shortfall'

        # (2.1.9) Concat to MASTER_OUTPUT
        MASTER_OUTPUT = pd.concat([MASTER_OUTPUT, matched_entries_df, shortfalls_df, leftovers_df, ttl_shortfall_df],
                                  ignore_index=True, sort=False)

    # (2.2) Sort MASTER_OUTPUT:
    #  stage (asc) -> output_type_num (asc) -> solo_jv (Solo -> JV) -> project_id (asc)
    #  -> loan_sub_type (T -> R -> Equity) -> loan_id (asc) -> loan_facility_id (asc)
    MASTER_OUTPUT['solo_jv_rank'] = MASTER_OUTPUT['solo_jv'].replace({'Solo': 1, 'JV': 2})
    # MASTER_OUTPUT['loan_sub_type_rank'] = MASTER_OUTPUT['loan_sub_type'].replace({'T': 1, 'R': 2, 'Equity': 3})
    MASTER_OUTPUT['loan_sub_type_rank'] = MASTER_OUTPUT['loan_sub_type'].astype(str).replace('nan', '') + \
                                          MASTER_OUTPUT['committed'].astype(str).replace('nan', '')
    MASTER_OUTPUT['loan_sub_type_rank'].replace({'T.*': 10, 'RC': 20, 'RU': 21, 'Equity': 30, '': 99},
                                                regex=True, inplace=True)
    MASTER_OUTPUT.sort_values(by=['stage', 'output_type_num', 'solo_jv_rank', 'project_id',
                                  'loan_sub_type_rank', 'loan_id', 'loan_facility_id'],
                              inplace=True)
    MASTER_OUTPUT.reset_index(drop=True, inplace=True)

    return


'''
--------------------
(III) Visualization
--------------------
'''


def get_gantt_1(stage_to_display, projects_to_display, *args, **kwargs):
    """Visualize data in gantt chart - version 1: 3 subplots for matched, shortfall, leftover, color = amount
    Args:
    - stage_to_display: str, 'stage 0' to 'stage 3'
    - projects_to_display: list of strings of project names
    """
    global MASTER_OUTPUT_VIS

    MASTER_OUTPUT_VIS = MASTER_OUTPUT.copy()
    MASTER_OUTPUT_VIS['trace_value_str'] = MASTER_OUTPUT_VIS['trace_value'].apply(lambda x: '%.3f' % x)

    gantts = ['gantt_matched', 'gantt_proj', 'gantt_fac']
    gantts_titles = ['Matched entries (HK$B)',
                     'Project needs (shortfall) (HK$B)',
                     'Loan facilities available (leftover) (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': MASTER_OUTPUT_VIS['stage'] == stage_to_display,
                             'project': MASTER_OUTPUT_VIS['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_matched'] = (MASTER_OUTPUT_VIS['output_type'] == 'matched') & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_proj'] = (MASTER_OUTPUT_VIS['output_type'] == 'shortfall') & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] != 'Equity')

    subplot_data = {g: MASTER_OUTPUT_VIS[conditions_to_display[g]] for g in gantts}

    # Dynamically define the heights of subplots given with the number of items
    # Calculate the number of items, at least 5
    subplot_num_items_to_display = {g: max(len(subplot_data[g]['y_label'].unique()), 5) for g in gantts}
    # Each bar 20px, Reserve 40px for subplot title
    subplot_heights = {g: subplot_num_items_to_display[g] * 20 + 40 for g in gantts}
    plot_height = sum(subplot_heights.values())
    subplot_height_shares = [subplot_heights[g] / plot_height for g in gantts]

    ## spacing between 2 subplots
    subplot_vertical_spacing = 0.05

    # marker_colorscale
    marker_colorscales = {k: v for k, v in zip(gantts, ['Greens', 'Reds', 'Purples'])}

    # Manually construct Bar graph_object
    fig = make_subplots(rows=len(gantts), cols=1,
                        shared_xaxes=True,
                        row_heights=subplot_height_shares,
                        vertical_spacing=subplot_vertical_spacing,
                        subplot_titles=gantts_titles)
    for g_idx, g in enumerate(gantts):
        plot_df = subplot_data[g].iloc[::-1]  # Rows are displayed from the bottom to the top
        fig.add_trace(go.Bar(x=plot_df[['from_date', 'to_date']].apply(lambda x: dt.date(1970, 1, 1) + (x[1] - x[0]), axis=1),
                             y=plot_df['y_label'],
                             orientation='h',
                             base=plot_df['from_date'],
                             marker_color=plot_df['trace_value'],
                             marker_colorscale=marker_colorscales[g],
                             text=plot_df['trace_value_str'],
                             customdata=plot_df['to_date'],
                             hovertemplate='%{y}<br>%{base: %Y-%B-%a %d} to %{customdata: %Y-%B-%a %d}<br>Amount in $B: %{text}'),
                      row=g_idx+1, col=1)

    fig.update_layout(height=plot_height)
    fig.update_yaxes(tickfont_size=9)
    fig.update_xaxes(type='date')
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    return fig


def get_gantt_2(stage_to_display, projects_to_display, *args, **kwargs):
    """Visualize data in gantt chart - version 2: 3 subplots for shortfall+matched, leftover, equity, color = amount
    Args:
    - stage_to_display: str, 'stage (0|1|2|...)'
    - projects_to_display: list of strings of project names
    """
    global MASTER_OUTPUT_VIS

    MASTER_OUTPUT_VIS = MASTER_OUTPUT.copy()
    MASTER_OUTPUT_VIS['trace_value_str'] = MASTER_OUTPUT_VIS['trace_value'].apply(lambda x: '%.3f' % x)

    gantts = ['gantt_main', 'gantt_fac', 'gantt_equity']
    gantts_titles = ['Project needs and matched entries (HK$B)',
                     'Loan facilities available (leftover) (HK$B)',
                     'Equity (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': MASTER_OUTPUT_VIS['stage'] == stage_to_display,
                             'project': MASTER_OUTPUT_VIS['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_main'] = (MASTER_OUTPUT_VIS['output_type'].isin(['shortfall', 'matched'])) & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] != 'Equity')
    conditions_to_display['gantt_equity'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] == 'Equity')

    subplot_data = {g: MASTER_OUTPUT_VIS[conditions_to_display[g]] for g in gantts}

    # Dynamically define the heights of subplots given with the number of items
    # Calculate the number of items, at least 5
    subplot_num_items_to_display = {g: max(len(subplot_data[g]['y_label'].unique()), 5) for g in gantts}
    # Each bar 20px, Reserve 40px for subplot title
    subplot_heights = {g: subplot_num_items_to_display[g] * 30 + 40 for g in gantts}
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
            'shortfall.*': 'indigo',
            'matched-T.*': 'yellow',
            'matched-R-normal.*': 'orange',
            'matched-R-replacement.*': 'deepskyblue',
            'matched-R-reserved.*': 'orange',
            'matched-Equity.*': 'hotpink',
            'leftover-T.*': 'yellow',
            'leftover-R.*-C': 'orange',
            'leftover-R.*-U': 'deepskyblue',
            'equity-Equity.*': 'hotpink'
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
                     ticktext=DATE_TICK_TEXTS,
                     tickvals=DATE_TICK_VALUES)
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    return fig


def get_gantt_3(stage_to_display, projects_to_display,
                bar_height=30, *args, **kwargs):
    """Visualize data in gantt chart - version 3: 4 subplots for shortfall+matched, leftover, total shortfall, equity,
     color = amount
    Args:
    - stage_to_display: str, 'stage (0|1|2|...)'
    - projects_to_display: list of strings of project names
    - bar_height: int, bar height
    """
    global MASTER_OUTPUT_VIS

    MASTER_OUTPUT_VIS = MASTER_OUTPUT.copy()
    MASTER_OUTPUT_VIS['trace_value_str'] = MASTER_OUTPUT_VIS['trace_value'].apply(lambda x: '%.3f' % x)

    gantts = ['gantt_main', 'gantt_fac', 'gantt_ttl_shortfall', 'gantt_equity']
    gantts_titles = ['Project needs and matched entries (HK$B)',
                     'Loan facilities available (leftover) (HK$B)',
                     'Total project needs (HK$B)',
                     'Equity (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': MASTER_OUTPUT_VIS['stage'] == stage_to_display,
                             'project': MASTER_OUTPUT_VIS['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_main'] = (MASTER_OUTPUT_VIS['output_type'].isin(['shortfall', 'matched'])) & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] != 'Equity')
    conditions_to_display['gantt_ttl_shortfall'] = (MASTER_OUTPUT_VIS['output_type'] == 'ttl_shortfall') & \
        conditions_to_display['stage']
    conditions_to_display['gantt_equity'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] == 'Equity')

    subplot_data = {g: MASTER_OUTPUT_VIS[conditions_to_display[g]] for g in gantts}

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
            'equity-Equity.*': 'hotpink'
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
                     ticktext=DATE_TICK_TEXTS,
                     tickvals=DATE_TICK_VALUES)
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    return fig


def get_gantt_diff_bar_width_3(stage_to_display, projects_to_display,
                               bar_height=30, trace_height_range=(0.25, 1.0), *args, **kwargs):
    """Visualize data in gantt chart - similar to get_gantt_3, but different bar widths according to amount
    Args:
    - stage_to_display: str, 'stage (0|1|2|...)'
    - projects_to_display: list of strings of project names
    - bar_height: int, bar height
    - trace_height_range: from 0 to 1
    """
    global MASTER_OUTPUT_VIS

    MASTER_OUTPUT_VIS = MASTER_OUTPUT.copy()
    MASTER_OUTPUT_VIS['trace_value_str'] = MASTER_OUTPUT_VIS['trace_value'].apply(lambda x: '%.3f' % x)
    # Trace width, apply log transformation to trace_value log2(x+4)*10
    # MASTER_OUTPUT_VIS['trace_height'] = MASTER_OUTPUT_VIS['trace_value'].apply(lambda x: int(np.log2(x + 4) * 10))

    gantts = ['gantt_main', 'gantt_fac', 'gantt_ttl_shortfall', 'gantt_equity']
    gantts_titles = ['Project needs and matched entries (HK$B)',
                     'Loan facilities available (leftover) (HK$B)',
                     'Total project needs (HK$B)',
                     'Equity (HK$B)']

    # Data filters for different gantt charts
    conditions_to_display = {'stage': MASTER_OUTPUT_VIS['stage'] == stage_to_display,
                             'project': MASTER_OUTPUT_VIS['project_name'].isin(projects_to_display)}
    conditions_to_display['gantt_main'] = (MASTER_OUTPUT_VIS['output_type'].isin(['shortfall', 'matched'])) & \
        conditions_to_display['stage'] & conditions_to_display['project']
    conditions_to_display['gantt_fac'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] != 'Equity')
    conditions_to_display['gantt_ttl_shortfall'] = (MASTER_OUTPUT_VIS['output_type'] == 'ttl_shortfall') & \
        conditions_to_display['stage']
    conditions_to_display['gantt_equity'] = (MASTER_OUTPUT_VIS['output_type'] == 'leftover') & \
        conditions_to_display['stage'] & (MASTER_OUTPUT_VIS['loan_type'] == 'Equity')

    subplot_data = {g: MASTER_OUTPUT_VIS[conditions_to_display[g]] for g in gantts}

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
            'equity-Equity.*': 'hotpink'
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
                     ticktext=DATE_TICK_TEXTS,
                     tickvals=DATE_TICK_VALUES)
    fig.update_layout(coloraxis=dict(colorscale='tempo'), showlegend=False,
                      barmode='overlay')
    fig.update_layout(xaxis_showticklabels=True, xaxis2_showticklabels=True)
    return fig


def add_dashboard(server):
    """Create Plotly Dash dashboard."""
    # Initial matching run with default values
    load_data()
    preprocess_data(tpp_date_delta_ymd=DEFAULT_TPP_DATE_DELTA_YMD,
                    equity_amt_in_b=[DASH_CONFIG['equity']['amt_in_billion']],
                    uc_evergreen=DASH_CONFIG['uc_evergreen'])
    init_working_data()
    matching_main_proc(scheme=DASH_CONFIG['matching_scheme']['scheme_id'])

    # Initialize Dash UI
    all_stage_dicts = list(STAGED_OUTPUTS_METADATA['stages'].values())
    all_stages = [stage_dict['value'] for stage_dict in all_stage_dicts]
    all_projects = MASTER_OUTPUT['project_name'].dropna().unique()

    dash_app = Dash(
        server=server,
        # routes_pathname_prefix='/dashboard_01/',
        url_base_pathname=URL_BASE,
        suppress_callback_exceptions=True,
        external_stylesheets=['/static/style.css']
    )

    dash_app.layout = \
        html.Div([
            html.H3('Matching 60% land costs with corporate loans'),

            # Panels
            html.Div([
                html.Div([
                    # Left: Matching parameters selection panel
                    html.H4('Matching parameters'),
                    # Target prepayment date delta (tpp_date_delta_ymd)
                    html.Label('Target prepayment date delta (year/ month/ day): ', style={'font-weight': 'bold'}),
                    dcc.Input(id='tpp-date-delta-year', type='number', placeholder='Year',
                              value=DEFAULT_TPP_DATE_DELTA_YMD[0], min=-99, max=0, step=1),
                    dcc.Input(id='tpp-date-delta-month', type='number', placeholder='Month',
                              value=DEFAULT_TPP_DATE_DELTA_YMD[1], min=-11, max=0, step=1),
                    dcc.Input(id='tpp-date-delta-day', type='number', placeholder='Day',
                              value=DEFAULT_TPP_DATE_DELTA_YMD[2], min=-31, max=0, step=1),
                    html.Br(),
                    # Equity
                    html.Label('Equity (HK$B): ', style={'font-weight': 'bold'}),
                    dcc.Input(id='input-equity-amt', type='number', placeholder='Equity',
                              value=DASH_CONFIG['equity']['amt_in_billion']),
                    html.Br(),
                    # Uncommitted Revolver options
                    html.Label('Uncommitted Revolver (UC) replacement options: ', style={'font-weight': 'bold'}),
                    dcc.Checklist(id='input-uc-options',
                                  options=[
                                      {'label': 'UC evergreen',
                                       'value': 'input_uc_evergreen'},
                                      {'label': 'UC need to fully cover Committed Revolver',
                                       'value': 'input_uc_full_cover'},
                                      {'label': 'Check replacement by net margin diff. x area '
                                                '(by net margin diff. only if not selected)',
                                       'value': 'input_uc_check_saving_by_area'},
                                  ],
                                  value=['input_uc_evergreen', 'input_uc_check_saving_by_area']),
                    # Button to rerun matching
                    html.Button('Rerun matching', id='btn-rerun-matching', className='button'),
                    # Button to refresh figure
                    # html.Button('Refresh chart', id='btn-refresh-chart', className='button'),
                    # Button to show/ hide matching scheme
                    html.Button('Show/ hide matching scheme', id='btn-show-scheme', className='button'),
                ], className='column left-column card'),
                # Right: Interactive visualization panel
                html.Div([
                    html.H4('Interactive visualization'),
                    dcc.Dropdown(options=all_stage_dicts, value=all_stages[0], id='filter-stage',),
                    dcc.Dropdown(options=all_projects, value=all_projects, multi=True, id='filter-projects',),
                    dcc.RadioItems(
                        options=[{'label': '1. Simple Gantt chart', 'value': 13},
                                 {'label': '2. Gantt chart with diff. bar height', 'value': 23}],
                        value=13,
                        inline=True,
                        id='chart-type',
                    ),
                    # html.Div([
                    #     html.Label('Bar height: ', style={'font-weight': 'bold'}, className='side-by-side'),
                    #     dcc.Slider(20, 100, value=30, included=True, marks=None, id='bar-height'),
                    # ]),
                    # html.Div([
                    #     html.Label('Bar height range (for Gantt with diff. bar height only): ',
                    #                style={'font-weight': 'bold'}, className='side-by-side'),
                    #     dcc.RangeSlider(0, 1.5, step=0.05, value=[0.25, 1], marks=None,
                    #                     id='bar-height-range'),
                    # ]),

                    html.Div([
                        html.Div([
                            html.Label('Bar height: ', style={'font-weight': 'bold'}, className='side-by-side'),
                            dcc.Slider(20, 100, value=30, included=True, marks=None, id='bar-height'),
                        ], className='column'),
                        html.Div([
                            html.Label('Bar height range (for chart type #2 only): ',
                                       style={'font-weight': 'bold'}, className='side-by-side'),
                            dcc.RangeSlider(0, 1.5, step=0.05, value=[0.25, 1], marks=None,
                                            id='bar-height-range'),
                        ], className='column'),
                    ], className='row'),

                ], className='column right-column card'),
            ], className='row'),

            # Display matching scheme
            html.Div([
                html.Div([
                    html.H4('Current matching scheme'),
                    html.Div(id='selected-input-display'),
                ], className='full-span-column card'),
            ], className='row', id='matching-scheme-info-box'),

            # Graph
            html.Div(dcc.Graph(id='figure-output'), className='card')
        ])

    # Initialize callbacks after the app is loaded
    init_callbacks(dash_app)

    # return dash_app.server
    return server


def init_callbacks(dash_app):
    @dash_app.callback(
        Output('selected-input-display', 'children'),
        Output('figure-output', 'figure'),
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
        State('input-uc-options', 'value')
    )
    def update_plot(*args):
        triggered_id = ctx.triggered_id
        global UC_FULL_COVER, UC_CHECK_SAVING_BY_AREA
        # stage_to_display, projects_to_display, chart_type, bar_height, bar_height_range, \
        #     _, _, tpp_date_delta_year, tpp_date_delta_month, tpp_date_delta_day, \
        #     input_equity_amt, input_uc_options = args
        stage_to_display, projects_to_display, chart_type, bar_height, bar_height_range, \
            _, tpp_date_delta_year, tpp_date_delta_month, tpp_date_delta_day, \
            input_equity_amt, input_uc_options = args

        # Set default values if None
        tpp_date_delta_year = tpp_date_delta_year if tpp_date_delta_year is not None else -1
        tpp_date_delta_month = tpp_date_delta_month if tpp_date_delta_year is not None else 0
        tpp_date_delta_day = tpp_date_delta_day if tpp_date_delta_day is not None else 0
        equity_amt = input_equity_amt if input_equity_amt is not None else DASH_CONFIG['equity']['amt_in_billion']
        uc_evergreen = 'input_uc_evergreen' in input_uc_options
        UC_FULL_COVER = 'input_uc_full_cover' in input_uc_options
        UC_CHECK_SAVING_BY_AREA = 'input_uc_check_saving_by_area' in input_uc_options

        # If clicked the rerun matching button, reload data and rerun matching and update master data
        if triggered_id == 'btn-rerun-matching':
            load_data()
            preprocess_data(tpp_date_delta_ymd=(tpp_date_delta_year, tpp_date_delta_month, tpp_date_delta_day),
                            equity_amt_in_b=[equity_amt],
                            uc_evergreen=uc_evergreen)
            init_working_data()
            matching_main_proc(scheme=DASH_CONFIG['matching_scheme']['scheme_id'])
        else:
            pass

        # Matching scheme display
        ms_display = list()
        # li 1
        ms_display.append(
            html.Li(STAGED_OUTPUTS_METADATA.get('short_description', '[description missing]'))
        )
        # li 2
        ms_display.append(
            html.Li(f'Target prepayment date for Term and Committed Revolver = Expiry date '
                    f'{tpp_date_delta_year} year {tpp_date_delta_month} month {tpp_date_delta_day} day')
        )
        # li 3
        if uc_evergreen:
            ms_display.append(html.Li('UC evergreen (no expiry date) assumed'))
        else:
            ms_display.append(html.Li('UC evergreen (no expiry date) not assumed'))
        # li 4
        ms_display_text_uc_rep = 'For UC replacement, '
        if UC_FULL_COVER:
            ms_display_text_uc_rep += 'UC has to fully cover the matched Committed Revolver entry, '
        else:
            ms_display_text_uc_rep += 'UC may partially cover the matched Committed Revolver entry, '
        if UC_CHECK_SAVING_BY_AREA:
            ms_display_text_uc_rep += 'and saving is calculated by net_margin difference x overlapping area'
        else:
            ms_display_text_uc_rep += 'and saving is calculated by net_margin difference only'

        ms_display.append(html.Li(ms_display_text_uc_rep))

        # Refresh figure
        # chart_type_dict = {
        #     # 11: get_gantt_1(stage_to_display, projects_to_display),
        #     # 12: get_gantt_2(stage_to_display, projects_to_display),
        #     13: get_gantt_3(stage_to_display, projects_to_display, bar_height=bar_height),
        #     23: get_gantt_diff_bar_width_3(stage_to_display, projects_to_display,
        #                                    bar_height=bar_height, trace_height_range=tuple(bar_height_range))
        # }
        # fig = chart_type_dict.get(chart_type, 13)  # If not found, use get_gantt_3

        if chart_type == 11:
            fig = get_gantt_1(stage_to_display, projects_to_display)
        elif chart_type == 12:
            fig = get_gantt_2(stage_to_display, projects_to_display)
        elif chart_type == 13:
            fig = get_gantt_3(stage_to_display, projects_to_display, bar_height=bar_height)
        elif chart_type == 23:
            fig = get_gantt_diff_bar_width_3(stage_to_display, projects_to_display,
                                             bar_height=bar_height, trace_height_range=tuple(bar_height_range))
        else:
            fig = get_gantt_3(stage_to_display, projects_to_display, bar_height=bar_height)

        return html.Ol(ms_display), fig

    @dash_app.callback(
        Output('matching-scheme-info-box', 'style'),
        Input('btn-show-scheme', 'n_clicks')
    )
    def toggle_matching_scheme_info_box(n_clicks):
        if n_clicks is None or n_clicks % 2 == 0:
            return {'display': 'none'}
        else:
            return {'display': 'block'}

