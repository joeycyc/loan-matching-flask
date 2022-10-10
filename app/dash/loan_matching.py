from .utils import *
import copy

class LoanMatching:

    def __init__(self, dash_config):
        """Initialize LoanMatching object."""
        '''(0) Read Config Data and Define Global Variables'''
        self.DASH_CONFIG = dash_config

        self.INPUT_DATA_DIR = self.DASH_CONFIG['data']['input']['dir']
        self.OUTPUT_DATA_DIR = self.DASH_CONFIG['data']['output']['dir']

        self.EXPORT_INTERIM_OUTPUT_CSV = self.DASH_CONFIG['data']['output']['to_export']['interim_output_csv']
        self.EXPORT_MASTER_OUTPUT_CSV = self.DASH_CONFIG['data']['output']['to_export']['master_output_csv']

        # Dates
        self.TODAY = dt.date.today()
        if self.DASH_CONFIG['dates']['day_zero']['is_today']:
            self.DAY_ZERO = self.TODAY  # Day 0 of matching
        else:
            self.DAY_ZERO = self.DASH_CONFIG['dates']['day_zero']['date']
        self.MIN_DATE = self.TODAY
        self.MAX_DATE = self.TODAY

        self.DATE_TICK_TEXTS = []
        self.DATE_TICK_VALUES = []

        # Containing the rawest form of data
        self.FACILITIES_DF = pd.DataFrame()
        self.PROJECTS_DF = pd.DataFrame()

        # Working data objects, which can be initialized, and mutated during matching
        self.WKING_FACILITIES_DF = pd.DataFrame()
        self.WKING_PROJECTS_DF = pd.DataFrame()
        self.WKING_FACILITIES_DICT = dict()
        # Example:
        # {545: {
        #     'loan_facility_id': 545, 'loan_id': 156, 'loan_sub_type': 'T', 'tranche': '', 'currency_id': 1,
        #     'facility_amount': 800000000.0, 'available_period_from': datetime.date(2021, 3, 31),
        #     'available_period_to': datetime.date(2021, 6, 30), 'reference_id': '202103MUFGHKD1700.0M3.0Y001',
        #     'borrower_id': 3.0, 'guarantor_id': 2.0, 'lender_id': 73.0, 'committed': 'C', 'loan_type': 'C',
        #     'project_loan_share': '', 'project_loan_share_jv': nan, 'facility_date': datetime.date(2021, 3, 31),
        #     'expiry_date': datetime.date(2024, 3, 29), 'withdrawn': 'N', 'lender_short_code': 'MUFG',
        #     'lender_name': 'MUFG Bank, Ltd.', 'lender_short_form': 'MUFG', 'company_name': 'W Finance Limited',
        #     'company_short_name': 'W Finance', 'company_type': 'BC', 'margin': 0.74, 'comm_fee_amount': nan,
        #     'comm_fee_margin': nan, 'comm_fee_margin_over': nan, 'comm_fee_margin_below': nan, 'all_in_price': 0.97,
        #     'net_margin': 0.74, 'upfront_fee': 0.69, 'facility_name': '3Y$0.8b-MUFG-CommittedTerm(545)',
        #     'target_prepayment_date': datetime.date(2023, 3, 29), 'facility_amount_inB': 0.8,
        #     'available_period_from_idx':-539, 'available_period_to_idx': -448, 'facility_date_idx': -539,
        #     'expiry_date_idx': 555, 'target_prepayment_date_idx': 189,
        #     'vector': array([0., 0., 0., ..., 0., 0., 0.])
        # },
        #     xxx: {}, xxx: {}, ...}
        self.WKING_PROJECTS_DICT = dict()
        # Example:
        # {19: {
        #     'project_id': 19, 'project_name': 'KTB', 'start_date': datetime.date(2018, 11, 30),
        #     'end_date': datetime.date(2022, 11, 30), 'land_cost': 6.36, 'project_loan_share_jv': nan,
        #     'land_cost_60_pct_inB': 3.81, 'solo_jv': 'Solo', 'reference': 'land_cost reviewed by BankingTeam',
        #     'start_date_idx': -1391, 'end_date_idx': 70, 'vector': array([0., 0., 0., ..., 0., 0., 0.])
        # },
        #     xxx: {}, xxx: {}, ...}
        self.MATCHED_ENTRIES = list()
        # Example:
        # [{'loan_facility_id': 545, 'project_id': 19, 'vector': array([0., 0., 0., ..., 0., 0., 0.]),
        #   'match_type': ('normal' | 'replacement' | 'reserved' | 'for_acquisition')},
        #  {}, {}]

        # Output data
        self.STAGED_OUTPUTS = dict()  # Output per stage in dict format
        self.STAGED_OUTPUTS_METADATA = dict()  # Description of the matching scheme
        self.MASTER_OUTPUT = pd.DataFrame()  # Master output in dataframe format, can be exported to CSV/ Excel
        # self.MASTER_OUTPUT_VIS = pd.DataFrame()  # Master output for visualization

        # Columns selected for output dataframes
        self.SELECTED_FAC_COLUMNS = [
            'loan_facility_id', 'loan_id', 'reference_id', 'facility_name',
            'lender_short_code', 'lender_short_form',
            'loan_type', 'committed', 'loan_sub_type', 'tranche',
            'available_period_from', 'available_period_to',
            'facility_date', 'expiry_date', 'target_prepayment_date',
            'facility_amount_inB', 'facility_amount',
            'margin',
            'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
            'comm_fee_margin_below', 'all_in_price', 'net_margin',
            'upfront_fee'
        ]
        self.SELECTED_PROJ_COLUMNS = [
            'project_id', 'project_name', 'solo_jv',
            'start_date', 'end_date',
            'land_cost_60_pct_inB'
        ]

        # Compatibility of matching scheme and visualization
        self.COMPATIBILITY_S2V = {
            'scheme_1': ['get_gantt_1', 'get_gantt_2', 'get_gantt_3'],
            'scheme_2': ['get_gantt_1', 'get_gantt_2', 'get_gantt_3']
        }
        self.COMPATIBILITY_V2S = dict()
        for k, vs in self.COMPATIBILITY_S2V.items():
            for v in vs:
                self.COMPATIBILITY_V2S[v] = self.COMPATIBILITY_V2S.get(v, []) + [k]

        # Global variables for matching
        # For load_data()
        self.LOAN_INPUT_DATA_TEMPLATE = self.DASH_CONFIG['data']['input']['loan_data_template']['selected']
        self.PROJECT_INPUT_DATA_TEMPLATE = self.DASH_CONFIG['data']['input']['project_data_template']['selected']
        self.LOAN_INPUT_DATA_METADATA = self.DASH_CONFIG['data']['input']['loan_data_template']\
            [self.LOAN_INPUT_DATA_TEMPLATE]
        self.PROJECT_INPUT_DATA_METADATA = self.DASH_CONFIG['data']['input']['project_data_template']\
            [self.PROJECT_INPUT_DATA_TEMPLATE]
        # For preprocess_data()
        self.TPP_DATE_DELTA_YMD = (-1, 0, 0)
        self.UC_EVERGREEN = self.DASH_CONFIG['uc_evergreen']
        # For matching_main_proc()
        # Uncommitted Revolver Replacement
        self.UC_FULL_COVER = self.DASH_CONFIG['matching_scheme']['full_cover']
        self.UC_CHECK_SAVING_BY_AREA = self.DASH_CONFIG['matching_scheme']['check_saving_by_area']
        # Committed Revolver Ceiling
        self.REVOLVER_CEILING = self.DASH_CONFIG['matching_scheme']['revolver_ceiling']
        self.REVOLVER_CEILING_FOR = self.DASH_CONFIG['matching_scheme']['revolver_ceiling_for']
        self.REVOLVER_TO_STAY = self.DASH_CONFIG['matching_scheme']['revolver_to_stay']
        # Manual matching
        self.MANUAL_MATCHING_RAW_STR = ''

        '''(1) Run initial matching'''
        self.load_data()
        self.preprocess_data()
        self.init_working_data()
        self.matching_main_proc(scheme=self.DASH_CONFIG['matching_scheme']['scheme_id'])

    def load_data(self,
                  loan_input_data_template=None,
                  project_input_data_template=None,
                  loan_input_data_metadata=None,
                  project_input_data_metadata=None,
                  input_data_dir=None):
        """Load data to FACILITIES_DF and PROJECTS_DF."""
        '''(0) Initialize data'''
        if loan_input_data_template is None:
            loan_input_data_template = self.LOAN_INPUT_DATA_TEMPLATE
        if project_input_data_template is None:
            project_input_data_template = self.PROJECT_INPUT_DATA_TEMPLATE
        if loan_input_data_metadata is None:
            loan_input_data_metadata = self.LOAN_INPUT_DATA_METADATA
        if project_input_data_metadata is None:
            project_input_data_metadata = self.PROJECT_INPUT_DATA_METADATA
        if input_data_dir is None:
            input_data_dir = self.INPUT_DATA_DIR


        '''(1) Load Loan Data'''
        if loan_input_data_template == 'template_l1':
            # Loan data from BTS
            bts_master_tbl_excel = loan_input_data_metadata['excels'][1]
            raw_df_company_grp = pd.read_excel(input_data_dir + bts_master_tbl_excel['name'],
                                               sheet_name=bts_master_tbl_excel['worksheets'][0])
            raw_df_company = pd.read_excel(input_data_dir + bts_master_tbl_excel['name'],
                                           sheet_name=bts_master_tbl_excel['worksheets'][1])
            raw_df_lender = pd.read_excel(input_data_dir + bts_master_tbl_excel['name'],
                                          sheet_name=bts_master_tbl_excel['worksheets'][2])

            bts_loan_profile_excel = loan_input_data_metadata['excels'][2]
            raw_df_loan = pd.read_excel(input_data_dir + bts_loan_profile_excel['name'],
                                        sheet_name=bts_loan_profile_excel['worksheets'][0])
            raw_df_facility = pd.read_excel(input_data_dir + bts_loan_profile_excel['name'],
                                            sheet_name=bts_loan_profile_excel['worksheets'][1])

            # Removing unwanted rows
            raw_df_company = raw_df_company[~raw_df_company['sys_company_id'].isna()]

            # Prepare single table for loan
            self.FACILITIES_DF = raw_df_facility.merge(raw_df_loan, left_on='loan_id', right_on='sys_loan_id',
                                                       how='left', suffixes=('_fac', '_loan')) \
                .merge(raw_df_lender, left_on='lender_id', right_on='sys_lender_id',
                       how='left', suffixes=('', '_lender')) \
                .merge(raw_df_company, left_on='borrower_id', right_on='sys_company_id',
                       how='left', suffixes=('', '_com'))
            self.FACILITIES_DF = self.FACILITIES_DF[[
                'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche',
                'currency_id', 'facility_amount', 'available_period_from',
                'available_period_to',
                'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
                'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
                'facility_date', 'expiry_date', 'withdrawn',
                'short_code', 'name', 'short_form',
                'company_name', 'company_short_name', 'type',
                'margin',
                'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
                'comm_fee_margin_below', 'all_in_price', 'net_margin',
                'upfront_fee'
            ]].copy()
            self.FACILITIES_DF.rename(columns={
                'sys_loan_facility_id': 'loan_facility_id',
                'short_code': 'lender_short_code',
                'short_form': 'lender_short_form',
                'name': 'lender_name',
                'type': 'company_type'
            }, inplace=True)
            # self.FACILITIES_DF.replace('NULL', None, inplace=True)
        elif loan_input_data_template == 'template_l2':
            # Loan data from BTS
            bts_tbl_excel = loan_input_data_metadata['excels'][1]
            raw_df_company_grp = pd.read_excel(input_data_dir + bts_tbl_excel['name'],
                                               sheet_name=bts_tbl_excel['worksheets'][0])
            raw_df_company = pd.read_excel(input_data_dir + bts_tbl_excel['name'],
                                           sheet_name=bts_tbl_excel['worksheets'][1])
            raw_df_lender = pd.read_excel(input_data_dir + bts_tbl_excel['name'],
                                          sheet_name=bts_tbl_excel['worksheets'][2])
            raw_df_loan = pd.read_excel(input_data_dir + bts_tbl_excel['name'],
                                        sheet_name=bts_tbl_excel['worksheets'][3])
            raw_df_facility = pd.read_excel(input_data_dir + bts_tbl_excel['name'],
                                            sheet_name=bts_tbl_excel['worksheets'][4])

            # Removing unwanted rows
            raw_df_company = raw_df_company[~raw_df_company['sys_company_id'].isna()]

            # Prepare single table for loan
            self.FACILITIES_DF = raw_df_facility.merge(raw_df_loan, left_on='loan_id', right_on='sys_loan_id',
                                                       how='left', suffixes=('_fac', '_loan')) \
                .merge(raw_df_lender, left_on='lender_id', right_on='sys_lender_id',
                       how='left', suffixes=('', '_lender')) \
                .merge(raw_df_company, left_on='borrower_id', right_on='sys_company_id',
                       how='left', suffixes=('', '_com'))
            self.FACILITIES_DF = self.FACILITIES_DF[[
                'sys_loan_facility_id', 'loan_id', 'loan_sub_type', 'tranche',
                'currency_id', 'facility_amount', 'available_period_from',
                'available_period_to',
                'reference_id', 'borrower_id', 'guarantor_id', 'lender_id',
                'committed', 'loan_type', 'project_loan_share', 'project_loan_share_jv',
                'facility_date', 'expiry_date', 'withdrawn',
                'short_code', 'name', 'short_form',
                'company_name', 'company_short_name', 'type',
                'margin',
                'comm_fee_amount', 'comm_fee_margin', 'comm_fee_margin_over',
                'comm_fee_margin_below', 'all_in_price', 'net_margin',
                'upfront_fee'
            ]].copy()
            self.FACILITIES_DF.rename(columns={
                'sys_loan_facility_id': 'loan_facility_id',
                'short_code': 'lender_short_code',
                'short_form': 'lender_short_form',
                'name': 'lender_name',
                'type': 'company_type'
            }, inplace=True)

        '''(2) Load Project Data'''
        # Project data
        # Project ID and Project Name
        if project_input_data_template == 'template_p1':
            project_tbl_excel = project_input_data_metadata['excels'][1]
            self.PROJECTS_DF = pd.read_excel(input_data_dir + project_tbl_excel['name'],
                                             sheet_name=project_tbl_excel['worksheets'][0])
            # Remove rows with NAs
            # self.PROJECTS_DF = self.PROJECTS_DF.dropna()

        '''(3) Update Global Variables'''
        # MIN_DATE not used for calculation, but DAY_ZERO
        self.MIN_DATE = min(*self.FACILITIES_DF['available_period_from'],
                            *self.PROJECTS_DF['start_date']).to_pydatetime().date()
        self.MAX_DATE = max(*self.FACILITIES_DF['expiry_date'],
                            *self.PROJECTS_DF['end_date']).to_pydatetime().date()
        self.DATE_TICK_TEXTS, self.DATE_TICK_VALUES = get_date_tick_lists(self.DAY_ZERO, self.MAX_DATE, short_form=True)

        return

    def preprocess_data(self, tpp_date_delta_ymd=None, equity_amt_in_b=None, uc_evergreen=None):
        """Pre-process data to update FACILITIES_DF and PROJECTS_DF.
        Args:
            - tpp_date_delta_ymd: The time delta in year, month, day between loan expiry date and target prepayment date
            - equity_amount_in_b: List of equity amount in HK$B
            - uc_evergreen: bool, whether the Uncommitted Revolver (UC-RTN) is assumed to have no maturity date
        """
        '''(0) Initialize data'''
        if tpp_date_delta_ymd is None:
            tpp_date_delta_ymd = self.TPP_DATE_DELTA_YMD
        if equity_amt_in_b is None:
            equity_amt_in_b = [self.DASH_CONFIG['equity']['amt_in_billion']]
        if uc_evergreen is None:
            uc_evergreen = self.UC_EVERGREEN

        '''(1) Pre-process Loan Data'''
        # Fill NA
        str_fields = ['lender_short_code', 'loan_type', 'project_loan_share', 'committed', 'loan_sub_type', 'tranche']
        self.FACILITIES_DF[str_fields] = self.FACILITIES_DF[str_fields].fillna('')

        # Concatenate info as identifier, e.g. '5Y$1.0b-CCB-CommittedTerm(123)'
        self.FACILITIES_DF['facility_name'] = \
            (self.FACILITIES_DF['expiry_date'] - self.FACILITIES_DF['facility_date']).apply(
                lambda x: str(round(x.days / 365 * 2) / 2).replace('.0', '')) + 'Y' + \
            '$' + round(self.FACILITIES_DF['facility_amount'] / 1E9, 1).astype(str) + 'b' + '-' + \
            self.FACILITIES_DF['lender_short_code'] + '-' + \
            self.FACILITIES_DF['committed'].replace({'C': 'Committed', 'U': 'Uncommitted'}) + \
            self.FACILITIES_DF['loan_sub_type'].replace({'T': 'Term', 'R': 'RTN'}) + \
            '(' + self.FACILITIES_DF['loan_facility_id'].astype(str) + ')'

        # Target prepayment date, default delta against expiry_date is year=-1, month=0, day=0
        if uc_evergreen:
            tpd_lst = []
            for row in self.FACILITIES_DF[['committed', 'loan_sub_type', 'expiry_date']].to_dict('records'):
                if row['committed'] == 'U' and row['loan_sub_type'] == 'R':
                    tpd_lst.append(self.MAX_DATE)
                elif row['loan_sub_type'] in ['T', 'R']:
                    tpd_lst.append(offset_date(row['expiry_date'], *tpp_date_delta_ymd))
                else:
                    tpd_lst.append(row['expiry_date'])
            self.FACILITIES_DF['target_prepayment_date'] = tpd_lst
        else:
            self.FACILITIES_DF['target_prepayment_date'] = \
                self.FACILITIES_DF[['loan_sub_type', 'expiry_date']].apply(
                    lambda x: offset_date(x[1], *tpp_date_delta_ymd) if x[0] in ['T', 'R'] else x[1], axis=1)

        # Facility amount in HK$B
        self.FACILITIES_DF['facility_amount_inB'] = self.FACILITIES_DF['facility_amount'] / 1E9

        # Dummy data: Equity, add equity entries one by one
        for i, amt_in_b in enumerate(equity_amt_in_b):
            if len(equity_amt_in_b) == 1:
                facility_name = 'W-Equity'
            else:
                facility_name = 'W-Equity-' + str(i + 1)
            equity = {
                'loan_facility_id': 99990 + i + 1,
                'loan_id': 99990 + i + 1,
                'loan_sub_type': 'Equity',
                'currency_id': 1,
                'facility_amount': amt_in_b * 1E9,
                'available_period_from': self.DAY_ZERO,
                'available_period_to': self.MAX_DATE,
                'loan_type': 'Equity',
                'facility_date': self.DAY_ZERO,
                'expiry_date': self.MAX_DATE,
                'withdrawn': 'N',
                'lender_short_code': 'W',
                'lender_name': 'Wheelock',
                'lender_short_form': 'W',
                'facility_name': facility_name,
                'target_prepayment_date': self.MAX_DATE,
                'facility_amount_inB': amt_in_b
            }

            self.FACILITIES_DF = pd.concat([self.FACILITIES_DF, pd.DataFrame({k: [v] for k, v in equity.items()})],
                                           ignore_index=True, sort=False)

        # Change data type of date
        fac_date_fields = ['available_period_from', 'available_period_to',
                           'facility_date', 'expiry_date', 'target_prepayment_date']
        for field in fac_date_fields:
            self.FACILITIES_DF[field] = self.FACILITIES_DF[field].apply(std_date)
            self.FACILITIES_DF[field + '_idx'] = self.FACILITIES_DF[field].apply(lambda x: date2idx(x, self.DAY_ZERO))

        '''(2) Pre-process Project Data'''
        # Change data type of date
        proj_date_fields = ['start_date', 'end_date']
        for field in proj_date_fields:
            self.PROJECTS_DF[field] = self.PROJECTS_DF[field].apply(std_date)
            self.PROJECTS_DF[field + '_idx'] = self.PROJECTS_DF[field].apply(lambda x: date2idx(x, self.DAY_ZERO))

        return

    def init_working_data(self):
        """Initialize WKING_FACILITIES_DF, WKING_PROJECTS_DF,
        WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES, and
        STAGED_OUTPUTS before matching."""

        # Filter unwanted records
        # 1. Project Loan: facilties.loan_type = 'P',
        # 2. Loan is cancelled: facilities.withdrawn == 'Y',
        # 3. Past loan/ project: loan.target_prepayment_date < DAY_ZERO project.end_date < DAY_ZERO
        self.WKING_FACILITIES_DF = self.FACILITIES_DF[
            (self.FACILITIES_DF['loan_type'] != 'P') &
            (self.FACILITIES_DF['withdrawn'] != 'Y') &
            (self.FACILITIES_DF['target_prepayment_date'] >= self.DAY_ZERO)
        ]
        self.WKING_PROJECTS_DF = self.PROJECTS_DF[self.PROJECTS_DF['end_date'] >= self.DAY_ZERO]

        # Find boundaries
        max_date_idx = max(*self.WKING_FACILITIES_DF['target_prepayment_date_idx'],
                           *self.WKING_PROJECTS_DF['end_date_idx'])

        # Initialize working data {loan_facility_id: np.array([vector])}
        wking_facilities_list = self.WKING_FACILITIES_DF.to_dict('records')
        for record in wking_facilities_list:
            start = max(0, record['available_period_from_idx'])
            end = record['target_prepayment_date_idx']
            vec = np.zeros(max_date_idx + 1)
            vec[start:end + 1] = record['facility_amount_inB']
            record['vector'] = vec
        # Convert to dict with loan_facility_id as index
        self.WKING_FACILITIES_DICT = {fac['loan_facility_id']: fac for fac in wking_facilities_list}

        # Initialize working data {project_id: np.array([vector])}
        wking_projects_list = self.WKING_PROJECTS_DF.to_dict('records')
        for record in wking_projects_list:
            start = max(0, record['start_date_idx'])
            end = record['end_date_idx']
            vec = np.zeros(max_date_idx + 1)
            vec[start:end + 1] = record['land_cost_60_pct_inB']
            record['vector'] = vec
        # Convert to dict with project_id as index
        self.WKING_PROJECTS_DICT = {proj['project_id']: proj for proj in wking_projects_list}

        # Initialize working data [{'loan_facility_id': int, 'project_id': int, 'vectors': [vector]}]
        self.MATCHED_ENTRIES = list()

        # Initialize staged output
        self.STAGED_OUTPUTS = dict()
        self.STAGED_OUTPUTS_METADATA = dict()
        self.MASTER_OUTPUT = pd.DataFrame()
        # self.MASTER_OUTPUT_VIS = pd.DataFrame()

        # 3 key working objects for matching: WKING_FACILITIES_DICT, WKING_PROJECTS_DICT, MATCHED_ENTRIES
        #  and their updated states per each stage of matching will be stored in STAGED_OUTPUTS
        return

    def matching_by_area(self, fac_idxs: list, proj_idxs: list, match_type: str = 'normal', **kwargs):
        """A procedure to perform matching given
        Args:
            - fac_idxs: list of facility ids
            - proj_idxs: list of project ids
            - match_type: a free text remark of the type of matched entry, e.g. 'normal', 'reserved', 'replacement'
        return: void
        """
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
                overlapping = np.minimum(self.WKING_FACILITIES_DICT[fac_idx]['vector'],
                                         self.WKING_PROJECTS_DICT[proj_idx]['vector'])
                overlapping_area = sum(overlapping)
                if overlapping_area > max_overlapping_area:  # Better match found
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
            else:  # max_overlapping_area > 0 -> There is a match
                self.MATCHED_ENTRIES.append(best_match_entry)
                # Update values in master
                self.WKING_FACILITIES_DICT[best_match_fac_idx]['vector'] -= best_overlapping
                self.WKING_PROJECTS_DICT[best_match_proj_idx]['vector'] -= best_overlapping
        return

    def save_to_staged_output(self, stage: str):
        """Standard procedure to get total shortfall vector and save staged output to STAGED_OUTPUTS
        Args:
            - stage: Stage number
        """
        # Total shortfall
        ttl_shortfall_vec = np.sum(np.stack([v['vector'] for v in self.WKING_PROJECTS_DICT.values()]), axis=0)
        # Output for a stage
        self.STAGED_OUTPUTS['stage ' + str(stage)] = {
            'fac': copy.deepcopy(self.WKING_FACILITIES_DICT),
            'proj': copy.deepcopy(self.WKING_PROJECTS_DICT),
            'matched': copy.deepcopy(self.MATCHED_ENTRIES),
            'ttl_shortfall_vec': ttl_shortfall_vec
        }
        return

    def std_solo_then_jv_matching(self, stage: str, loan_sub_type: str):
        """Standard Solo -> JV matching for a specific stage
        Args:
            - stage: Stage number
            - loan_sub_type: Loan subtype, either '<Initial>', 'T', 'R', 'Equity'
        """
        for sjn, solo_jv in enumerate(['Solo', 'JV']):
            # Sub-set the matching batch
            proj_idxs = [proj['project_id'] for proj in list(self.WKING_PROJECTS_DICT.values())
                         if proj['solo_jv'] == solo_jv]
            if loan_sub_type == 'R':  # For Stage 2, only select Committed Revolver
                fac_idxs = [fac['loan_facility_id'] for fac in list(self.WKING_FACILITIES_DICT.values())
                            if (fac['loan_sub_type'] == loan_sub_type) and (fac['committed'] == 'C')]
            else:
                fac_idxs = [fac['loan_facility_id'] for fac in list(self.WKING_FACILITIES_DICT.values())
                            if fac['loan_sub_type'] == loan_sub_type]
            # Matching by area
            self.matching_by_area(fac_idxs, proj_idxs)
        # Calculate total shortfall and output for this stage
        self.save_to_staged_output(stage)
        return

    def replace_matched_entries(self, full_cover: bool = True, check_saving_by_area: bool = True):
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
        # loan_facility_id of Uncommitted Revolver (UC-RTN)
        ucrtn_idxs = [fac['loan_facility_id'] for fac in list(self.WKING_FACILITIES_DICT.values())
                      if (fac['loan_sub_type'] == 'R') and (fac['committed'] == 'U')]
        # The position nos. in MATCHED_ENTRIES for matched Committed Revolver (C-RTN)
        crtn_matched_entry_poss = [i for i, me in enumerate(self.MATCHED_ENTRIES)
                                   if ((self.WKING_FACILITIES_DICT[me['loan_facility_id']]['loan_sub_type'] == 'R') and
                                       (self.WKING_FACILITIES_DICT[me['loan_facility_id']]['committed'] == 'C'))]
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
                    ucrtn_info = self.WKING_FACILITIES_DICT[ucrtn_idx]  # Get the latest UC-RTN info
                    best_replacement_vector = np.array([])
                    max_cost_saving = 0
                    best_overlapping_area = 0  # The overlapping area of best match, not necessarily the max
                    crtn_matched_entry_to_reserve_pos = -1  # Position no. in MATCHED_ENTRIES, NOT loan_facility_id
                    crtn_matched_entry_to_reserve_fac_idx = -1  # loan_facility_id of C-RTN matched entry
                    crtn_matched_entry_to_reserve_proj_idx = -1  # project_id of C-RTN matched entry
                    # End of initialization #

                    # Loop over the matched C-RTNs to check if any replacement could be made #
                    for crtn_matched_entry_pos in crtn_matched_entry_poss:
                        crtn_matched_entry = self.MATCHED_ENTRIES[crtn_matched_entry_pos]
                        wking_crtn_fac_idx = crtn_matched_entry['loan_facility_id']
                        wking_proj_idx = crtn_matched_entry['project_id']
                        crtn_info = self.WKING_FACILITIES_DICT[wking_crtn_fac_idx]
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
                                len(self.MATCHED_ENTRIES)
                            self.MATCHED_ENTRIES.append({
                                'loan_facility_id': ucrtn_idx,
                                'project_id': crtn_matched_entry_to_reserve_proj_idx,
                                'vector': best_replacement_vector.copy(),
                                'match_type': 'replacement'
                            })
                        else:  # Matched entry already exists, update by adding vector values
                            pos = ucrtn_matched_entry_poss_dict[(ucrtn_idx, crtn_matched_entry_to_reserve_proj_idx)]
                            self.MATCHED_ENTRIES[pos]['vector'] += best_replacement_vector
                        # Update UC-RTN vector values in master
                        self.WKING_FACILITIES_DICT[ucrtn_idx]['vector'] -= best_replacement_vector

                        # Append (if new)/ Update (if exist) 'replaced' matched C-RTN entry
                        if (crtn_matched_entry_to_reserve_fac_idx, crtn_matched_entry_to_reserve_proj_idx) \
                                not in crtn_reserved_entry_poss_dict:
                            # New reserved entry, append
                            crtn_reserved_entry_poss_dict[
                                (crtn_matched_entry_to_reserve_fac_idx,
                                 crtn_matched_entry_to_reserve_proj_idx)
                            ] = len(self.MATCHED_ENTRIES)
                            self.MATCHED_ENTRIES.append({
                                'loan_facility_id': crtn_matched_entry_to_reserve_fac_idx,
                                'project_id': crtn_matched_entry_to_reserve_proj_idx,
                                'vector': best_replacement_vector.copy(),
                                'match_type': 'reserved'
                            })
                        else:  # Replaced entry already exists, update by adding vector values
                            pos = crtn_reserved_entry_poss_dict[(crtn_matched_entry_to_reserve_fac_idx,
                                                                 crtn_matched_entry_to_reserve_proj_idx)]
                            self.MATCHED_ENTRIES[pos]['vector'] += best_replacement_vector
                        # Update matched C-RTN entry in master
                        self.MATCHED_ENTRIES[crtn_matched_entry_to_reserve_pos]['vector'] -= best_replacement_vector

                    # End of Check if there is any replacement, if yes, update the master data #

            # Remove matched entries if sum of all vector elements == 0
            self.MATCHED_ENTRIES = [matched_entry for matched_entry in self.MATCHED_ENTRIES
                                    if sum(matched_entry['vector']) != 0]

        else:  # No UC-RTN or matched C-RTN
            pass

        return

    def set_aside_revolver(self,
                           revolver_ceiling=99999.0,
                           revolver_ceiling_for='loan_matching',
                           revolver_to_stay='max_cost'):
        """A procedure for setting aside revolver
        Args:
            - revolver_ceiling: Maximum amount of Committed Revolver included in matching (in HK$B)
            - revolver_ceiling_for: Whether the ceiling is set for 'loan_matching' or 'acquisition'
            - revolver_to_stay: Criteria of choose which revolver to stay for loan_matching or acquisition,
                either it is 'max_cost', 'min_cost', 'max_area', 'min_area', 'max_amount', 'min_amount',
                'max_period', 'min_period', 'max_net_margin' or 'min_net_margin',

        Target is to divide Committed Revolver facilities into 2 groups: For loan matching OR For acquisition;
        If revolver_ceiling of $10B is set for loan_matching with revolver_to_stay='max_cost',
        it means only max $10B of Committed Revolver can be used for loan matching,
        and the amount exceeding $10B will be set aside for acquisition;
        the criteria for picking which Committed Revolver to stay for loan matching would be by 'max_cost'

        Criteria of picking which Committed Revolver facilities to stay for loan matching or acquisition:
            - 'max_cost'/ 'min_cost': Facilities with high/ low total cost (net_margin x area) will stay
            - 'max_area'/ 'min_area': Facilities with large/ small area (period x facility_amount) will stay
            - 'max_amount'/ 'min_amount': Facilities with high/ low facility_amount with stay
            - 'max_period'/ 'min_period': Facilities with long/ short period (no. of days) will stay
            - 'max_net_margin'/ 'min_net_margin': Facilities with high/ low net margin will stay

        return: void
        """
        '''Input parameter value validation'''
        if not isinstance(revolver_ceiling, (int, float)):
            revolver_ceiling = 99999.0
        if revolver_ceiling_for not in ['loan_matching', 'acquisition']:
            revolver_ceiling_for = 'loan_matching'
        if revolver_to_stay not in ['max_cost', 'min_cost', 'max_area', 'min_area', 'max_amount', 'min_amount',
                                    'max_period', 'min_period', 'max_net_margin', 'min_net_margin']:
            revolver_to_stay = 'max_cost'
        mm_ = revolver_to_stay[:3]  # 'max' or 'min'
        metric_ = revolver_to_stay[4:]

        '''Initialize working data'''
        # Committed Revolver facilities indexes
        cr_fac_idxs = [fac['loan_facility_id'] for fac in list(self.WKING_FACILITIES_DICT.values())
                       if (fac['loan_sub_type'] == 'R') and (fac['committed'] == 'C')]
        # abort if there is no Committed Revolver
        if len(cr_fac_idxs) == 0:
            return
        cr_fac_dict = {k: v for k, v in self.WKING_FACILITIES_DICT.items() if k in cr_fac_idxs}
        # cr_arr = np.stack([v['vector'] for v in cr_fac_dict.values()])
        # Set the quota vector given with revolver_ceiling, when the quota used up (0) for a day,
        # no more revolver can stay
        stay_quota_vector = np.ones((self.MAX_DATE - self.DAY_ZERO).days + 1) * revolver_ceiling
        # Interim output: stay or leave, {fac_idx: vector}
        stay_dict = dict()
        leave_dict = dict()

        '''Sort the facilities by metric'''
        # Arrange data in DataFrame
        cr_fac_df = pd.DataFrame([[
            v['loan_facility_id'], v['facility_amount_inB'], v['available_period_from_idx'],
            v['target_prepayment_date_idx'], v['net_margin']
        ] for v in cr_fac_dict.values()])
        cr_fac_df.columns = ['loan_facility_id', 'facility_amount_inB', 'available_period_from_idx',
                             'target_prepayment_date_idx', 'net_margin']
        loan_facility_id_list = []
        amount_list = []
        period_list = []
        area_list = []
        cost_list = []
        for k, v in cr_fac_dict.items():
            loan_facility_id_list.append(k)
            amount_list.append(max(v['vector']))  # Max. loan facility amount as the amount
            period_list.append(np.count_nonzero(v['vector']))  # No. of days with non-zero amount
            area_list.append(sum(v['vector']))
            cost_list.append(sum(v['vector']) * v['net_margin'])
        cr_fac_df2 = pd.DataFrame({
            'loan_facility_id': loan_facility_id_list,
            'amount': amount_list,
            'period': period_list,
            'area': area_list,
            'cost': cost_list
        })
        cr_fac_df = cr_fac_df.merge(cr_fac_df2, how='left', on='loan_facility_id')
        # cr_fac_df.columns = ['loan_facility_id', 'amount', 'available_period_from_idx',
        #                      'target_prepayment_date_idx', 'net_margin']
        # cr_fac_df['period'] = cr_fac_df[['available_period_from_idx', 'target_prepayment_date_idx']].apply(
        #     lambda x: x[1] - max(0, x[0]) + 1, axis=1)
        # cr_fac_df['area'] = cr_fac_df['amount'] * cr_fac_df['period']
        # cr_fac_df['cost'] = cr_fac_df['area'] * cr_fac_df['net_margin']
        # Sorting
        asc_ = True if mm_ == 'min' else False
        cr_fac_df.sort_values(by=metric_, ascending=asc_, ignore_index=True, inplace=True)
        cr_fac_idxs_sorted = list(cr_fac_df['loan_facility_id'])

        '''Do the division: stay or leave'''
        # Loop cr_fac_idxs_sorted multiple times until exhausting the stay_quota_vector
        # Input: cr_fac_dict, stay_quota_vector
        # Output: stay_dict and leave_dict
        while True:
            # Initialization
            no_more_overlapping = True
            for cr_fac_idx in cr_fac_idxs_sorted:
                overlapping = np.minimum(cr_fac_dict[cr_fac_idx]['vector'], stay_quota_vector)
                overlapping_area = sum(overlapping)
                if overlapping_area > 0:
                    no_more_overlapping = False
                    if cr_fac_idx in stay_dict:
                        stay_dict[cr_fac_idx] += overlapping
                    else:  # cr_fac_idx is new for stay_dict
                        stay_dict[cr_fac_idx] = overlapping
                    cr_fac_dict[cr_fac_idx]['vector'] -= overlapping
                    stay_quota_vector -= overlapping
            if no_more_overlapping:  # No more overlapping between Committed Revolvers' vectors and stay_quota_vector
                # Dump the rest to leave_dict
                for cr_fac_idx in cr_fac_idxs_sorted:
                    if sum(cr_fac_dict[cr_fac_idx]['vector']) > 0:
                        leave_dict[cr_fac_idx] = cr_fac_dict[cr_fac_idx]['vector'].copy()
                break

        '''Update WKING_FACILITIES_DICT's vector and Append 'for_acquisition' entries to MATCHED_ENTRIES'''
        if revolver_ceiling_for == 'loan_matching':
            # stay for loan matching
            for fac_idx in cr_fac_idxs_sorted:
                if fac_idx in stay_dict:
                    self.WKING_FACILITIES_DICT[fac_idx]['vector'] = stay_dict[fac_idx].copy()
                else:  # not found in stay_dict, put zero vector
                    self.WKING_FACILITIES_DICT[fac_idx]['vector'] = np.zeros((self.MAX_DATE - self.DAY_ZERO).days + 1)
            # leave for acquisition
            for fac_idx, vec in leave_dict.items():
                self.MATCHED_ENTRIES.append({'loan_facility_id': fac_idx,
                                             'project_id': -1,
                                             'vector': vec.copy(),
                                             'match_type': 'for_acquisition'})
        else:  # revolver_ceiling_for == 'acquisition'
            # leave for loan matching
            for fac_idx in cr_fac_idxs_sorted:
                if fac_idx in leave_dict:
                    self.WKING_FACILITIES_DICT[fac_idx]['vector'] = leave_dict[fac_idx].copy()
                else:  # not found in leave_dict, put zero vector
                    self.WKING_FACILITIES_DICT[fac_idx]['vector'] = np.zeros((self.MAX_DATE - self.DAY_ZERO).days + 1)
            # stay for acquisition
            for fac_idx, vec in stay_dict.items():
                self.MATCHED_ENTRIES.append({'loan_facility_id': fac_idx,
                                             'project_id': -1,
                                             'vector': vec.copy(),
                                             'match_type': 'for_acquisition'})

        return

    def matching_main_proc(self,
                           scheme: int = 1,
                           uc_full_cover=None,
                           uc_check_saving_by_area=None,
                           revolver_ceiling=None,
                           revolver_ceiling_for=None,
                           revolver_to_stay=None):
        """Main matching procedures to produce staged outputs.
        Args:
            - scheme: int, the matching scheme id
        """
        '''(0) Initialize data'''
        if uc_full_cover is None:
            uc_full_cover = self.UC_FULL_COVER
        if uc_check_saving_by_area is None:
            uc_check_saving_by_area = self.UC_CHECK_SAVING_BY_AREA
        if revolver_ceiling is None:
            revolver_ceiling = self.REVOLVER_CEILING
        if revolver_ceiling_for is None:
            revolver_ceiling_for = self.REVOLVER_CEILING_FOR
        if revolver_to_stay is None:
            revolver_to_stay = self.REVOLVER_TO_STAY

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
                self.std_solo_then_jv_matching(str(ln), loan_sub_type)

            # === Scheme 1: matching scheme metadata === #
            self.STAGED_OUTPUTS_METADATA = {
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
        # ===== End of Matching Scheme 1 ===== #

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
                self.std_solo_then_jv_matching(str(ln), loan_sub_type)

            # == Stage 2a: Replace matched Committed Revolver (C-RTN) with Uncommitted Revolver (UC-RTN) == #
            # For STAGED_OUTPUTS, 'proj' remains the same as that in Stage 2,
            # update 'fac' for Committed Revolver and
            # 'matched': MATCHED_ENTRIES (list of dicts), with each dict's format as
            #  {'loan_facility_id': fac_idx, 'project_id': proj_idx, 'vector': overlapping, 'match_type': match_type}
            self.replace_matched_entries(full_cover=uc_full_cover,
                                         check_saving_by_area=uc_check_saving_by_area)
            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('2a')

            # == Stage 3: Match equity == #
            self.std_solo_then_jv_matching('3', 'Equity')

            # === Scheme 2: matching scheme metadata === #
            self.STAGED_OUTPUTS_METADATA = {
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
                        'label': 'Stage 2a: Term + Committed RTN + Uncommitted RTN Replacement'},
                    4: {'value': 'stage 3',
                        'label': 'Stage 3: Term + Committed RTN + Uncommitted RTN Replacement + Equity'}
                }
            }
        # ===== End of Matching Scheme 2 ===== #

        # ===== Matching Scheme 3 ===== #
        elif scheme == 3:
            # === Scheme 3: matching procedure === #
            # Stage 0: Initial
            # Stage 0b: Revolver ceiling applied, set aside revolver (for acquisition) not to be included in matching
            # Stage 1: Term Facilities vs. Solo then JV
            # Stage 2: Committed Revolver vs. Solo then JV
            # Stage 2a: Uncommitted Revolver to replace Committed Revolver
            # Stage 3: Equity vs. Solo then JV

            # Output to STAGED_OUTPUTS
            # == Stage 0: Initial == #
            self.std_solo_then_jv_matching('0', '<Initial>')

            # == Stage 0b: Set aside Committed Revolver due to revolver ceiling == #
            self.set_aside_revolver(revolver_ceiling, revolver_ceiling_for, revolver_to_stay)
            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('0b')

            # == Stage 1 & 2: Term and Committed Revolver == #
            self.std_solo_then_jv_matching('1', 'T')
            self.std_solo_then_jv_matching('2', 'R')

            # == Stage 2a: Replace matched Committed Revolver (C-RTN) with Uncommitted Revolver (UC-RTN) == #
            # For STAGED_OUTPUTS, 'proj' remains the same as that in Stage 2,
            # update 'fac' for Committed Revolver and
            # 'matched': MATCHED_ENTRIES (list of dicts), with each dict's format as
            #  {'loan_facility_id': fac_idx, 'project_id': proj_idx, 'vector': overlapping, 'match_type': match_type}
            self.replace_matched_entries(full_cover=uc_full_cover,
                                         check_saving_by_area=uc_check_saving_by_area)
            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('2a')

            # == Stage 3: Match equity == #
            self.std_solo_then_jv_matching('3', 'Equity')

            # === Scheme 3: matching scheme metadata === #
            self.STAGED_OUTPUTS_METADATA = {
                '_id': 3,
                'name': 'T-R-E plus UC-RTN replacement and revolver ceiling',
                'description': 'Stage 0: Initial; '
                               'Stage 0b: Set aside Committed Revolver for acquisition'
                               'Stage 1: Term Facilities vs. Solo then JV; '
                               'Stage 2: Committed Revolver vs. Solo then JV; '
                               'Stage 2a: Uncommitted Revolver to replace Committed Revolver; '
                               'Stage 3: Equity vs. Solo then JV',
                'short_description': 'Term -> Committed Revolver -> UC replacement -> Equity, Solo -> JV per stage, '
                                     'with revolver ceiling',
                'stages': {
                    0: {'value': 'stage 0', 'label': 'Stage 0: Initial'},
                    1: {'value': 'stage 0b',
                        'label': 'Stage 0b: Set aside Committed RTN for acquisition'},
                    2: {'value': 'stage 1', 'label': 'Stage 1: Term'},
                    3: {'value': 'stage 2', 'label': 'Stage 2: Term + Committed RTN'},
                    4: {'value': 'stage 2a',
                        'label': 'Stage 2a: Term + Committed RTN + Uncommitted RTN Replacement'},
                    5: {'value': 'stage 3',
                        'label': 'Stage 3: Term + Committed RTN + Uncommitted RTN Replacement + Equity'}
                }
            }
        # ===== End of Matching Scheme 3 ===== #

        # ===== Matching Scheme 4 ===== #
        elif scheme == 4:
            # === Scheme 3: matching procedure === #
            # Stage 0: Initial
            # Stage 0a: Manual matching
            # Stage 0b: Revolver ceiling applied, set aside revolver (for acquisition) not to be included in matching
            # Stage 1: Term Facilities vs. Solo then JV
            # Stage 2: Committed Revolver vs. Solo then JV
            # Stage 2a: Uncommitted Revolver to replace Committed Revolver
            # Stage 3: Equity vs. Solo then JV

            # Output to STAGED_OUTPUTS
            # == Stage 0: Initial == #
            self.std_solo_then_jv_matching('0', '<Initial>')

            # == Stage 0a: Manual matching == #
            # Validate input string
            all_projects = self.WKING_PROJECTS_DF['project_name'].unique()
            proj_idx_dict = {k: v for k, v in self.WKING_PROJECTS_DF[[
                'project_name', 'project_id']].drop_duplicates().to_dict('tight')['data']}  # {project_name: proj_idx}
            all_fac_idxs = self.WKING_FACILITIES_DF['loan_facility_id'].unique()
            manual_matching_entries = []  # [(proj_idx, fac_idx)]
            if isinstance(self.MANUAL_MATCHING_RAW_STR, str):
                manual_matching_raw_str = self.MANUAL_MATCHING_RAW_STR.rstrip('; ')
                manual_matching_raw_str = re.sub(';+', ';', manual_matching_raw_str)
                manual_matching_entry_strs = manual_matching_raw_str.split(';')
                for manual_matching_entry_str in manual_matching_entry_strs:
                    proj, fac_idx = None, None
                    if len(re.findall('\|', manual_matching_entry_str)) == 1:
                        x1, x2 = manual_matching_entry_str.split('|')
                        x1 = x1.strip()
                        x2 = x2.strip()
                        if re.match(r'^\d+$', x1):
                            proj, fac_idx = x2, int(x1)
                        elif re.match(r'^\d+$', x2):
                            proj, fac_idx = x1, int(x2)
                        else:
                            pass
                    if (proj in all_projects) and (fac_idx in all_fac_idxs):
                        manual_matching_entries.append((proj_idx_dict[proj], fac_idx))  # append (proj_idx, fac_idx)
            # Manual matching
            if len(manual_matching_entries) > 0:
                for p, f in manual_matching_entries:
                    self.matching_by_area([f], [p])

            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('0a')

            # == Stage 0b: Set aside Committed Revolver due to revolver ceiling == #
            self.set_aside_revolver(revolver_ceiling, revolver_ceiling_for, revolver_to_stay)
            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('0b')

            # == Stage 1 & 2: Term and Committed Revolver == #
            self.std_solo_then_jv_matching('1', 'T')
            self.std_solo_then_jv_matching('2', 'R')

            # == Stage 2a: Replace matched Committed Revolver (C-RTN) with Uncommitted Revolver (UC-RTN) == #
            # For STAGED_OUTPUTS, 'proj' remains the same as that in Stage 2,
            # update 'fac' for Committed Revolver and
            # 'matched': MATCHED_ENTRIES (list of dicts), with each dict's format as
            #  {'loan_facility_id': fac_idx, 'project_id': proj_idx, 'vector': overlapping, 'match_type': match_type}
            self.replace_matched_entries(full_cover=uc_full_cover,
                                         check_saving_by_area=uc_check_saving_by_area)
            # Calculate total shortfall and output for this stage
            self.save_to_staged_output('2a')

            # == Stage 4: Match equity == #
            self.std_solo_then_jv_matching('3', 'Equity')

            # === Scheme 4: matching scheme metadata === #
            self.STAGED_OUTPUTS_METADATA = {
                '_id': 4,
                'name': 'T-R-E plus manual matching, revolver ceiling, and UC-RTN replacement',
                'description': 'Stage 0: Initial; '
                               'Stage 0a: Manual matching; '
                               'Stage 0b: Set aside Committed Revolver for acquisition'
                               'Stage 1: Term Facilities vs. Solo then JV; '
                               'Stage 2: Committed Revolver vs. Solo then JV; '
                               'Stage 2a: Uncommitted Revolver to replace Committed Revolver; '
                               'Stage 3: Equity vs. Solo then JV',
                'short_description': 'Term -> Manual matching -> Set aside Committed Revolver -> '
                                     'Committed Revolver -> UC replacement -> Equity, Solo -> JV per stage, ',
                'stages': {
                    0: {'value': 'stage 0', 'label': 'Stage 0: Initial'},
                    1: {'value': 'stage 0a', 'label': 'Stage 0a: Manual matching'},
                    2: {'value': 'stage 0b',
                        'label': 'Stage 0b: Set aside Committed RTN for acquisition'},
                    3: {'value': 'stage 1', 'label': 'Stage 1: Term'},
                    4: {'value': 'stage 2', 'label': 'Stage 2: Term + Committed RTN'},
                    5: {'value': 'stage 2a',
                        'label': 'Stage 2a: Term + Committed RTN + Uncommitted RTN Replacement'},
                    6: {'value': 'stage 3',
                        'label': 'Stage 3: Term + Committed RTN + Uncommitted RTN Replacement + Equity'}
                }
            }
        # ===== End of Matching Scheme 4 ===== #

        '''(2) Tidy up result'''
        self.MASTER_OUTPUT = pd.DataFrame()
        # (2.1) Loop over stages and store to centralized dfs

        def get_master_output_subset_df(stage_: str, stage_state_: dict, output_type: str, output_type_num: int,
                                        self_obj=self):
            """A procedure to generate subset of DataFrame in MASTER_OUTPUT
            Args:
                - stage_: value in format 'stage xx'
                - stage_state_: dictionary saving the staged output info
                - output_type (output_type_num): either 'matched' (1), 'shortfall' (2), 'leftover' (3),
                 'ttl_shortfall' (4), or 'for_acquisition' (5)
            """
            # Convert to dataframe
            output_dict = {
                'loan_facility_id': [],     # for output_type_num = 1,    3,    5
                'project_id': [],           # for output_type_num = 1, 2
                'match_type': [],           # for output_type_num = 1,          5
                'from_date_idx': [],        # for output_type_num = 1, 2, 3, 4, 5
                'to_date_idx': [],          # for output_type_num = 1, 2, 3, 4, 5
                'matched_amt_inB': [],      # for output_type_num = 1,          5
                'shortfall_amt_inB': [],    # for output_type_num =    2,    4
                'leftover_amt_inB': [],     # for output_type_num =       3
                'trace_value': []           # for output_type_num = 1, 2, 3, 4, 5
            }
            output_dict_keys = list(output_dict.keys())
            if output_type == 'matched':  # output_type_num = 1
                for entry_ in stage_state_['matched']:
                    if entry_['match_type'] not in ['for_acquisition']:  # Exclude those for_acquisition
                        for ff, tt, aa in vec2rects(entry_['vector']):
                            for output_dict_key, value in zip(
                                    output_dict_keys,
                                    [entry_['loan_facility_id'], entry_['project_id'], entry_['match_type'],
                                     ff, tt, aa, np.nan, np.nan, aa]):
                                output_dict[output_dict_key].append(value)
            elif output_type == 'shortfall':  # output_type_num = 2
                for kk, vv in stage_state_['proj'].items():
                    # Special for project: Show zero even the amount is filled
                    for ff, tt, aa in vec2rects(vv['vector'],
                                                preserve_zero=True,
                                                preserve_zero_st_idx=max(0, vv['start_date_idx']),
                                                preserve_zero_end_idx=vv['end_date_idx']):
                        for output_dict_key, value in zip(
                                output_dict_keys,
                                [np.nan, kk, '', ff, tt, np.nan, aa, np.nan, aa]):
                            output_dict[output_dict_key].append(value)
            elif output_type == 'leftover':  # output_type_num = 3
                for kk, vv in stage_state_['fac'].items():
                    for ff, tt, aa in vec2rects(vv['vector']):
                        for output_dict_key, value in zip(
                                output_dict_keys,
                                [kk, np.nan, '', ff, tt, np.nan, np.nan, aa, aa]):
                            output_dict[output_dict_key].append(value)
            elif output_type == 'ttl_shortfall':  # output_type_num = 4
                pzei_ = len(stage_state_['ttl_shortfall_vec']) - 1
                for ff, tt, aa in vec2rects(stage_state_['ttl_shortfall_vec'],
                                            preserve_zero=True,
                                            preserve_zero_st_idx=0,
                                            preserve_zero_end_idx=pzei_):
                    for output_dict_key, value in zip(
                            output_dict_keys,
                            [np.nan, np.nan, '', ff, tt, np.nan, aa, np.nan, aa]):
                        output_dict[output_dict_key].append(value)
            elif output_type == 'for_acquisition':  # output_type_num = 5
                for entry_ in stage_state_['matched']:
                    if entry_['match_type'] == 'for_acquisition':  # Only include those for_acquisition
                        for ff, tt, aa in vec2rects(entry_['vector']):
                            for output_dict_key, value in zip(
                                    output_dict_keys,
                                    [entry_['loan_facility_id'], np.nan, entry_['match_type'],
                                     ff, tt, aa, np.nan, np.nan, aa]):
                                output_dict[output_dict_key].append(value)
            else:  # Unknown output_type_num!!
                pass
            output_df = pd.DataFrame(output_dict)
            # Convert back to date columns
            output_df['from_date'] = output_df['from_date_idx'].apply(lambda x: idx2date(x, self_obj.DAY_ZERO))
            output_df['to_date'] = output_df['to_date_idx'].apply(lambda x: idx2date(x, self_obj.DAY_ZERO))
            # Add indicator columns
            output_df['stage'] = stage_
            output_df['output_type'] = output_type
            output_df['output_type_num'] = output_type_num
            # Augment dataframe - project and facility columns
            output_df = output_df.sort_values(['project_id', 'loan_facility_id']).reset_index(drop=True)

            if output_type in ['matched', 'shortfall']:  # for output_type_num = 1, 2 only
                output_df = output_df.merge(self_obj.PROJECTS_DF[self_obj.SELECTED_PROJ_COLUMNS],
                                            on='project_id',
                                            how='left')

            if output_type in ['matched', 'leftover', 'for_acquisition']:  # for output_type_num = 1, 3, 5 only
                output_df = output_df.merge(self_obj.FACILITIES_DF[self_obj.SELECTED_FAC_COLUMNS],
                                            on='loan_facility_id',
                                            how='left')

            # Augment dataframe - y_label
            if output_type == 'matched':  # for output_type_num = 1 only
                output_df['projectXfacility'] = output_df['project_name'] + ' | ' + output_df['facility_name'] + \
                                                ' (' + output_df['match_type'] + ')'
                output_df['projectXfacility'] = output_df['projectXfacility'].str.replace(' (normal)', '', regex=False)

            y_label_col_mapping = {
                'matched': 'projectXfacility',
                'shortfall': 'project_name',
                'leftover': 'facility_name',
                'for_acquisition': 'facility_name'
            }

            if output_type in y_label_col_mapping:  # for output_type_num = 1, 2, 3, 5 only
                output_df['y_label'] = output_df[y_label_col_mapping[output_type]]
            elif output_type == 'ttl_shortfall':  # for output_type_num = 4 only
                output_df['y_label'] = 'Total Shortfall'

            return output_df

        for stage, stage_state in self.STAGED_OUTPUTS.items():
            # (2.1.1 - 2.1.5)
            matched_entries_df = get_master_output_subset_df(stage, stage_state, 'matched', 1)
            shortfalls_df = get_master_output_subset_df(stage, stage_state, 'shortfall', 2)
            leftovers_df = get_master_output_subset_df(stage, stage_state, 'leftover', 3)
            ttl_shortfall_df = get_master_output_subset_df(stage, stage_state, 'ttl_shortfall', 4)
            for_acq_df = get_master_output_subset_df(stage, stage_state, 'for_acquisition', 5)

            # (2.1.9) Concat to MASTER_OUTPUT
            self.MASTER_OUTPUT = pd.concat(
                [self.MASTER_OUTPUT, matched_entries_df, shortfalls_df, leftovers_df, ttl_shortfall_df, for_acq_df],
                ignore_index=True, sort=False
            )

        # (2.2) Sort MASTER_OUTPUT:
        #  stage (asc) -> output_type_num (asc) -> solo_jv (Solo -> JV) -> project_id (asc)
        #  -> loan_sub_type (T -> R -> Equity) -> loan_id (asc) -> loan_facility_id (asc)
        self.MASTER_OUTPUT['solo_jv_rank'] = self.MASTER_OUTPUT['solo_jv'].replace({'Solo': 1, 'JV': 2})
        # self.MASTER_OUTPUT['loan_sub_type_rank'] =
        # self.MASTER_OUTPUT['loan_sub_type'].replace({'T': 1, 'R': 2, 'Equity': 3})
        self.MASTER_OUTPUT['loan_sub_type_rank'] = \
            self.MASTER_OUTPUT['loan_sub_type'].astype(str).replace('nan', '') + \
            self.MASTER_OUTPUT['committed'].astype(str).replace('nan', '')
        self.MASTER_OUTPUT['loan_sub_type_rank'].replace({'T.*': 10, 'RC': 20, 'RU': 21, 'Equity': 30, '': 99},
                                                         regex=True, inplace=True)
        self.MASTER_OUTPUT.sort_values(
            by=[
                'stage', 'output_type_num', 'solo_jv_rank', 'project_id',
                'loan_sub_type_rank', 'loan_id', 'loan_facility_id'
            ], inplace=True)
        self.MASTER_OUTPUT.reset_index(drop=True, inplace=True)

        return
