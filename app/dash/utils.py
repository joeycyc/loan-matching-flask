import re
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse as dateparse
import itertools


def std_date(input_date):
    """Standardize date into datetime.date data type
    Args:
    - input_date: any type, date info
    """
    try:
        input_date = str(input_date)
        if re.match('\d{1,2}/\d{1,2}/\d{4}', input_date):
            return dateparse(input_date, dayfirst=True).date()
        else:
            return dateparse(input_date).date()
    except:
        print(f'std_date: error with {str(input_date)}')
        return input_date


def offset_date(input_date: dt.date, year: int = 0, month: int = 0, day: int = 0) -> dt.date:
    """Offset the input date by no. of year/ month/ day
    Args:
    - input_date: datetime.date, input date
    - year/ month/ day: int, negative means offset to the earlier date
    """
    out_year, out_month = input_date.year, input_date.month
    year, month, day = int(year), int(month), int(day)
    # Change year
    out_year += year
    # Change month
    if out_month + month < 1:
        out_year -= 1
        out_month += month + 12
    elif out_month + month > 12:
        out_year += 1
        out_month += month - 12
    else:
        out_month += month
    # Final output with day changed
    return dt.date(out_year, out_month, 1) + dt.timedelta(days=(input_date.day + day - 1))


def date2idx(input_date:dt.date, day_zero:dt.date) -> int:
    """Convert date to index"""
    if isinstance(input_date, dt.date) and isinstance(day_zero, dt.date):
        return (input_date - day_zero).days
    else:
        print('date2idx: wrong data type')
        return np.nan


def idx2date(input_idx:int, day_zero:dt.date) -> dt.date:
    """Convert index to date"""
    if isinstance(input_idx, int) and isinstance(day_zero, dt.date):
        return day_zero + dt.timedelta(days=input_idx)
    else:
        print('idx2date: wrong data type')
        return None


def vec2rects(vec, preserve_zero=False, preserve_zero_st_idx=0, preserve_zero_end_idx=0):
    """Convert a 1D vector into 'rectangles', which is a list of lists,
        with each child list corresponds 1 'rectangle',
        each child list has 3 elements:
            1. start date index, int starting from zero
            2. end date index, int starting from zero
            3. amount of money in HK$B, float"""
    if preserve_zero and (preserve_zero_st_idx > preserve_zero_end_idx):
        print('WARNING: vec2rects: preserve_zero_st_idx should be smaller or equal to preserve_zero_end_idx')
        preserve_zero = False  # Not to include preserve_zero_st_idx and preserve_zero_end_idx for further calculation.

    rects = []
    wk_start_d, wk_end_d, wk_amt = 0, 0, 0.0

    for i, amt in enumerate(vec):
        if amt == wk_amt:
            wk_end_d = i
        else:  # amt != wk_amt
            if wk_amt != 0:  # Must handle if wk_amt != 0
                rects.append([wk_start_d, wk_end_d, wk_amt])
            else:  # wk_amt == 0
                if preserve_zero and (max(wk_start_d, preserve_zero_st_idx) <= min(wk_end_d, preserve_zero_end_idx)):
                    rects.append([max(wk_start_d, preserve_zero_st_idx), min(wk_end_d, preserve_zero_end_idx), wk_amt])
                else:
                    # either we don't preserve_zero or the preserve_zero range does not overlap with working range
                    pass
            # Start another sub-list
            wk_start_d, wk_end_d, wk_amt = i, i, amt

    # Final element
    if wk_amt != 0:  # Must handle if wk_amt != 0
        rects.append([wk_start_d, wk_end_d, wk_amt])
    else:  # wk_amt == 0
        if preserve_zero and (max(wk_start_d, preserve_zero_st_idx) <= min(wk_end_d, preserve_zero_end_idx)):
            rects.append([max(wk_start_d, preserve_zero_st_idx), min(wk_end_d, preserve_zero_end_idx), wk_amt])
        else:
            # either we don't preserve_zero or the preserve_zero range does not overlap with working range
            pass

    return rects


def vec2rects_deprecated(vec):
    """Convert a 1D vector into 'rectangles', which is a list of lists,
        with each child list corresponds 1 'rectangle',
        each child list has 3 elements:
            1. start date index, int starting from zero
            2. end date index, int starting from zero
            3. amount of money in HK$B, float"""
    rects = []
    wk_start_d, wk_end_d, wk_amt = 0, 0, 0.0
    for i, amt in enumerate(vec):
        if amt == wk_amt:
            wk_end_d = i
        else:  # amt != wk_amt
            # Only handle when wk_amt != 0
            if wk_amt != 0: rects.append([wk_start_d, wk_end_d, wk_amt])
            wk_start_d, wk_end_d, wk_amt = i, i, amt  # Start another sub-list
    # Final element
    if wk_amt != 0:
        rects.append([wk_start_d, wk_end_d, wk_amt])
    return rects


def get_date_tick_lists(start, end, short_form=False):
    """Return tick texts and tick values within the date range from start to end
    Args:
        - start: datetime.date
        - end: datetime.date
        - short_form: bool, tick text format is 'yyQq' if True, 'yyyy Qq' if False

    """
    # Generate full list
    ymd_list = [[y, md[0], md[1]]
                for y, md in itertools.product(range(start.year, end.year + 1),
                                               [(1, 1), (2, 15), (4, 1), (5, 15),
                                                (7, 1), (8, 15), (10, 1), (11, 15)])] + [[end.year, 1, 1]]
    tick_texts, tick_values = [], []
    for y, m, d in ymd_list:
        if m in (2, 5, 8, 11):
            if short_form:
                tick_texts.append(str(y)[-2:] + 'Q' + str(m//3+1))
            else:
                tick_texts.append(str(y) + ' Q' + str(m//3+1))
        else:
            tick_texts.append('')
        tick_values.append(dt.datetime.strftime(dt.date(y, m, d), '%Y-%m-%d'))

    # Trim
    start_val = start.year * 10000 + start.month * 100 + start.day
    end_val = end.year * 10000 + end.month * 100 + end.day
    start_i = [i for i, val in enumerate([y * 10000 + m * 100 + d for y, m, d in ymd_list]) if val < start_val][-1]
    end_i = [i for i, val in enumerate([y * 10000 + m * 100 + d for y, m, d in ymd_list]) if val > end_val][0]
    # Make sure the cut-off is at Jan/ Apr/ Jul/ Oct
    if ymd_list[start_i][1] not in (1, 4, 7, 10):
        start_i -= 1
    if ymd_list[end_i][1] not in (1, 4, 7, 10):
        end_i += 1
    tick_texts = tick_texts[start_i: end_i+1]
    tick_values = tick_values[start_i: end_i + 1]

    return tick_texts, tick_values
