# Script: pull demographic data for US cities from the Census Bureau API
# Usage: set environment variable CENSUS_API_KEY (optional)

import os
import time
import requests
import pandas as pd
from typing import List, Tuple, Optional

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", None)
ACS_YEAR = "2021"  # change to desired ACS 5-year release ("2022")

# minimal state abbrev -> fips mapping (all states + DC)
STATE_FIPS = {
    "AL": "01","AK": "02","AZ": "04","AR": "05","CA": "06","CO": "08","CT": "09","DE": "10",
    "DC": "11","FL": "12","GA": "13","HI": "15","ID": "16","IL": "17","IN": "18","IA": "19",
    "KS": "20","KY": "21","LA": "22","ME": "23","MD": "24","MA": "25","MI": "26","MN": "27",
    "MS": "28","MO": "29","MT": "30","NE": "31","NV": "32","NH": "33","NJ": "34","NM": "35",
    "NY": "36","NC": "37","ND": "38","OH": "39","OK": "40","OR": "41","PA": "42","RI": "44",
    "SC": "45","SD": "46","TN": "47","TX": "48","UT": "49","VT": "50","VA": "51","WA": "53",
    "WV": "54","WI": "55","WY": "56"
}

# ACS variables to fetch (5-year)
ACS_VARS = {
    "total_population": "B01003_001E",
    "median_age": "B01002_001E",
    "median_household_income": "B19013_001E",
    "white": "B02001_002E",
    "black": "B02001_003E",
    "asian": "B02001_005E",
    "hispanic": "B03003_003E",
    "gender_male": "B01001_002E",
    "gender_female": "B01001_026E",
    # poverty (below poverty level)
    "poverty_count": "B17001_002E",
    # foreign born (common variable from B05002 family; if not available will be ignored)
    "foreign_born": "B05002_013E",
    # unemployment (unemployed persons; labor force variable will be used to compute pct if available)
    "unemployed": "B23025_005E",
}
# B01001 detailed sex-by-age variables (we will request the whole block and compute age bins)
# These are the standard B01001 variables from _002E .. _049E (male and female age bins).
# We'll request them dynamically below (AGE_VARS list).
AGE_VARS = [f"B01001_{i:03d}E" for i in range(2, 50)]  # B01001_002E .. B01001_049E

# Educational attainment: Bachelor's degree and higher are commonly B15003_022E..B15003_025E
EDUCATION_BA_OR_HIGHER = [f"B15003_{i:03d}E" for i in range(22, 26)]  # 22-25

def _call_census_api(url: str, params: dict) -> dict:
    if CENSUS_API_KEY:
        params = dict(params)
        params["key"] = CENSUS_API_KEY
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def geocode_place(city: str, state_abbrev: str) -> Optional[Tuple[str,str]]:
    """
    Find the place code (PLACE) for a city by querying the ACS list of places for the state.
    Returns (place_code, state_fips) or None if not found.
    This avoids the street-level geocoder which can return 400 for address-less queries.
    """
    state_abbrev = state_abbrev.strip().upper()
    state_fips = STATE_FIPS.get(state_abbrev)
    if not state_fips:
        return None

    # Query ACS for all places in the state and filter by NAME
    url = f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
    params = {
        "get": "NAME,PLACE",
        "for": "place:*",
        "in": f"state:{state_fips}"
    }
    try:
        raw = _call_census_api(url, params)
    except Exception:
        return None

    if not raw or len(raw) < 2:
        return None

    header = raw[0]
    rows = raw[1:]
    df_places = pd.DataFrame(rows, columns=header)

    # NAME values look like "Austin city, Texas" or "Anchorage municipality, Alaska"
    city_lc = city.strip().lower()

    # prefer names that start with the city token (handles "Austin city, Texas")
    mask_starts = df_places["NAME"].str.lower().str.startswith(city_lc)
    candidates = df_places[mask_starts]

    # fallback to contains if no startswith match
    if candidates.empty:
        mask_contains = df_places["NAME"].str.lower().str.contains(city_lc)
        candidates = df_places[mask_contains]

    if candidates.empty:
        return None

    # pick best candidate (first). PLACE column is the place code.
    place_code = candidates.iloc[0].get("PLACE") or candidates.iloc[0].get("place")
    if place_code is None:
        return None

    return place_code, state_fips

def _chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _fetch_vars_in_chunks(year: str, vars_list: list, place_fips: str, state_fips: str, chunk_size: int = 40):
    """
    Request variables in chunks and merge the returned rows into a single dict.
    Returns dict of variable -> value for the place.
    """
    combined = {}
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    for chunk in _chunk_list(vars_list, chunk_size):
        params = {
            "get": ",".join(chunk + ["NAME"]),
            "for": f"place:{place_fips}",
            "in": f"state:{state_fips}"
        }
        raw = _call_census_api(url, params)
        if not raw or len(raw) < 2:
            raise RuntimeError(f"No data returned for vars chunk (len={len(chunk)}).")
        header = raw[0]
        values = raw[1]
        part = dict(zip(header, values))
        # merge part into combined (preserve NAME if already present)
        for k, v in part.items():
            if k == "NAME" and "NAME" in combined:
                continue
            combined[k] = v
    return combined

def fetch_city_demographics(city: str, state_abbrev: str, year: str = ACS_YEAR) -> dict:
    """
    Return a dict with requested ACS variables (raw values and simple derived percentages).
    Uses chunked requests to avoid parameter length limits.
    """
    geocode = geocode_place(city, state_abbrev)
    if not geocode:
        raise ValueError(f"Could not geocode place: {city}, {state_abbrev}")
    place_fips, state_fips = geocode

    # include base vars + age block + education block + civilian labor force (for unemployment pct)
    extra_vars = AGE_VARS + EDUCATION_BA_OR_HIGHER + ["B23025_003E"]  # B23025_003E = civilian labor force
    vars_list = list(set(list(ACS_VARS.values()) + extra_vars))

    # fetch in chunks to avoid URL/parameter length limits
    try:
        combined_row = _fetch_vars_in_chunks(year, vars_list, place_fips, state_fips, chunk_size=35)
    except Exception as e:
        raise RuntimeError(f"Census API chunked request failed for {city}, {state_abbrev}: {e}")

    # use combined_row as the row mapping
    row = combined_row

    # map to friendly names
    out = {"city": city, "state": state_abbrev, "name": row.get("NAME")}

    # fields that should remain floats (not coerced to int)
    FLOAT_VARS = {"median_age"}

    # populate simple ACS_VARS
    for friendly, var in ACS_VARS.items():
        val = row.get(var)
        if val is None or val == "":
            out[friendly] = None
            continue
        try:
            num = float(val)
        except (ValueError, TypeError):
            out[friendly] = None
            continue
        out[friendly] = round(num, 2) if friendly in FLOAT_VARS else int(round(num))

    # compute age-group counts using the B01001 block we requested
    # IMPORTANT: Use ONLY male bins (002-025) to avoid double-counting
    # B01001_002E-B01001_025E are male age groups
    # B01001_026E-B01001_049E are female age groups (we'll ignore these)
    
    # Male age bins (avoiding double-counting):
    # Under 18: Male bins 3-6 (under 5, 5-9, 10-14, 15-17)
    under18_codes = [f"B01001_{i:03d}E" for i in range(3, 7)]  # 003-006
    
    # 18-64: Male bins 7-19 (18-19, 20, 21, 22-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-61, 62-64)
    age18_64_codes = [f"B01001_{i:03d}E" for i in range(7, 20)]  # 007-019
    
    # 65+: Male bins 20-25 (65-66, 67-69, 70-74, 75-79, 80-84, 85+)
    age65_plus_codes = [f"B01001_{i:03d}E" for i in range(20, 26)]  # 020-025

    def safe_sum(codes):
        s = 0.0
        found = False
        for code in codes:
            v = row.get(code)
            if v in (None, ""):
                continue
            try:
                s += float(v)
                found = True
            except Exception:
                continue
        return int(round(s)) if found else None

    # Get male counts
    male_under18 = safe_sum(under18_codes)
    male_18_64 = safe_sum(age18_64_codes)
    male_65_plus = safe_sum(age65_plus_codes)
    
    # Get female counts (same structure but starting at 026)
    # Female under 18: bins 27-30
    female_under18_codes = [f"B01001_{i:03d}E" for i in range(27, 31)]
    # Female 18-64: bins 31-43
    female_18_64_codes = [f"B01001_{i:03d}E" for i in range(31, 44)]
    # Female 65+: bins 44-49
    female_65_plus_codes = [f"B01001_{i:03d}E" for i in range(44, 50)]
    
    female_under18 = safe_sum(female_under18_codes)
    female_18_64 = safe_sum(female_18_64_codes)
    female_65_plus = safe_sum(female_65_plus_codes)
    
    # Combine male + female for each age group
    out["age_under_18"] = (male_under18 + female_under18) if (male_under18 is not None and female_under18 is not None) else None
    out["age_18_to_64"] = (male_18_64 + female_18_64) if (male_18_64 is not None and female_18_64 is not None) else None
    out["age_65_plus"] = (male_65_plus + female_65_plus) if (male_65_plus is not None and female_65_plus is not None) else None

    # education: BA or higher (sum EDUCATION_BA_OR_HIGHER)
    ba_count = 0.0
    ba_found = False
    for code in EDUCATION_BA_OR_HIGHER:
        v = row.get(code)
        if v in (None, ""):
            continue
        try:
            ba_count += float(v)
            ba_found = True
        except Exception:
            continue
    out["ba_or_higher_count"] = int(round(ba_count)) if ba_found else None

    # derive percentages relative to total population when possible
    tp = out.get("total_population")
    if tp and tp > 0:
        for k in ["age_under_18", "age_18_to_64", "age_65_plus"]:
            if out.get(k) is not None:
                out[f"{k}_pct"] = round(out[k] / tp * 100, 2)
            else:
                out[f"{k}_pct"] = None
        out["ba_or_higher_pct"] = round(out["ba_or_higher_count"] / tp * 100, 2) if out.get("ba_or_higher_count") is not None else None
        out["poverty_pct"] = round(out["poverty_count"] / tp * 100, 2) if out.get("poverty_count") is not None else None
        out["foreign_born_pct"] = round(out["foreign_born"] / tp * 100, 2) if out.get("foreign_born") is not None else None
    else:
        out["age_under_18_pct"] = out["age_18_to_64_pct"] = out["age_65_plus_pct"] = None
        out["ba_or_higher_pct"] = out["poverty_pct"] = out["foreign_born_pct"] = None

    # unemployment pct: use unemployed and civilian labor force (B23025_003E)
    civ_lf = row.get("B23025_003E")
    try:
        civ_lf_val = int(round(float(civ_lf))) if civ_lf not in (None, "") else None
    except Exception:
        civ_lf_val = None
    unemployed_val = out.get("unemployed")
    out["unemployment_pct"] = round(unemployed_val / civ_lf_val * 100, 2) if (unemployed_val is not None and civ_lf_val and civ_lf_val > 0) else None

    # compute race percentages relative to total population if available
    if tp and tp > 0:
        for r in ("white", "black", "asian", "hispanic"):
            v = out.get(r)
            out[f"{r}_pct"] = round((v / tp) * 100, 2) if v is not None else None
    else:
        for r in ("white", "black", "asian", "hispanic"):
            out[f"{r}_pct"] = None

    return out


def fetch_multiple_cities(cities: List[Tuple[str,str]], year: str = ACS_YEAR, delay: float = 0.5) -> pd.DataFrame:
    """
    cities: list of (city_name, state_abbrev) e.g. [("Austin","TX"), ("Denver","CO")]
    returns pandas DataFrame
    """
    rows = []
    for city, state in cities:
        try:
            rows.append(fetch_city_demographics(city.strip(), state.strip().upper(), year=year))
        except Exception as e:
            rows.append({"city": city, "state": state, "error": str(e)})
        time.sleep(delay)  # be polite to API
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Example: replace/add any cities you need
    city_list = [("Atlanta", "GA"), ("Austin", "TX"),("Chicago", "IL"),("Dallas", "TX"),("Denver", "CO"),("Detroit", "MI"),("Honolulu", "HI"),("Houston", "TX"),("Los Angeles", "CA"),("Miami", "FL"),("Philadelphia", "PA"),("San Diego", "CA"), ("San Francisco", "CA"), ("New York", "NY"), ("Seattle", "WA"), ("Washington", "DC")]
    df = fetch_multiple_cities(city_list, year=ACS_YEAR)
    out_path = os.path.expanduser("/Users/albinmeli/CS5870/ds_final_project/city_demographics.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")