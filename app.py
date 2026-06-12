# ============================================================
# Imports
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
# This dashboard uses a mock census dataset to support evidence-based local policy planning through data cleaning, 
# feature engineering, exploratory analysis and transparent decision scoring.

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Census Policy Planning Dashboard",
    layout="wide"
)

st.title("Census Policy Planning Dashboard")

st.markdown(
    """
    An interactive data science dashboard for analysing mock census data and supporting
    evidence-based local infrastructure planning.
    """
)

st.markdown(
    """
    **Final recommendation:** Build a **train station** and prioritise
    **employment and training investment**.
    """
)

with st.expander("Project context"):
    st.write(
        """
        This project was developed from a census-style policy planning analysis.
        The dataset contains anonymised mock census records, including demographic,
        household, occupation, religion and infirmity-related variables.

        The aim is to demonstrate how data cleaning, feature engineering,
        exploratory analysis and transparent scoring can support local government
        planning decisions.
        """
    )

with st.expander("Methodology summary"):
    st.markdown(
        """
        The dashboard follows five main steps:

        1. Load the raw census dataset.
        2. Clean missing, inconsistent and incorrectly typed values.
        3. Engineer policy-relevant features such as household size, age group,
           commuter proxy, working-age status and unemployment indicators.
        4. Explore demographic, transport and employment patterns.
        5. Compare policy options using an explainable decision matrix.
        """
    )

# Project paths
PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mock_census.csv"


# ============================================================
# Helper functions
# ============================================================

@st.cache_data
def load_data(path):
    """Load the raw census dataset."""
    return pd.read_csv(path)

def clean_census_data(df):
    """Clean the raw census dataset for dashboard analysis."""
    cleaned = df.copy()

    # Standardise column names
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )

    # Strip whitespace from text columns and convert blank strings to missing values
    text_cols = cleaned.select_dtypes(include="object").columns

    for col in text_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA
        })

    # Rename long relationship column
    cleaned = cleaned.rename(
        columns={"relationship_to_head_of_house": "relationship_to_head"}
    )

    # Convert age to numeric
    cleaned["age_raw"] = cleaned["age"]
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")

    # Flag invalid or missing age values
    cleaned["age_imputed_flag"] = cleaned["age"].isna() | (cleaned["age"] < 0) | (cleaned["age"] > 115)
    cleaned.loc[cleaned["age_imputed_flag"], "age"] = pd.NA

    # Impute age using median age
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median()).round().astype(int)

    # Standardise gender
    cleaned["gender"] = cleaned["gender"].replace({
        "Female": "Female",
        "female": "Female",
        "F": "Female",
        "f": "Female",
        "Male": "Male",
        "male": "Male",
        "M": "Male",
        "m": "Male"
    })

    # Standardise marital status
    cleaned["marital_status"] = cleaned["marital_status"].replace({
        "S": "Single",
        "M": "Married",
        "D": "Divorced",
        "W": "Widowed"
    })

    # Missing marital status: minors are N/A, adults are Unknown
    cleaned.loc[
        (cleaned["age"] < 16) & (cleaned["marital_status"].isna()),
        "marital_status"
    ] = "N/A"

    cleaned["marital_status"] = cleaned["marital_status"].fillna("Unknown")

    # Missing personal identifiers are not fabricated
    cleaned["first_name"] = cleaned["first_name"].fillna("Unknown")
    cleaned["surname"] = cleaned["surname"].fillna("Unknown")

    # Missing relationship is unknown
    cleaned["relationship_to_head"] = cleaned["relationship_to_head"].fillna("Unknown")

    # Missing religion is not imputed
    cleaned["religion"] = cleaned["religion"].fillna("Not stated")

    # Missing infirmity means no infirmity recorded in the census field
    cleaned["infirmity"] = cleaned["infirmity"].fillna("None recorded")

    return cleaned

def add_census_features(cleaned):
    """Add analytical features for policy planning analysis."""
    cleaned = cleaned.copy()

    # Household identifier
    cleaned["household_id"] = (
        cleaned["house_number"].astype(str).str.strip()
        + "_"
        + cleaned["street"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Age groups
    age_bins = [0, 4, 10, 15, 18, 24, 34, 44, 54, 64, 74, 84, float("inf")]
    age_labels = [
        "0-4", "5-10", "11-15", "16-18", "19-24", "25-34",
        "35-44", "45-54", "55-64", "65-74", "75-84", "85+"
    ]

    cleaned["age_group"] = pd.cut(
        cleaned["age"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True
    )

    # Life-stage indicators
    cleaned["is_child"] = cleaned["age"] < 16
    cleaned["is_school_age"] = cleaned["age"].between(5, 18, inclusive="both")
    cleaned["is_working_age"] = cleaned["age"].between(16, 64, inclusive="both")
    cleaned["is_retirement_age"] = cleaned["age"] >= 65
    cleaned["is_older_elderly"] = cleaned["age"] >= 80
    cleaned["is_under_5"] = cleaned["age"] <= 4
    cleaned["is_age_0_to_1"] = cleaned["age"] <= 1

    # Occupation-based indicators
    occupation = cleaned["occupation"].fillna("").str.lower()

    cleaned["is_unemployed"] = occupation.str.contains("unemployed", case=False, na=False)
    cleaned["is_student"] = occupation.str.contains("student", case=False, na=False)
    cleaned["is_university_student"] = occupation.str.contains("university student", case=False, na=False)

    # Commuter proxy: indirect indicator, not direct journey-to-work evidence
    commuter_keywords = [
        "university student", "consultant", "engineer", "developer", "analyst",
        "manager", "lecturer", "research", "scientist", "accountant",
        "solicitor", "broker", "architect"
    ]

    cleaned["is_commuter_proxy"] = occupation.map(
        lambda value: any(keyword in value for keyword in commuter_keywords)
    )

    # Professional/affluence proxy: indirect indicator only, as income is unavailable
    professional_keywords = [
        "manager", "director", "engineer", "doctor", "solicitor", "lawyer",
        "accountant", "consultant", "architect", "scientist", "analyst",
        "developer", "surgeon", "pharmacist", "lecturer", "research",
        "broker", "executive"
    ]

    cleaned["is_professional_proxy"] = occupation.map(
        lambda value: any(keyword in value for keyword in professional_keywords)
    )

    # Household size and occupancy bands
    cleaned["household_size"] = cleaned.groupby("household_id")["household_id"].transform("size")

    cleaned["occupancy_band"] = pd.cut(
        cleaned["household_size"],
        bins=[0, 1, 2, 4, float("inf")],
        labels=[
            "Single-person",
            "Two-person",
            "Medium household",
            "Large household"
        ]
    )

    return cleaned

def clamp01(value):
    """Restrict a numeric score to the 0–1 range."""
    return max(0, min(1, value))

def build_decision_matrices(df):
    """Build explainable land-use and investment decision matrices."""
    total_population = len(df)
    working_age_count = df["is_working_age"].sum()

    commuter_share = df["is_commuter_proxy"].mean()
    university_student_share = df["is_university_student"].mean()

    unemployment_rate = (
        (df["is_unemployed"] & df["is_working_age"]).sum() / working_age_count
        if working_age_count
        else 0
    )

    retirement_share = df["is_retirement_age"].mean()
    older_elderly_share = df["is_older_elderly"].mean()
    school_age_share = df["is_school_age"].mean()
    under5_share = df["is_under_5"].mean()
    professional_share = df["is_professional_proxy"].mean()
    recorded_infirmity_share = (~df["infirmity"].eq("None recorded")).mean()

    household_level = df.drop_duplicates("household_id")
    large_household_share = household_level["occupancy_band"].eq("Large household").mean()

    # Religion evidence is population-weighted and confidence-adjusted.
    religion_known = df[df["religion"] != "Not stated"]
    stated_religion_share = len(religion_known) / total_population if total_population else 0
    not_stated_religion_share = 1 - stated_religion_share

    top_non_catholic = "None"
    top_non_catholic_total_share = 0

    if not religion_known.empty:
        non_catholic = religion_known[
            ~religion_known["religion"].str.lower().eq("catholic")
        ]

        if not non_catholic.empty:
            top_non_catholic = non_catholic["religion"].value_counts().idxmax()
            top_non_catholic_count = non_catholic["religion"].value_counts().iloc[0]
            top_non_catholic_total_share = top_non_catholic_count / total_population

    religious_building_score = clamp01(
        (top_non_catholic_total_share / 0.30) * 0.70
        + (stated_religion_share / 0.70) * 0.30
    )

    # Conservative cap: if more than half did not state religion,
    # religious-building demand should not dominate the land-use recommendation.
    if not_stated_religion_share > 0.50:
        religious_building_score = min(religious_building_score, 0.55)

    land_use = pd.DataFrame([
        {
            "Option": "Train station",
            "Score": round(
                clamp01(
                    (commuter_share / 0.25) * 0.70
                    + (university_student_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Commuter proxy share: {commuter_share:.1%}; "
                f"university student share: {university_student_share:.1%}"
            ),
            "Rationale": "High commuter proxy and confirmed university-student commuting pressure",
            "Evidence limitation": "No direct journey-to-work, rail-demand or traffic-volume data",
        },
        {
            "Option": "High-density housing",
            "Score": round(
                clamp01(
                    (large_household_share / 0.30) * 0.50
                    + ((under5_share + school_age_share) / 0.35) * 0.50
                ),
                3
            ),
            "Main evidence": (
                f"Large household share: {large_household_share:.1%}; "
                f"under-19 share: {(under5_share + school_age_share):.1%}"
            ),
            "Rationale": "Large households and younger population imply possible housing pressure",
            "Evidence limitation": "No rent, house-price, overcrowding, waiting-list or population-growth data",
        },
        {
            "Option": "Low-density housing",
            "Score": round(
                clamp01(
                    (professional_share / 0.25) * 0.60
                    + (large_household_share / 0.25) * 0.40
                ),
                3
            ),
            "Main evidence": (
                f"Professional proxy share: {professional_share:.1%}; "
                f"large household share: {large_household_share:.1%}"
            ),
            "Rationale": "Professional occupation proxy and larger households may support family housing",
            "Evidence limitation": "Income, property values and affordability are unavailable",
        },
        {
            "Option": "Religious building",
            "Score": round(religious_building_score, 3),
            "Main evidence": (
                f"{top_non_catholic} share of total population: "
                f"{top_non_catholic_total_share:.1%}; "
                f"religion not stated: {not_stated_religion_share:.1%}"
            ),
            "Rationale": f"Largest recorded non-Catholic group: {top_non_catholic}",
            "Evidence limitation": (
                "More than half of residents did not state a religion; no worship-capacity "
                "or unmet-demand data is available"
            ),
        },
        {
            "Option": "Emergency medical building",
            "Score": round(
                clamp01(
                    (recorded_infirmity_share / 0.08) * 0.35
                    + (retirement_share / 0.20) * 0.35
                    + (under5_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Recorded infirmity: {recorded_infirmity_share:.1%}; "
                f"65+: {retirement_share:.1%}; under-5: {under5_share:.1%}"
            ),
            "Rationale": "Older residents, infirmity records and young-child population may create medical demand",
            "Evidence limitation": "No hospital admissions, GP waiting-time, accident or pregnancy data",
        },
    ]).sort_values("Score", ascending=False)

    investment = pd.DataFrame([
        {
            "Option": "Employment and training",
            "Score": round(clamp01(unemployment_rate / 0.10), 3),
            "Main evidence": f"Working-age unemployment rate: {unemployment_rate:.1%}",
            "Rationale": "Working-age unemployment creates direct need for skills and retraining",
            "Evidence limitation": "Occupation data does not show unemployment duration or specific skills gaps",
        },
        {
            "Option": "Old age care",
            "Score": round(
                clamp01(
                    (retirement_share / 0.20) * 0.70
                    + (older_elderly_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"65+ share: {retirement_share:.1%}; "
                f"80+ share: {older_elderly_share:.1%}"
            ),
            "Rationale": "Older population share indicates future care demand",
            "Evidence limitation": "No direct care-needs assessment or mortality history",
        },
        {
            "Option": "Increase spending for schooling",
            "Score": round(
                clamp01(
                    (school_age_share / 0.22) * 0.65
                    + (under5_share / 0.08) * 0.35
                ),
                3
            ),
            "Main evidence": (
                f"School-age share: {school_age_share:.1%}; "
                f"under-5 share: {under5_share:.1%}"
            ),
            "Rationale": "School-age and under-five population indicate schooling demand",
            "Evidence limitation": "No school capacity, class-size or catchment-area data",
        },
        {
            "Option": "General infrastructure",
            "Score": round(
                clamp01(
                    (commuter_share / 0.25) * 0.35
                    + (large_household_share / 0.30) * 0.35
                    + ((school_age_share + under5_share) / 0.35) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Commuter share: {commuter_share:.1%}; "
                f"large households: {large_household_share:.1%}; "
                f"under-19: {(school_age_share + under5_share):.1%}"
            ),
            "Rationale": "Commuting, household pressure and younger residents imply service demand",
            "Evidence limitation": "No historical service-capacity or maintenance-cost data",
        },
    ]).sort_values("Score", ascending=False)

    return land_use, investment

def build_decision_matrices(df):
    """Build explainable land-use and investment decision matrices."""
    total_population = len(df)
    working_age_count = df["is_working_age"].sum()

    commuter_share = df["is_commuter_proxy"].mean()
    university_student_share = df["is_university_student"].mean()

    unemployment_rate = (
        (df["is_unemployed"] & df["is_working_age"]).sum() / working_age_count
        if working_age_count
        else 0
    )

    retirement_share = df["is_retirement_age"].mean()
    older_elderly_share = df["is_older_elderly"].mean()
    school_age_share = df["is_school_age"].mean()
    under5_share = df["is_under_5"].mean()
    professional_share = df["is_professional_proxy"].mean()
    recorded_infirmity_share = (~df["infirmity"].eq("None recorded")).mean()

    household_level = df.drop_duplicates("household_id")
    large_household_share = household_level["occupancy_band"].eq("Large household").mean()

    # Religion evidence is population-weighted and confidence-adjusted.
    religion_known = df[df["religion"] != "Not stated"]
    stated_religion_share = len(religion_known) / total_population if total_population else 0
    not_stated_religion_share = 1 - stated_religion_share

    top_non_catholic = "None"
    top_non_catholic_total_share = 0

    if not religion_known.empty:
        non_catholic = religion_known[
            ~religion_known["religion"].str.lower().eq("catholic")
        ]

        if not non_catholic.empty:
            top_non_catholic = non_catholic["religion"].value_counts().idxmax()
            top_non_catholic_count = non_catholic["religion"].value_counts().iloc[0]
            top_non_catholic_total_share = top_non_catholic_count / total_population

    religious_building_score = clamp01(
        (top_non_catholic_total_share / 0.30) * 0.70
        + (stated_religion_share / 0.70) * 0.30
    )

    # Conservative cap: if more than half did not state religion,
    # religious-building demand should not dominate the land-use recommendation.
    if not_stated_religion_share > 0.50:
        religious_building_score = min(religious_building_score, 0.55)

    land_use = pd.DataFrame([
        {
            "Option": "Train station",
            "Score": round(
                clamp01(
                    (commuter_share / 0.25) * 0.70
                    + (university_student_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Commuter proxy share: {commuter_share:.1%}; "
                f"university student share: {university_student_share:.1%}"
            ),
            "Rationale": "High commuter proxy and confirmed university-student commuting pressure",
            "Evidence limitation": "No direct journey-to-work, rail-demand or traffic-volume data",
        },
        {
            "Option": "High-density housing",
            "Score": round(
                clamp01(
                    (large_household_share / 0.30) * 0.50
                    + ((under5_share + school_age_share) / 0.35) * 0.50
                ),
                3
            ),
            "Main evidence": (
                f"Large household share: {large_household_share:.1%}; "
                f"under-19 share: {(under5_share + school_age_share):.1%}"
            ),
            "Rationale": "Large households and younger population imply possible housing pressure",
            "Evidence limitation": "No rent, house-price, overcrowding, waiting-list or population-growth data",
        },
        {
            "Option": "Low-density housing",
            "Score": round(
                clamp01(
                    (professional_share / 0.25) * 0.60
                    + (large_household_share / 0.25) * 0.40
                ),
                3
            ),
            "Main evidence": (
                f"Professional proxy share: {professional_share:.1%}; "
                f"large household share: {large_household_share:.1%}"
            ),
            "Rationale": "Professional occupation proxy and larger households may support family housing",
            "Evidence limitation": "Income, property values and affordability are unavailable",
        },
        {
            "Option": "Religious building",
            "Score": round(religious_building_score, 3),
            "Main evidence": (
                f"{top_non_catholic} share of total population: "
                f"{top_non_catholic_total_share:.1%}; "
                f"religion not stated: {not_stated_religion_share:.1%}"
            ),
            "Rationale": f"Largest recorded non-Catholic group: {top_non_catholic}",
            "Evidence limitation": (
                "More than half of residents did not state a religion; no worship-capacity "
                "or unmet-demand data is available"
            ),
        },
        {
            "Option": "Emergency medical building",
            "Score": round(
                clamp01(
                    (recorded_infirmity_share / 0.08) * 0.35
                    + (retirement_share / 0.20) * 0.35
                    + (under5_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Recorded infirmity: {recorded_infirmity_share:.1%}; "
                f"65+: {retirement_share:.1%}; under-5: {under5_share:.1%}"
            ),
            "Rationale": "Older residents, infirmity records and young-child population may create medical demand",
            "Evidence limitation": "No hospital admissions, GP waiting-time, accident or pregnancy data",
        },
    ]).sort_values("Score", ascending=False)

    investment = pd.DataFrame([
        {
            "Option": "Employment and training",
            "Score": round(clamp01(unemployment_rate / 0.10), 3),
            "Main evidence": f"Working-age unemployment rate: {unemployment_rate:.1%}",
            "Rationale": "Working-age unemployment creates direct need for skills and retraining",
            "Evidence limitation": "Occupation data does not show unemployment duration or specific skills gaps",
        },
        {
            "Option": "Old age care",
            "Score": round(
                clamp01(
                    (retirement_share / 0.20) * 0.70
                    + (older_elderly_share / 0.08) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"65+ share: {retirement_share:.1%}; "
                f"80+ share: {older_elderly_share:.1%}"
            ),
            "Rationale": "Older population share indicates future care demand",
            "Evidence limitation": "No direct care-needs assessment or mortality history",
        },
        {
            "Option": "Increase spending for schooling",
            "Score": round(
                clamp01(
                    (school_age_share / 0.22) * 0.65
                    + (under5_share / 0.08) * 0.35
                ),
                3
            ),
            "Main evidence": (
                f"School-age share: {school_age_share:.1%}; "
                f"under-5 share: {under5_share:.1%}"
            ),
            "Rationale": "School-age and under-five population indicate schooling demand",
            "Evidence limitation": "No school capacity, class-size or catchment-area data",
        },
        {
            "Option": "General infrastructure",
            "Score": round(
                clamp01(
                    (commuter_share / 0.25) * 0.35
                    + (large_household_share / 0.30) * 0.35
                    + ((school_age_share + under5_share) / 0.35) * 0.30
                ),
                3
            ),
            "Main evidence": (
                f"Commuter share: {commuter_share:.1%}; "
                f"large households: {large_household_share:.1%}; "
                f"under-19: {(school_age_share + under5_share):.1%}"
            ),
            "Rationale": "Commuting, household pressure and younger residents imply service demand",
            "Evidence limitation": "No historical service-capacity or maintenance-cost data",
        },
    ]).sort_values("Score", ascending=False)

    return land_use, investment

@st.cache_data
def convert_df_to_csv(df):
    """Convert a dataframe to CSV for download."""
    return df.to_csv(index=False).encode("utf-8")

def add_footer():
    """Add a simple dashboard footer."""
    st.divider()
    st.caption(
        """
        Census Policy Planning Dashboard | Built with Python, Pandas, Plotly and Streamlit.
        This project uses mock census data for portfolio and policy-analysis demonstration purposes.
        """
    )

if DATA_PATH.exists():
    raw_df = load_data(DATA_PATH)
    cleaned_df = clean_census_data(raw_df)
    cleaned_df = add_census_features(cleaned_df)
    land_use_matrix, investment_matrix = build_decision_matrices(cleaned_df)

    page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🧹 Data Quality",
        "👥 Demographics",
        "🚆 Transport & Commuting",
        "💼 Employment",
        "📊 Decision Matrix",
        "⚠️ Limitations"
    ]
)
    
    st.sidebar.divider()

    st.sidebar.subheader("Project Summary")
    st.sidebar.caption(
        "Mock census dashboard for evidence-based local planning."
    )

    st.sidebar.markdown("**Recommended land use**")
    st.sidebar.success("Train station")

    st.sidebar.markdown("**Investment priority**")
    st.sidebar.info("Employment & training")

    st.sidebar.caption("Built with Python, Pandas, Plotly and Streamlit.")

    st.sidebar.divider()

    st.sidebar.subheader("Filters")
    st.sidebar.caption("Use 'All' selected to include every category.")

    gender_options = sorted(cleaned_df["gender"].dropna().unique().tolist())
    gender_filter_options = ["All"] + gender_options

    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=gender_filter_options,
        default=["All"]
    )

    if "All" in selected_gender or not selected_gender:
        selected_gender = gender_options

    age_group_options = sorted(cleaned_df["age_group"].dropna().astype(str).unique().tolist())
    age_group_filter_options = ["All"] + age_group_options

    selected_age_groups = st.sidebar.multiselect(
        "Age group",
        options=age_group_filter_options,
        default=["All"]
    )

    if "All" in selected_age_groups or not selected_age_groups:
        selected_age_groups = age_group_options

    occupancy_options = sorted(cleaned_df["occupancy_band"].dropna().astype(str).unique().tolist())
    occupancy_filter_options = ["All"] + occupancy_options

    selected_occupancy = st.sidebar.multiselect(
        "Occupancy band",
        options=occupancy_filter_options,
        default=["All"]
    )

    if "All" in selected_occupancy or not selected_occupancy:
        selected_occupancy = occupancy_options

    filtered_df = cleaned_df[
        cleaned_df["gender"].isin(selected_gender)
        & cleaned_df["age_group"].astype(str).isin(selected_age_groups)
        & cleaned_df["occupancy_band"].astype(str).isin(selected_occupancy)
    ].copy()

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please adjust the sidebar filters.")
        st.stop()

    st.sidebar.caption(f"Filtered residents: {len(filtered_df):,}")

    filters_active = len(filtered_df) != len(cleaned_df)

    if filters_active:
        filter_status = "Filters active"
        filter_status_message = f"Showing {len(filtered_df):,} of {len(cleaned_df):,} residents"
    else:
        filter_status = "Full dataset"
        filter_status_message = f"Showing all {len(cleaned_df):,} residents"

    if filtered_df.empty:
        st.warning("No records match the selected filters. Please adjust the sidebar filters.")
        st.stop()

    if page == "🏠 Overview":
        st.header("Town Overview")

        st.write(
            """
            This dashboard analyses a mock census dataset to support two local
            government planning decisions: what to build on an unoccupied plot
            of land, and which public investment area should be prioritised.
            """
        )

        status_col1, status_col2, status_col3 = st.columns(3)

        status_col1.success("Dataset loaded")
        status_col1.caption(f"{cleaned_df.shape[0]:,} records available")

        if filters_active:
            status_col2.warning(filter_status)
        else:
            status_col2.info(filter_status)

        status_col2.caption(filter_status_message)

        status_col3.info("Decision matrix uses full town data")
        status_col3.caption("Final recommendation is not filter-dependent")

        total_residents = filtered_df.shape[0]
        total_households = filtered_df["household_id"].nunique()
        working_age_count = filtered_df["is_working_age"].sum()

        commuter_share = filtered_df["is_commuter_proxy"].mean()
        university_student_share = filtered_df["is_university_student"].mean()
        
        working_age_unemployment_rate = (
            (filtered_df["is_unemployed"] & filtered_df["is_working_age"]).sum()
            / working_age_count
            if working_age_count
            else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total residents", f"{total_residents:,}")
        col2.metric("Total households", f"{total_households:,}")
        col3.metric("Commuter proxy share", f"{commuter_share:.1%}")
        col4.metric("Working-age unemployment", f"{working_age_unemployment_rate:.1%}")

        st.subheader("Final Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown("### 🚆 Train station")
            st.caption("Recommended land use")
            st.metric("Decision score", "0.942")
            st.write(
                "Supported by commuter proxy share of 25.1% and university student share of 6.4%."
        )

        with rec_col2:
            st.markdown("### 💼 Employment and training")
            st.caption("Recommended investment priority")
            st.metric("Decision score", "0.882")
            st.write(
                "Supported by working-age unemployment rate of 8.8%."
        )

        st.subheader("About This Project")

        st.write(
            """
            This project demonstrates how census-style demographic data can be cleaned,
            transformed and analysed to support local policy planning. The dashboard
            compares infrastructure and investment options using transparent indicators
            derived from the dataset.
            """
        )

        about_col1, about_col2, about_col3 = st.columns(3)

        with about_col1:
            st.markdown("#### Data Processing")
            st.write(
                """
                Raw census records were cleaned by standardising column names,
                handling missing values, converting age fields and creating
                household-level features.
                """
            )

        with about_col2:
            st.markdown("#### Policy Analysis")
            st.write(
                """
                The analysis uses demographic, employment, household and commuter
                indicators to compare possible land-use and investment priorities.
                """
            )

        with about_col3:
            st.markdown("#### Decision Support")
            st.write(
                """
                A transparent scoring matrix is used to explain why a train station
                and employment-focused investment are recommended.
                """
            )

        st.subheader("Methods Used")

        method_col1, method_col2, method_col3, method_col4 = st.columns(4)

        method_col1.metric("Data cleaning", "✓")
        method_col2.metric("Feature engineering", "✓")
        method_col3.metric("Exploratory analysis", "✓")
        method_col4.metric("Decision matrix", "✓")

        st.subheader("Key Outputs")

        st.markdown(
            """
            - Cleaned census dataset with standardised variables and engineered features.
            - Demographic profile of the town by age, gender and household structure.
            - Commuter-proxy analysis supporting transport infrastructure planning.
            - Employment analysis supporting skills and training investment.
            - Decision matrix comparing land-use and investment options.
            - Downloadable cleaned dataset and decision matrix outputs.
            """
        )
        add_footer()

    elif page == "🧹 Data Quality":
        st.header("Data Quality and Cleaning")

        st.subheader("Raw Dataset Preview")
        st.dataframe(raw_df.head(), use_container_width=True)

        st.subheader("Headline Raw Dataset Metrics")

        total_residents = raw_df.shape[0]
        total_columns = raw_df.shape[1]
        total_households = raw_df[["House Number", "Street"]].drop_duplicates().shape[0]
        total_missing_values = raw_df.isna().sum().sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total residents", f"{total_residents:,}")
        col2.metric("Total households", f"{total_households:,}")
        col3.metric("Columns", f"{total_columns:,}")
        col4.metric("Missing values", f"{total_missing_values:,}")

        st.subheader("Missing Values by Column")

        missing_summary = raw_df.isna().sum().reset_index()
        missing_summary.columns = ["Column", "Missing values"]
        missing_summary["Missing percentage"] = (
            missing_summary["Missing values"] / len(raw_df) * 100
        ).round(2)

        st.dataframe(missing_summary, use_container_width=True)

        st.subheader("Missing Values Chart")

        missing_chart = missing_summary[missing_summary["Missing values"] > 0]

        fig_missing = px.bar(
            missing_chart,
            x="Column",
            y="Missing values",
            text="Missing percentage",
            title="Columns with Missing Values"
        )

        fig_missing.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig_missing.update_layout(
            xaxis_title="Column",
            yaxis_title="Missing values",
            showlegend=False
        )

        st.plotly_chart(
            fig_missing,
            use_container_width=True,
            key="data_quality_missing_values_chart")

        st.subheader("Cleaned Dataset Preview")

        st.write(
            """
            The cleaned dataset standardises column names, fixes value inconsistencies,
            converts age to numeric format, and handles missing values using conservative rules.
            """
        )

        st.dataframe(cleaned_df.head(), use_container_width=True)

        cleaned_csv = convert_df_to_csv(cleaned_df)

        st.download_button(
            label="Download cleaned dataset",
            data=cleaned_csv,
            file_name="cleaned_census_dataset.csv",
            mime="text/csv"
        )

        cleaned_missing_values = cleaned_df.isna().sum().sum()

        clean_col1, clean_col2, clean_col3 = st.columns(3)

        clean_col1.metric("Cleaned rows", f"{cleaned_df.shape[0]:,}")
        clean_col2.metric("Cleaned columns", f"{cleaned_df.shape[1]:,}")
        clean_col3.metric("Remaining missing values", f"{cleaned_missing_values:,}")
        add_footer()
        
    elif page == "👥 Demographics":
        st.header("Demographic Profile")

        st.write(
            """
            This section explores the age and gender structure of the town.
            These indicators help assess schooling demand, ageing pressure,
            working-age population size, and future service needs.
            """
        )
        if filters_active:
            st.warning(
                f"Filters are active. This page is showing {len(filtered_df):,} "
                f"of {len(cleaned_df):,} residents."
            )    
        age_counts = (
            filtered_df["age_group"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        age_counts.columns = ["Age group", "Residents"]

        fig_age = px.bar(
            age_counts,
            x="Age group",
            y="Residents",
            title="Population by Age Group"
        )

        st.plotly_chart(fig_age, use_container_width=True, key="demographics_age_chart")

        gender_counts = filtered_df["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Residents"]

        fig_gender = px.pie(
            gender_counts,
            names="Gender",
            values="Residents",
            title="Gender Distribution"
        )

        st.plotly_chart(fig_gender, use_container_width=True, key="demographics_gender_chart")

        st.subheader("Life Stage Indicators")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("School-age share", f"{filtered_df['is_school_age'].mean():.1%}")
        col2.metric("Working-age share", f"{filtered_df['is_working_age'].mean():.1%}")
        col3.metric("Retirement-age share", f"{filtered_df['is_retirement_age'].mean():.1%}")
        col4.metric("Under-5 share", f"{filtered_df['is_under_5'].mean():.1%}")
        add_footer()

    elif page == "🚆 Transport & Commuting":
        st.header("Transport and Commuting Analysis")

        st.write(
            """
            This section explores the commuter indicators used to support the train
            station recommendation. The dataset does not contain direct journey-to-work
            data, so commuting is estimated using proxy indicators such as university
            student status and selected occupations that may involve travel to nearby cities.
            """
        )

        if filters_active:
            st.warning(
                f"Filters are active. This page is showing {len(filtered_df):,} "
                f"of {len(cleaned_df):,} residents."
            ) 

        commuter_share = filtered_df["is_commuter_proxy"].mean()
        university_student_share = filtered_df["is_university_student"].mean()

        col1, col2 = st.columns(2)

        col1.metric("Commuter proxy share", f"{commuter_share:.1%}")
        col2.metric("University student share", f"{university_student_share:.1%}")

        st.info(
            """
            University students are treated as a strong commuter indicator because the town
            has no university. This means university students living in the town are likely
            to travel elsewhere for study.
            """
        )

        commuter_counts = (
            filtered_df["is_commuter_proxy"]
            .map({True: "Commuter proxy", False: "Non-commuter proxy"})
            .value_counts()
            .reset_index()
        )

        commuter_counts.columns = ["Group", "Residents"]

        fig_commuter = px.bar(
            commuter_counts,
            x="Group",
            y="Residents",
            text="Residents",
            title="Commuter Proxy Distribution"
        )

        fig_commuter.update_traces(textposition="outside")

        fig_commuter.update_layout(
            xaxis_title="Group",
            yaxis_title="Number of residents",
            showlegend=False
        )

        st.plotly_chart(
            fig_commuter,
            use_container_width=True,
            key="transport_commuter_proxy_chart"
        )

        st.subheader("Top Occupations Contributing to Commuter Proxy")

        commuter_occupations = (
            filtered_df[filtered_df["is_commuter_proxy"]]
            ["occupation"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        commuter_occupations.columns = ["Occupation", "Residents"]

        fig_commuter_jobs = px.bar(
            commuter_occupations,
            x="Residents",
            y="Occupation",
            orientation="h",
            text="Residents",
            title="Top 10 Commuter-Proxy Occupations"
        )

        fig_commuter_jobs.update_traces(textposition="outside")

        fig_commuter_jobs.update_layout(
            xaxis_title="Number of residents",
            yaxis_title="Occupation",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_commuter_jobs,
            use_container_width=True,
            key="transport_commuter_jobs_chart"
        )

        st.subheader("Interpretation")

        st.write(
            """
            The commuter proxy share is one of the strongest signals in the analysis.
            With around a quarter of residents falling into the commuter-proxy category,
            and a notable university-student population, the data supports the case for
            improved transport connectivity.

            This does not prove that every commuter-proxy resident would use rail.
            However, it provides enough evidence to treat transport infrastructure as a
            serious planning priority. In a real-world setting, this recommendation would
            need to be validated with journey-to-work data, traffic counts, transport
            surveys and cost-benefit analysis.
            """
        )

        st.success(
            "Land-use recommendation supported by this page: Build a train station."
        )
        add_footer()

    elif page == "💼 Employment":
        st.header("Employment and Training Analysis")

        st.write(
            """
            This section explores the employment indicators used to support the
            investment recommendation. The main focus is working-age unemployment,
            because this provides direct evidence for employment support, retraining
            and skills-development programmes.
            """
        )

        if filters_active:
            st.warning(
                f"Filters are active. This page is showing {len(filtered_df):,} "
                f"of {len(cleaned_df):,} residents."
            )

        working_age_df = filtered_df[filtered_df["is_working_age"]].copy()
        working_age_count = len(working_age_df)

        unemployment_rate = (
            working_age_df["is_unemployed"].mean()
            if working_age_count
            else 0
        )

        student_share = filtered_df["is_student"].mean()
        university_student_share = filtered_df["is_university_student"].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric("Working-age residents", f"{working_age_count:,}")
        col2.metric("Working-age unemployment", f"{unemployment_rate:.1%}")
        col3.metric("University student share", f"{university_student_share:.1%}")

        st.info(
            """
            Employment and training is recommended because the town has a sizeable
            working-age population and a notable unemployment rate. This creates a
            case for skills development, retraining, employability support and
            education-to-work transition programmes.
            """
        )

        st.subheader("Unemployment by Gender")

        unemployment_by_gender = (
            working_age_df
            .groupby("gender", dropna=False)["is_unemployed"]
            .mean()
            .reset_index()
        )

        unemployment_by_gender["Unemployment rate"] = (
            unemployment_by_gender["is_unemployed"] * 100
        ).round(2)

        fig_unemployment_gender = px.bar(
            unemployment_by_gender,
            x="gender",
            y="Unemployment rate",
            text="Unemployment rate",
            title="Working-Age Unemployment Rate by Gender"
        )

        fig_unemployment_gender.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig_unemployment_gender.update_layout(
            xaxis_title="Gender",
            yaxis_title="Unemployment rate (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig_unemployment_gender,
            use_container_width=True,
            key="employment_unemployment_gender_chart"
        )

        st.subheader("Unemployment by Age Group")

        unemployment_by_age = (
            working_age_df
            .groupby("age_group", observed=True)["is_unemployed"]
            .mean()
            .reset_index()
        )

        unemployment_by_age["Unemployment rate"] = (
            unemployment_by_age["is_unemployed"] * 100
        ).round(2)

        fig_unemployment_age = px.bar(
            unemployment_by_age,
            x="age_group",
            y="Unemployment rate",
            text="Unemployment rate",
            title="Working-Age Unemployment Rate by Age Group"
        )

        fig_unemployment_age.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig_unemployment_age.update_layout(
            xaxis_title="Age group",
            yaxis_title="Unemployment rate (%)",
            showlegend=False
        )

        st.plotly_chart(
            fig_unemployment_age,
            use_container_width=True,
            key="employment_unemployment_age_chart"
        )

        st.subheader("Top Occupations")

        top_occupations = (
            filtered_df["occupation"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_occupations.columns = ["Occupation", "Residents"]

        fig_top_occupations = px.bar(
            top_occupations,
            x="Residents",
            y="Occupation",
            orientation="h",
            text="Residents",
            title="Top 10 Occupations in the Town"
        )

        fig_top_occupations.update_traces(textposition="outside")

        fig_top_occupations.update_layout(
            xaxis_title="Number of residents",
            yaxis_title="Occupation",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"}
        )

        st.plotly_chart(
            fig_top_occupations,
            use_container_width=True,
            key="employment_top_occupations_chart"
        )

        st.subheader("Interpretation")

        st.write(
            """
            The working-age unemployment rate is the strongest investment signal
            in the dashboard. It points to a need for targeted employment support,
            retraining, graduate transition programmes and skills-development
            initiatives.

            This recommendation is also complementary to the train station
            recommendation. Better transport connectivity can help residents access
            opportunities in nearby cities, while employment and training programmes
            can help them compete for those opportunities.
            """
        )

        st.success(
            "Investment recommendation supported by this page: Employment and training."
        )
        add_footer()

    elif page == "📊 Decision Matrix":
        st.header("Decision Matrix")

        st.write(
            """
            This page compares the available land-use and investment options using
            transparent proxy indicators from the cleaned census dataset. The scores
            are not absolute truth; they are a structured way to compare the strength
            of available evidence.
            """
        )

        st.info(
            """
            This page is based on the full town dataset, not the sidebar-filtered view.
            The final recommendation represents the whole town.
            """
        )

        st.warning(
            """
            Important: some options rely on stronger evidence than others. For example,
            employment status and age are direct census variables, while commuting and
            religious-building demand are proxy-based. Religion is treated conservatively
            because more than half of residents did not state a religion.
            """
        )

        st.subheader("Land-Use Decision Matrix")

        st.dataframe(
            land_use_matrix,
            use_container_width=True,
            hide_index=True
        )

        land_use_csv = convert_df_to_csv(land_use_matrix)

        st.download_button(
            label="Download land-use decision matrix",
            data=land_use_csv,
            file_name="land_use_decision_matrix.csv",
            mime="text/csv"
        )

        fig_land_use_scores = px.bar(
            land_use_matrix,
            x="Score",
            y="Option",
            orientation="h",
            text="Score",
            title="Land-Use Option Scores"
        )

        fig_land_use_scores.update_traces(textposition="outside")

        fig_land_use_scores.update_layout(
            xaxis_title="Score",
            yaxis_title="Land-use option",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
            xaxis_range=[0, 1.1]
        )

        st.plotly_chart(
            fig_land_use_scores,
            use_container_width=True,
            key="decision_land_use_scores_chart"
        )

        top_land_use = land_use_matrix.iloc[0]

        st.success(
            f"Recommended land use: {top_land_use['Option']} "
            f"(score: {top_land_use['Score']})"
        )

        st.write(f"**Main evidence:** {top_land_use['Main evidence']}")
        st.write(f"**Rationale:** {top_land_use['Rationale']}")
        st.write(f"**Evidence limitation:** {top_land_use['Evidence limitation']}")

        st.divider()

        st.subheader("Investment Decision Matrix")

        st.dataframe(
            investment_matrix,
            use_container_width=True,
            hide_index=True
        )

        investment_csv = convert_df_to_csv(investment_matrix)

        st.download_button(
            label="Download investment decision matrix",
            data=investment_csv,
            file_name="investment_decision_matrix.csv",
            mime="text/csv"
        )

        fig_investment_scores = px.bar(
            investment_matrix,
            x="Score",
            y="Option",
            orientation="h",
            text="Score",
            title="Investment Option Scores"
        )

        fig_investment_scores.update_traces(textposition="outside")

        fig_investment_scores.update_layout(
            xaxis_title="Score",
            yaxis_title="Investment option",
            showlegend=False,
            yaxis={"categoryorder": "total ascending"},
            xaxis_range=[0, 1.1]
        )

        st.plotly_chart(
            fig_investment_scores,
            use_container_width=True,
            key="decision_investment_scores_chart"
        )

        top_investment = investment_matrix.iloc[0]

        st.info(
            f"Recommended investment priority: {top_investment['Option']} "
            f"(score: {top_investment['Score']})"
        )

        st.write(f"**Main evidence:** {top_investment['Main evidence']}")
        st.write(f"**Rationale:** {top_investment['Rationale']}")
        st.write(f"**Evidence limitation:** {top_investment['Evidence limitation']}")
        add_footer()

    elif page == "⚠️ Limitations":
        st.header("Limitations and Assumptions")

        st.write(
            """
            This dashboard is designed as a data science and policy-planning exercise.
            The recommendations are based on indicators available in the mock census
            dataset, but several important real-world planning variables are not present.
            """
        )

        st.subheader("Key Limitations")

        st.markdown(
            """
            - **Synthetic dataset:** The census data is mock data, so the recommendations
              should be interpreted as a portfolio case study rather than a real policy decision.

            - **Commuting is inferred:** The dataset does not include direct journey-to-work
              data, transport mode, travel frequency, car ownership, or traffic volume.
              Commuter demand is estimated using university student status and selected
              occupation keywords.

            - **Religion has high missingness:** More than half of residents did not state
              a religion. For this reason, religious-building demand is treated conservatively
              in the scoring framework.

            - **Housing pressure is incomplete:** The dataset does not include rent levels,
              house prices, overcrowding standards, housing waiting lists, income, or
              historical population growth.

            - **Healthcare demand is incomplete:** The dataset does not include GP waiting
              times, hospital admissions, pregnancy records, accident rates, disability
              support needs, or distance to existing health facilities.

            - **Schooling demand is incomplete:** The dataset includes age and student status,
              but does not include school capacity, class sizes, catchment areas, school
              performance, or projected enrolment growth.

            - **Scores are decision-support indicators:** The score values should not be
              interpreted as exact probabilities. They are a structured way to compare
              evidence strength across options.
            """
        )

        st.subheader("Main Assumptions")

        assumptions = pd.DataFrame([
            {
                "Area": "Commuting",
                "Assumption": (
                    "University students are likely commuters because the town has no university."
                ),
                "Risk": (
                    "Some students may study remotely or outside the assumed nearby-city pattern."
                ),
            },
            {
                "Area": "Occupation",
                "Assumption": (
                    "Certain occupations are used as commuter proxies because they may require "
                    "travel to nearby cities."
                ),
                "Risk": (
                    "Occupation alone does not confirm actual commuting behaviour."
                ),
            },
            {
                "Area": "Religion",
                "Assumption": (
                    "Missing religion values are treated as 'Not stated' rather than imputed."
                ),
                "Risk": (
                    "True religious affiliation may be undercounted."
                ),
            },
            {
                "Area": "Infirmity",
                "Assumption": (
                    "Missing infirmity values are treated as 'None recorded'."
                ),
                "Risk": (
                    "This may understate actual health or accessibility needs."
                ),
            },
            {
                "Area": "Housing",
                "Assumption": (
                    "Household size is used as a partial proxy for housing pressure."
                ),
                "Risk": (
                    "Household size does not directly measure overcrowding or affordability."
                ),
            },
            {
                "Area": "Employment",
                "Assumption": (
                    "Residents marked as unemployed within working age are used to estimate "
                    "employment-support need."
                ),
                "Risk": (
                    "The dataset does not show duration of unemployment, skills gaps, or job-search status."
                ),
            },
        ])

        st.dataframe(
            assumptions,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("What Additional Data Would Improve the Recommendation?")

        additional_data = pd.DataFrame([
            {
                "Decision area": "Train station",
                "Useful additional data": (
                    "Journey-to-work flows, traffic counts, public transport usage, car ownership, "
                    "commuter surveys, rail feasibility and cost-benefit analysis."
                ),
            },
            {
                "Decision area": "Housing",
                "Useful additional data": (
                    "House prices, rent levels, income, overcrowding measures, housing waiting lists, "
                    "planning applications and population growth trends."
                ),
            },
            {
                "Decision area": "Religious building",
                "Useful additional data": (
                    "Worship attendance, facility capacity, community consultation, distance to existing "
                    "places of worship and unmet-demand evidence."
                ),
            },
            {
                "Decision area": "Emergency medical building",
                "Useful additional data": (
                    "GP waiting times, hospital admissions, A&E usage, accident rates, pregnancy records, "
                    "distance to health facilities and local care demand."
                ),
            },
            {
                "Decision area": "Employment and training",
                "Useful additional data": (
                    "Skills gaps, job vacancies, unemployment duration, education history, employer demand "
                    "and training completion outcomes."
                ),
            },
            {
                "Decision area": "Schooling",
                "Useful additional data": (
                    "School capacity, class sizes, catchment pressure, enrolment projections and birth trends."
                ),
            },
        ])

        st.dataframe(
            additional_data,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Final Interpretation")

        st.info(
            """
            The dashboard recommends a train station because the commuter proxy and
            university-student indicators are the strongest land-use signals in the
            available data. It recommends employment and training because working-age
            unemployment is the clearest investment signal.

            These recommendations should be treated as evidence-based starting points,
            not final policy decisions. In a real setting, they would need to be tested
            against cost, feasibility, community consultation and external administrative data.
            """
        )
        add_footer()


else:
    st.error(
        "Dataset not found. Please make sure your CSV is saved in "
        "`data/raw/mock_census.csv`."
    )

