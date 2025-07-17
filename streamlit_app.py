import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# App configuration
st.set_page_config(page_title="Energy Savings Calculator", layout="wide")
st.title("Energy Savings Calculator")
st.markdown("""
Calculate energy savings by comparing pre-installation and post-installation electrical data.
Upload your voltage, current, and power factor data or enter average values manually.
""")

# Constants
TIME_INTERVAL_HOURS = 5/60  # 5 minutes in hours

# Helper functions
def safe_convert_to_float(value):
    """Safely convert values to float, handling various formats"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    if pd.isna(value):
        return 0.0
    return float(value)

def calculate_energy(voltage, current, pf):
    """Calculate energy consumption in kWh with proper handling of mixed input types"""
    try:
        # Convert all inputs to numeric series if they aren't already
        if not isinstance(voltage, (pd.Series, np.ndarray)):
            voltage = pd.Series([float(voltage)])
        if not isinstance(current, (pd.Series, np.ndarray)):
            current = pd.Series([float(current)])
        if not isinstance(pf, (pd.Series, np.ndarray)):
            pf = pd.Series([float(pf)])
        
        # Ensure all series have the same length
        max_len = max(len(voltage), len(current), len(pf))
        voltage = voltage if len(voltage) == max_len else pd.Series([voltage.iloc[0]] * max_len)
        current = current if len(current) == max_len else pd.Series([current.iloc[0]] * max_len)
        pf = pf if len(pf) == max_len else pd.Series([pf.iloc[0]] * max_len)
        
        # Calculate energy
        energy = (voltage * current * pf * TIME_INTERVAL_HOURS / 1000).sum()
        return energy
    except Exception as e:
        st.error(f"Energy calculation error: {str(e)}")
        return 0

def process_uploaded_file(uploaded_file, expected_columns, is_pf=False):
    """Process uploaded file with better error handling"""
    if uploaded_file is None:
        return None
        
    try:
        # Try reading as Excel first, then CSV
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        expected_columns = [col.strip().lower() for col in expected_columns]
        
        # Check required columns
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return None
        
        # Convert all numeric columns (except timestamp)
        for col in df.columns:
            if col != 'collection time':
                # Handle European decimal format and other non-numeric cases
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    st.warning(f"Some values in column {col} couldn't be converted to numbers")
        
        # Validate power factor range if needed
        if is_pf:
            pf_cols = [col for col in df.columns if 'pf' in col.lower()]
            for col in pf_cols:
                if (df[col] < 0).any() or (df[col] > 1).any():
                    st.error(f"Power factor values in {col} must be between 0 and 1")
                    return None
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def display_data_summary(df, title):
    """Display summary of uploaded data"""
    with st.expander(f"{title} Data Summary"):
        if df is not None:
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            st.dataframe(df.head(3))
            st.write("Statistical Summary:")
            st.dataframe(df.describe())
        else:
            st.write("No data uploaded")

def calculate_cumulative_energy(voltage_data, current_data, pf_data, 
                              avg_voltage=None, avg_current=None, avg_pf=None):
    """More robust cumulative energy calculation"""
    try:
        # Get voltage values
        if avg_voltage is not None:
            ua = ub = uc = safe_convert_to_float(avg_voltage)
        elif voltage_data is not None:
            ua = pd.to_numeric(voltage_data['ua(v)'], errors='coerce').fillna(0)
            ub = pd.to_numeric(voltage_data['ub(v)'], errors='coerce').fillna(0)
            uc = pd.to_numeric(voltage_data['uc(v)'], errors='coerce').fillna(0)
        else:
            raise ValueError("No voltage data provided")
        
        # Get current values
        if avg_current is not None:
            ia = ib = ic = safe_convert_to_float(avg_current)
        elif current_data is not None:
            ia = pd.to_numeric(current_data['ia(a)'], errors='coerce').fillna(0)
            ib = pd.to_numeric(current_data['ib(a)'], errors='coerce').fillna(0)
            ic = pd.to_numeric(current_data['ic(a)'], errors='coerce').fillna(0)
        else:
            raise ValueError("No current data provided")
        
        # Get power factor values
        if avg_pf is not None:
            pfa = pfb = pfc = safe_convert_to_float(avg_pf)
        elif pf_data is not None:
            pfa = pd.to_numeric(pf_data['pfa'], errors='coerce').fillna(0)
            pfb = pd.to_numeric(pf_data['pfb'], errors='coerce').fillna(0)
            pfc = pd.to_numeric(pf_data['pfc'], errors='coerce').fillna(0)
        else:
            raise ValueError("No power factor data provided")
        
        # Calculate cumulative energy with NaN protection
        cumulative_energy = (
            (ua * ia * pfa +
             ub * ib * pfb +
             uc * ic * pfc) * 
            (TIME_INTERVAL_HOURS / 1000)
        ).cumsum().fillna(0)
        
        return cumulative_energy
    except Exception as e:
        st.error(f"Cumulative energy calculation error: {str(e)}")
        return None

# Sidebar for settings
with st.sidebar:
    st.header("Calculation Settings")
    time_interval = st.number_input("Measurement interval (minutes)", 
                                  min_value=1, max_value=60, value=5)
    TIME_INTERVAL_HOURS = time_interval / 60
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Upload data or enter averages for any combination of parameters
    2. At least one source must be provided for each parameter
    3. Click 'Calculate Savings'
    4. View results and download report
    """)

# Main calculation function
def calculate_savings():
    """Main function to calculate and display energy savings"""
    st.header("Pre-Installation Data")
    pre_col1, pre_col2, pre_col3 = st.columns(3)
    
    with pre_col1:
        st.subheader("Voltage Data")
        pre_voltage_method = st.radio("Pre Voltage Input Method",
                                    ("Upload Data", "Enter Average"),
                                    key="pre_voltage_method")
        if pre_voltage_method == "Upload Data":
            pre_voltage_file = st.file_uploader("Upload Pre-Installation Voltage Data",
                                              type=["xlsx", "xls", "csv"],
                                              key="pre_voltage")
            pre_voltage_df = process_uploaded_file(pre_voltage_file, 
                                                 ["Collection time", "Ua(V)", "Ub(V)", "Uc(V)"])
            display_data_summary(pre_voltage_df, "Pre-Installation Voltage")
            pre_voltage_avg = None
        else:
            pre_voltage_avg = st.number_input("Enter Average Phase Voltage (V)",
                                             min_value=100.0, max_value=300.0, 
                                             value=230.0, step=1.0,
                                             key="pre_voltage_avg")
            pre_voltage_df = None
    
    with pre_col2:
        st.subheader("Current Data")
        pre_current_method = st.radio("Pre Current Input Method",
                                    ("Upload Data", "Enter Average"),
                                    key="pre_current_method")
        if pre_current_method == "Upload Data":
            pre_current_file = st.file_uploader("Upload Pre-Installation Current Data",
                                              type=["xlsx", "xls", "csv"],
                                              key="pre_current")
            pre_current_df = process_uploaded_file(pre_current_file, 
                                                 ["Collection time", "Ia(A)", "Ib(A)", "Ic(A)"])
            display_data_summary(pre_current_df, "Pre-Installation Current")
            pre_current_avg = None
        else:
            pre_current_avg = st.number_input("Enter Average Phase Current (A)",
                                             min_value=0.0, max_value=1000000.0,
                                             value=50.0, step=1.0,
                                             key="pre_current_avg")
            pre_current_df = None
    
    with pre_col3:
        st.subheader("Power Factor Data")
        pre_pf_method = st.radio("Pre PF Input Method",
                                ("Upload Data", "Enter Average"),
                                key="pre_pf_method")
        if pre_pf_method == "Upload Data":
            pre_pf_file = st.file_uploader("Upload Pre-Installation Power Factor Data",
                                          type=["xlsx", "xls", "csv"],
                                          key="pre_pf")
            pre_pf_df = process_uploaded_file(pre_pf_file, 
                                           ["Collection time", "PFa", "PFb", "PFc"],
                                           is_pf=True)
            display_data_summary(pre_pf_df, "Pre-Installation Power Factor")
            pre_pf_avg = None
        else:
            pre_pf_avg = st.number_input("Enter Average Power Factor",
                                        min_value=0.0, max_value=1.0, 
                                        value=0.85, step=0.01,
                                        key="pre_pf_avg")
            pre_pf_df = None
    
    st.markdown("---")
    st.header("Post-Installation Data")
    post_col1, post_col2, post_col3 = st.columns(3)
    
    with post_col1:
        st.subheader("Voltage Data")
        post_voltage_method = st.radio("Post Voltage Input Method",
                                      ("Upload Data", "Enter Average"),
                                      key="post_voltage_method")
        if post_voltage_method == "Upload Data":
            post_voltage_file = st.file_uploader("Upload Post-Installation Voltage Data",
                                                type=["xlsx", "xls", "csv"],
                                                key="post_voltage")
            post_voltage_df = process_uploaded_file(post_voltage_file, 
                                                   ["Collection time", "Ua(V)", "Ub(V)", "Uc(V)"])
            display_data_summary(post_voltage_df, "Post-Installation Voltage")
            post_voltage_avg = None
        else:
            post_voltage_avg = st.number_input("Enter Average Phase Voltage (V)",
                                             min_value=100.0, max_value=300.0, 
                                             value=230.0, step=1.0,
                                             key="post_voltage_avg")
            post_voltage_df = None
    
    with post_col2:
        st.subheader("Current Data")
        post_current_method = st.radio("Post Current Input Method",
                                      ("Upload Data", "Enter Average"),
                                      key="post_current_method")
        if post_current_method == "Upload Data":
            post_current_file = st.file_uploader("Upload Post-Installation Current Data",
                                                type=["xlsx", "xls", "csv"],
                                                key="post_current")
            post_current_df = process_uploaded_file(post_current_file, 
                                                   ["Collection time", "Ia(A)", "Ib(A)", "Ic(A)"])
            display_data_summary(post_current_df, "Post-Installation Current")
            post_current_avg = None
        else:
            post_current_avg = st.number_input("Enter Average Phase Current (A)",
                                             min_value=0.0, max_value=1000000.0,
                                             value=40.0, step=1.0,
                                             key="post_current_avg")
            post_current_df = None
    
    with post_col3:
        st.subheader("Power Factor Data")
        post_pf_method = st.radio("Post PF Input Method",
                                ("Upload Data", "Enter Average"),
                                key="post_pf_method")
        if post_pf_method == "Upload Data":
            post_pf_file = st.file_uploader("Upload Post-Installation Power Factor Data",
                                          type=["xlsx", "xls", "csv"],
                                          key="post_pf")
            post_pf_df = process_uploaded_file(post_pf_file, 
                                             ["Collection time", "PFa", "PFb", "PFc"],
                                             is_pf=True)
            display_data_summary(post_pf_df, "Post-Installation Power Factor")
            post_pf_avg = None
        else:
            post_pf_avg = st.number_input("Enter Average Power Factor",
                                        min_value=0.0, max_value=1.0, 
                                        value=0.95, step=0.01,
                                        key="post_pf_avg")
            post_pf_df = None
    
    # Calculate button
    if st.button("Calculate Energy Savings", use_container_width=True):
        # Validate inputs
        validation_errors = []
        
        # Pre-installation validation
        if pre_voltage_method == "Upload Data" and pre_voltage_df is None:
            validation_errors.append("Please upload pre-installation voltage data or enter average")
        elif pre_voltage_method == "Enter Average" and pre_voltage_avg is None:
            validation_errors.append("Please enter pre-installation average voltage")
        
        if pre_current_method == "Upload Data" and pre_current_df is None:
            validation_errors.append("Please upload pre-installation current data or enter average")
        elif pre_current_method == "Enter Average" and pre_current_avg is None:
            validation_errors.append("Please enter pre-installation average current")
        
        if pre_pf_method == "Upload Data" and pre_pf_df is None:
            validation_errors.append("Please upload pre-installation power factor data or enter average")
        elif pre_pf_method == "Enter Average" and pre_pf_avg is None:
            validation_errors.append("Please enter pre-installation average power factor")
        
        # Post-installation validation
        if post_voltage_method == "Upload Data" and post_voltage_df is None:
            validation_errors.append("Please upload post-installation voltage data or enter average")
        elif post_voltage_method == "Enter Average" and post_voltage_avg is None:
            validation_errors.append("Please enter post-installation average voltage")
        
        if post_current_method == "Upload Data" and post_current_df is None:
            validation_errors.append("Please upload post-installation current data or enter average")
        elif post_current_method == "Enter Average" and post_current_avg is None:
            validation_errors.append("Please enter post-installation average current")
        
        if post_pf_method == "Upload Data" and post_pf_df is None:
            validation_errors.append("Please upload post-installation power factor data or enter average")
        elif post_pf_method == "Enter Average" and post_pf_avg is None:
            validation_errors.append("Please enter post-installation average power factor")
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return
        
        try:
            # Calculate pre-installation energy
            pre_energy = calculate_energy(
                pre_voltage_df['ua(v)'] if pre_voltage_method == "Upload Data" else pre_voltage_avg,
                pre_current_df['ia(a)'] if pre_current_method == "Upload Data" else pre_current_avg,
                pre_pf_df['pfa'] if pre_pf_method == "Upload Data" else pre_pf_avg
            ) + calculate_energy(
                pre_voltage_df['ub(v)'] if pre_voltage_method == "Upload Data" else pre_voltage_avg,
                pre_current_df['ib(a)'] if pre_current_method == "Upload Data" else pre_current_avg,
                pre_pf_df['pfb'] if pre_pf_method == "Upload Data" else pre_pf_avg
            ) + calculate_energy(
                pre_voltage_df['uc(v)'] if pre_voltage_method == "Upload Data" else pre_voltage_avg,
                pre_current_df['ic(a)'] if pre_current_method == "Upload Data" else pre_current_avg,
                pre_pf_df['pfc'] if pre_pf_method == "Upload Data" else pre_pf_avg
            )
            
            # Calculate post-installation energy
            post_energy = calculate_energy(
                post_voltage_df['ua(v)'] if post_voltage_method == "Upload Data" else post_voltage_avg,
                post_current_df['ia(a)'] if post_current_method == "Upload Data" else post_current_avg,
                post_pf_df['pfa'] if post_pf_method == "Upload Data" else post_pf_avg
            ) + calculate_energy(
                post_voltage_df['ub(v)'] if post_voltage_method == "Upload Data" else post_voltage_avg,
                post_current_df['ib(a)'] if post_current_method == "Upload Data" else post_current_avg,
                post_pf_df['pfb'] if post_pf_method == "Upload Data" else post_pf_avg
            ) + calculate_energy(
                post_voltage_df['uc(v)'] if post_voltage_method == "Upload Data" else post_voltage_avg,
                post_current_df['ic(a)'] if post_current_method == "Upload Data" else post_current_avg,
                post_pf_df['pfc'] if post_pf_method == "Upload Data" else post_pf_avg
            )
            
            # Calculate savings
            energy_savings = pre_energy - post_energy
            savings_percentage = (energy_savings / pre_energy) * 100 if pre_energy != 0 else 0
            
            # Display results
            st.success("Calculation Complete!")
            st.markdown("---")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pre-Installation Energy", f"{pre_energy:,.1f} kWh")
            with col2:
                st.metric("Post-Installation Energy", f"{post_energy:,.1f} kWh")
            with col3:
                st.metric("Energy Savings", 
                         f"{energy_savings:,.1f} kWh", 
                         f"{savings_percentage:.1f}%")
            
            # Visualization
            st.markdown("---")
            st.subheader("Energy Consumption Comparison")
            
            # Create comparison data
            comparison_data = pd.DataFrame({
                "Period": ["Pre-Installation", "Post-Installation"],
                "Energy (kWh)": [pre_energy, post_energy]
            })
            
            # Bar chart
            st.bar_chart(comparison_data.set_index('Period'))
            
            # Time series plot (if we have enough data)
            try:
                pre_cumulative = calculate_cumulative_energy(
                    pre_voltage_df if pre_voltage_method == "Upload Data" else None,
                    pre_current_df if pre_current_method == "Upload Data" else None,
                    pre_pf_df if pre_pf_method == "Upload Data" else None,
                    pre_voltage_avg if pre_voltage_method == "Enter Average" else None,
                    pre_current_avg if pre_current_method == "Enter Average" else None,
                    pre_pf_avg if pre_pf_method == "Enter Average" else None
                )
                
                post_cumulative = calculate_cumulative_energy(
                    post_voltage_df if post_voltage_method == "Upload Data" else None,
                    post_current_df if post_current_method == "Upload Data" else None,
                    post_pf_df if post_pf_method == "Upload Data" else None,
                    post_voltage_avg if post_voltage_method == "Enter Average" else None,
                    post_current_avg if post_current_method == "Enter Average" else None,
                    post_pf_avg if post_pf_method == "Enter Average" else None
                )
                
                # Get time column from the first available uploaded data
                time_col = None
                if pre_voltage_method == "Upload Data":
                    time_col = pre_voltage_df['collection time']
                elif post_voltage_method == "Upload Data":
                    time_col = post_voltage_df['collection time']
                elif pre_current_method == "Upload Data":
                    time_col = pre_current_df['collection time']
                elif post_current_method == "Upload Data":
                    time_col = post_current_df['collection time']
                elif pre_pf_method == "Upload Data":
                    time_col = pre_pf_df['collection time']
                elif post_pf_method == "Upload Data":
                    time_col = post_pf_df['collection time']
                
                if time_col is not None and pre_cumulative is not None and post_cumulative is not None:
                    st.subheader("Time Series Comparison")
                    time_series_data = pd.DataFrame({
                        "Time": time_col,
                        "Pre-Installation": pre_cumulative,
                        "Post-Installation": post_cumulative
                    })
                    st.line_chart(time_series_data.set_index('Time'))
                else:
                    st.warning("Time series chart requires at least one uploaded dataset with time information")
            except Exception as e:
                st.warning(f"Could not generate time series chart: {str(e)}")
            
            # Download report
            st.markdown("---")
            st.subheader("Download Report")
            
            # Create a simple report
            report = f"""
            Energy Savings Report
            
            Pre-Installation Energy: {pre_energy:,.1f} kWh
            Post-Installation Energy: {post_energy:,.1f} kWh
            Energy Savings: {energy_savings:,.1f} kWh ({savings_percentage:.1f}%)
            
            Calculation Parameters:
            - Time Interval: {time_interval} minutes
            - Pre-Installation Method: {'Uploaded Data' if pre_voltage_method == 'Upload Data' else 'Average Values'}
            - Post-Installation Method: {'Uploaded Data' if post_voltage_method == 'Upload Data' else 'Average Values'}
            """
            
            # Download button
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name="energy_savings_report.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"An error occurred during calculation: {str(e)}")
            st.error("Please check that:")
            st.error("- All required files are uploaded")
            st.error("- Uploaded files contain all required columns")
            st.error("- All numeric values are valid")

# Run the app
calculate_savings()

# Footer
st.markdown("---")
st.markdown("""
**Note:** This calculator assumes balanced three-phase power. 
For unbalanced systems, please upload complete phase data for accurate results.
""")