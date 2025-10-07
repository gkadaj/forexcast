import numpy as np
import altair as alt
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import prediction_model as pm
import pytz

def fetch_training_data(period: str):
    # Define the USD/JPY ticker
    ticker = "USDJPY=X"

    # Determine the last trading day
    now = datetime.now()
    start_time = now - timedelta(days=5)
    start_time = start_time.strftime('%Y-%m-%d')
    next_day = now + timedelta(days=1)
    end_time = next_day.strftime('%Y-%m-%d')

    # Download 1-minute interval data using yfinance
    data = yf.download(ticker, start=start_time, interval=period, ignore_tz=True)
    
    # Ensure that data is available and convert it to a DataFrame
    if data.empty:
        st.error("No data retrieved. Please ensure the stock market is open or check the ticker.")
    
    # Reset index and format DataFrame to include 'x' (time) and 'y' (price)
    data = data.reset_index()
    return data

# Fetch 1-minute historical data for USD/JPY for the last trading day
def fetch_usdjpy_data(interval: str):
    # Define the USD/JPY ticker
    ticker = "USDJPY=X"

    # Determine the last trading day
    now = datetime.now()
    previous_day = now - timedelta(days=1)
    start_time = previous_day.strftime('%Y-%m-%d')
    next_day = now + timedelta(days=1)
    end_time = next_day.strftime('%Y-%m-%d')

    try:
        data = yf.download(ticker, start=start_time, end=end_time, interval=interval, ignore_tz=False)
    except:
        st.error("Well, something went wrong")

    # Ensure that data is available and convert it to a DataFrame
    if data.empty:
        st.error("No data retrieved. Please ensure the stock market is open or check the ticker.")
    
    # Reset index and format DataFrame to include 'x' (time) and 'y' (price)
    data = data.reset_index()
    data['time'] = data['Datetime']
    data['close'] = data['Close']
    data['high'] = data['High']
    data['low'] = data['Low']
    return data[['time', 'close', 'low', 'high']]


def calculate_y_scale(df):
    # Calculate y-axis range from the data
    y_min = min(df.min(), st.session_state["targets_df"].min()) if "targets_df" in st.session_state else df.min()
    y_max = max(df.max(), st.session_state["targets_df"].max()) if "targets_df" in st.session_state else df.max()
    return [y_min, y_max]

def fetch_today_max_min(start_time: str):
    # Define the currency pair and starting time
    symbol = "JPY=X"  # USD/JPY ticker

    # Adjust time to UTC to match yfinance's timezone
    utc = pytz.UTC
    local_tz = pytz.timezone('Europe/Warsaw')  # Replace 'YourTimeZone' with your timezone, e.g., 'America/New_York'

    # Get today's date in local timezone
    local_now = datetime.now(local_tz).date()
    local_start_time = local_tz.localize(datetime.combine(local_now, datetime.strptime(start_time, "%H:%M").time()))
    utc_start_time = local_start_time.astimezone(utc)

    st.write(local_now)
    st.stop()
    # Fetch today's USD/JPY data with minute intervals
    data = yf.download(tickers=symbol, start=str(local_now), interval="1m")

    min_price = 0
    max_price = 0
    # Check if data exists
    if not data.empty:
        # Filter with respect to UTC timings (Datetime index is already in UTC)
        filtered_data = data[data.index >= utc_start_time]

        if not filtered_data.empty:
            # Extract the minimum and maximum prices
            min_price = float(filtered_data['Low'].min())
            max_price = float(filtered_data['High'].max())

    return [min_price, max_price]

st.set_page_config(layout="wide")
st.logo("imgs/logo.png", size="large")
st.html("""
  <style>
    [alt=Logo] {
      height: 4rem;
    }
  </style>
        """)

st.warning("Do not treat the presented data as investment advice. Use the data at your own risk.")


with st.sidebar:
    st.markdown("**BETA version**")
    targeting_period = st.selectbox("Prediction Interval (mins)", [5, 15])
    visible_minutes = st.selectbox("Chart range (mins)", [30, 60])

# Fetch currency data
usdjpy_data = fetch_usdjpy_data("1m")
usdjpy_data = usdjpy_data.iloc[max(0, len(usdjpy_data) - visible_minutes):len(usdjpy_data) + 1]

# Create a placeholder for the chart
chart_row = st.empty()
stat_cols = st.columns(2)

with stat_cols[0]:
    score_ui = st.empty()
    with score_ui.container():
        st.metric("AI Agent Performance Score", "N/A", help="The performance indicator is calculated by comparing the price valid at the time of calculation with the subsequent prices up to the present.")

with stat_cols[1]:
    current_target_ui = st.empty()
    with current_target_ui.container():
        st.write("Prediction (Target) awaiting...")
        st.text(f"Prediction triggers every {targeting_period}m")


# Create a placeholder for targets grid
targets_grid = st.empty()

# Initialize a separate DataFrame to store all target positions
st.session_state["targets_df"] = pd.DataFrame(columns=['time', 'price', 'label'])

# Initialize a variable to keep track of the last target's timestamp
last_target_time = None
last_time = time.time()  # Record the current time
i = 0
while True:
# Main loop to process data points
    print(i)
    # Get a subset of the most recent data (last 30 minutes)
    if i > len(usdjpy_data):
        print("getting...", time.time() - last_time)
        if time.time() - last_time >= 60:  # 60 seconds = 1 minute
            print("fetching...")
            usdjpy_data = fetch_usdjpy_data("1m")
            usdjpy_data = usdjpy_data.iloc[max(0, len(usdjpy_data) - visible_minutes):len(usdjpy_data) + 1]
            last_time = time.time()
        else:
            time.sleep(1)
            continue
    else:
        i += 1
        
    data_to_plot = usdjpy_data.iloc[max(0, i - visible_minutes):i + 1]

    # Retrieve the last visible point (current point for visualization)
    if len(data_to_plot) > 0:
        recent_time = data_to_plot.iloc[-1]['time']
    else:
        continue

    # Ensure recent_time is a scalar timestamp
    if isinstance(recent_time, pd.Series):
        recent_time = recent_time.iloc[0]

    # Check if this is the first iteration or if n minutes have passed since the last added target
    if datetime.now().astimezone(pytz.timezone('CET')) < recent_time.astimezone(pytz.timezone('CET')) + timedelta(minutes=1) and (last_target_time is None or (recent_time - last_target_time) >= timedelta(minutes=targeting_period)):
        with score_ui.container():
            if st.session_state["targets_df"]["time"].count() > 0:
                all_done_count = 0
                all_count = 0
                for index, item in st.session_state["targets_df"].iterrows():
                    target_time = item["time"]
                    target_time = target_time.astimezone(pytz.timezone('CET')) - timedelta(minutes=targeting_period)
                    target_price = round(float(item["price"]), 3)
                    min_max = fetch_today_max_min(target_time.strftime('%H:%M'))
                    all_count += 1
                    if target_price >= min_max[0] and target_price <= min_max[1]:
                        all_done_count += 1

                st.metric("AI Agent Performance Score", f"{round((all_done_count / all_count) * 100, 2)}%", help="The performance indicator is calculated by comparing the price valid at the time of calculation with the subsequent prices up to the present.")

        dfNm = fetch_training_data(f"{targeting_period}m")
        target_x = recent_time + timedelta(minutes=targeting_period)
        with current_target_ui.container():
            with st.spinner("Next target calculation..."):
                target_y = pm.predict(dfNm)

        st.session_state["current_target"] = {"time": target_x, "rate": target_y}
        with current_target_ui.container():
            st.metric("AI Agent Prediction (Last Target)", target_y)

        # Generate the target label
        target_x = target_x.astimezone(pytz.timezone('CET'))
        target_label = f"Target ({target_x.strftime('%Y-%m-%d %H:%M')}, {target_y})"

        # Add the new target to the `targets_df`
        st.session_state["targets_df"] = pd.concat([st.session_state["targets_df"], pd.DataFrame([{'time': target_x, 'price': target_y, 'label': target_label}])])

        # Update the last target's timestamp
        last_target_time = recent_time

        # Debugging: Print calculated range and target
        print(f"x_min: {data_to_plot['time'].min()}, x_max: {data_to_plot['time'].max()}, target_x: {recent_time}")

        with targets_grid.container(border=False):
            st.write("Current targets")
            st.write(st.session_state["targets_df"])


    # Define the x-axis range (extend to ensure target inclusion)
    x_min = data_to_plot['time'].min()
    x_max = max(data_to_plot['time'].max(), recent_time + timedelta(minutes=2))  # Extend padding

    # Ensure x_min and x_max are scalar timestamps
    if isinstance(x_min, pd.Series):
        x_min = x_min.iloc[0]
    if isinstance(x_max, pd.Series):
        x_max = x_max.iloc[0]

    #y_scale = calculate_y_scale(data_to_plot)
    y_min = data_to_plot["low"].min()
    y_max = data_to_plot["high"].max()

    if "targets_df" in st.session_state and st.session_state["targets_df"]["time"].count() > 0:
        y_min = min(y_min, st.session_state["targets_df"]["price"].min())
        y_max = max(y_max, st.session_state["targets_df"]["price"].max())
    
    # Create a line chart with the last 30 minutes
    line_chart = alt.Chart(data_to_plot).transform_fold(
        fold=['low', 'high'],  # Specify the columns to "fold" into a key-value pair
        as_=['Type', 'Value']  # Converts columns into 'Type' and 'Value'
        ).mark_line(point=True).encode(
            x=alt.X('time:T', title='Time', scale=alt.Scale(domain=(x_min, x_max))),
            y=alt.Y('Value:Q', title='USD/JPY', scale=alt.Scale(domain=(y_min, y_max))),
            color=alt.Color('Type:N', title='Data Type'),  # Color lines based on 'low' & 'high'
            tooltip=['time:T', 'Type:N', 'Value:Q']  # Tooltip showing the type and value
        ).properties(
            title="USD/JPY Real-Time 1-Minute Data (Last 30 Minutes)",
            width=600,
            height=400
        )

    # Add points to the chart
    target_points = alt.Chart(st.session_state["targets_df"][-6:-1]).mark_point(
        size=100,
        color='blue'
    ).encode(
        x='time:T',
        y='price:Q',
        tooltip=['time:T', 'price:Q', 'label:N']  # Add tooltips for interactivity
    )

    # Add labels to the chart
    target_labels = alt.Chart(st.session_state["targets_df"][-6:-1]).mark_text(
        dx=10,  # Shift slightly to the right
        dy=-10,  # Shift slightly above the point
        fontSize=12,
        color='blue'
    ).encode(
        x='time:T',
        y='price:Q',
        text='label:N',
        tooltip=['label:N']  # Optional: Add tooltips to the text labels if needed
    )
    
    # Combine the line chart, the target point, and the target label
    combined_chart = line_chart + target_points + target_labels

    # Render the chart in Streamlit
    chart_row.altair_chart(combined_chart, use_container_width=True)