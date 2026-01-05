import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim

st.set_page_config(page_title="Uber Fare Predictor", layout="wide")


@st.cache_resource
def load_model():
    """Loads the trained model from the pickle file."""
    try:
        with open("model.pkr", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'model.pkr' not found. Please ensure it is in the same directory."
        )
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_coords(address):
    """Geocodes an address to (latitude, longitude)."""
    try:
        geolocator = Nominatim(user_agent="uber_fare_app_v1")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None


NYC_LOCATIONS = {
    "Select a location...": None,
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Times Square": (40.7580, -73.9855),
    "Empire State Building": (40.7488, -73.9857),
    "Central Park": (40.7851, -73.9683),
    "One World Trade Center": (40.7127, -74.0134),
    "Chrysler Building": (40.7516, -73.9755),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Custom Address": "CUSTOM",
    "Manual Coordinates": "MANUAL",
}


def main():
    st.title("🚖 Uber Fare Predictor")
    st.markdown("Predict your Uber fare in NYC based on location and time.")

    model = load_model()
    if not model:
        return

    # User Inputs
    with st.container():
        col1, col2 = st.columns(2)

        # Pickup Location
        with col1:
            st.subheader("📍 Pickup")
            pickup_option = st.selectbox(
                "Choose Pickup Location",
                list(NYC_LOCATIONS.keys()),
                key="pickup_select",
            )

            pickup_lat, pickup_lon = None, None

            if pickup_option == "Custom Address":
                pickup_address = st.text_input(
                    "Enter Pickup Address", key="pickup_custom"
                )
                if pickup_address:
                    with st.spinner("Finding pickup location..."):
                        pickup_lat, pickup_lon = get_coords(pickup_address)
                        if pickup_lat is None:
                            st.warning(
                                "Could not find pickup address. Please try again."
                            )

            elif pickup_option == "Manual Coordinates":
                c1, c2 = st.columns(2)
                with c1:
                    pickup_lat = st.number_input(
                        "Pickup Latitude",
                        value=40.7128,
                        format="%.6f",
                        key="pickup_lat_input",
                    )
                with c2:
                    pickup_lon = st.number_input(
                        "Pickup Longitude",
                        value=-74.0060,
                        format="%.6f",
                        key="pickup_lon_input",
                    )

            elif pickup_option and pickup_option != "Select a location...":
                pickup_lat, pickup_lon = NYC_LOCATIONS[pickup_option]

            # Display Coordinates
            if pickup_lat is not None and pickup_lon is not None:
                st.info(f"**Pickup Coordinates:** {pickup_lat:.6f}, {pickup_lon:.6f}")

        # Dropoff Location
        with col2:
            st.subheader("🏁 Dropoff")
            dropoff_option = st.selectbox(
                "Choose Dropoff Location",
                list(NYC_LOCATIONS.keys()),
                index=0,
                key="dropoff_select",
            )

            dropoff_lat, dropoff_lon = None, None

            if dropoff_option == "Custom Address":
                dropoff_address = st.text_input(
                    "Enter Dropoff Address", key="dropoff_custom"
                )
                if dropoff_address:
                    with st.spinner("Finding dropoff location..."):
                        dropoff_lat, dropoff_lon = get_coords(dropoff_address)
                        if dropoff_lat is None:
                            st.warning(
                                "Could not find dropoff address. Please try again."
                            )

            elif dropoff_option == "Manual Coordinates":
                c3, c4 = st.columns(2)
                with c3:
                    dropoff_lat = st.number_input(
                        "Dropoff Latitude",
                        value=40.7128,
                        format="%.6f",
                        key="dropoff_lat_input",
                    )
                with c4:
                    dropoff_lon = st.number_input(
                        "Dropoff Longitude",
                        value=-74.0060,
                        format="%.6f",
                        key="dropoff_lon_input",
                    )

            elif dropoff_option and dropoff_option != "Select a location...":
                dropoff_lat, dropoff_lon = NYC_LOCATIONS[dropoff_option]

            # Display Coordinates
            if dropoff_lat is not None and dropoff_lon is not None:
                st.info(
                    f"**Dropoff Coordinates:** {dropoff_lat:.6f}, {dropoff_lon:.6f}"
                )

    st.markdown("---")

    # Date, Time, Passengers
    with st.container():
        col3, col4, col5 = st.columns(3)
        with col3:
            ride_date = st.date_input("Date", datetime.now())
        with col4:
            ride_time = st.time_input("Time", datetime.now())
        with col5:
            passengers = st.number_input(
                "Passenger Count", min_value=1, max_value=6, value=1
            )

    # Prediction Button
    if st.button("Estimate Fare", type="primary", use_container_width=True):
        if pickup_lat is None or dropoff_lat is None:
            st.error("Please select valid Pickup and Dropoff locations.")
        else:
            # Prepare data for model
            # Model expects: pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, hour, day_of_week

            # Combine date and time
            ride_datetime = datetime.combine(ride_date, ride_time)
            hour = ride_datetime.hour
            day_of_week = ride_datetime.weekday()  # Monday is 0, Sunday is 6

            input_data = pd.DataFrame(
                {
                    "pickup_longitude": [pickup_lon],
                    "pickup_latitude": [pickup_lat],
                    "dropoff_longitude": [dropoff_lon],
                    "dropoff_latitude": [dropoff_lat],
                    "passenger_count": [passengers],
                    "hour": [hour],
                    "day_of_week": [day_of_week],
                }
            )

            try:
                prediction = model.predict(input_data)
                fare = prediction[0]

                st.markdown(
                    f"""
                <div style="text-align: center; padding: 5px; background-color: transparent; border-radius: 10px; margin-top: 5px;">
                    <h3>Estimated Fare</h3>
                    <h1 style="color: #00CC00;">${fare:.2f}</h1>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.write("Debug info - Input Data:")
                st.write(input_data)


if __name__ == "__main__":
    main()
