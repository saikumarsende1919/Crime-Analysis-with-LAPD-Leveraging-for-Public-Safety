import streamlit as st
import pandas as pd
from sklearn.ensemble import  GradientBoostingClassifier

# Load your dataset
@st.cache_data()
def load_data():
    return pd.read_csv("newCrimeData_from_2020_to_Present.csv")  # Replace "cleannedandencoded.csv" with the path to your dataset

df = load_data()

# Define the AREA NAME mapping
area_mapping = {
    1: 'Central',
    2: 'Rampart',
    3: 'Southwest',
    4: 'Hollenbeck',
    5: 'Harbor',
    6: 'Hollywood',
    7: 'Wilshire',
    8: 'West LA',
    9: 'Van Nuys',
    10: 'West Valley',
    11: 'Northeast',
    12: '77th Street',
    13: 'Newton',
    14: 'Pacific',
    15: 'N Hollywood',
    16: 'Foothill',
    17: 'Devonshire',
    18: 'Southeast',
    19: 'Mission',
    20: 'Olympic',
    21: 'Topanga'
}

# Define the WEAPON DESC mapping
weapon_mapping = {
    101: 'REVOLVER',
    102: 'HAND GUN',
    103: 'RIFLE',
    104: 'SHOTGUN',
    105: 'SAWED OFF RIFLE/SHOTGUN',
    106: 'UNKNOWN FIREARM',
    107: 'OTHER FIREARM',
    108: 'AUTOMATIC WEAPON/SUB-MACHINE GUN',
    109: 'SEMI-AUTOMATIC PISTOL',
    110: 'SEMI-AUTOMATIC RIFLE',
    111: 'STARTER PISTOL/REVOLVER',
    112: 'TOY GUN',
    113: 'SIMULATED GUN',
    114: 'AIR PISTOL/REVOLVER/RIFLE/BB GUN',
    115: 'ASSAULT WEAPON/UZI/AK47/ETC',
    116: 'ANTIQUE FIREARM',
    117: 'UNK TYPE SEMIAUTOMATIC ASSAULT RIFLE',
    118: 'UZI SEMIAUTOMATIC ASSAULT RIFLE',
    119: 'MAC-10 SEMIAUTOMATIC ASSAULT WEAPON',
    120: 'MAC-11 SEMIAUTOMATIC ASSAULT WEAPON',
    121: 'HECKLER & KOCH 91 SEMIAUTOMATIC ASSAULT RIFLE',
    122: 'HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE',
    123: 'M1-1 SEMIAUTOMATIC ASSAULT RIFLE',
    124: 'M-14 SEMIAUTOMATIC ASSAULT RIFLE',
    125: 'RELIC FIREARM',
    200: 'KNIFE WITH BLADE 6INCHES OR LESS',
    201: 'KNIFE WITH BLADE OVER 6 INCHES IN LENGTH',
    202: 'BOWIE KNIFE',
    203: 'DIRK/DAGGER',
    204: 'FOLDING KNIFE',
    205: 'KITCHEN KNIFE',
    206: 'SWITCH BLADE',
    207: 'OTHER KNIFE',
    208: 'RAZOR',
    209: 'STRAIGHT RAZOR',
    210: 'RAZOR BLADE',
    211: 'AXE',
    212: 'BOTTLE',
    213: 'CLEAVER',
    214: 'ICE PICK',
    215: 'MACHETE',
    216: 'SCISSORS',
    217: 'SWORD',
    218: 'OTHER CUTTING INSTRUMENT',
    219: 'SCREWDRIVER',
    220: 'SYRINGE',
    221: 'GLASS',
    223: 'UNKNOWN TYPE CUTTING INSTRUMENT',
    300: 'BLACKJACK',
    301: 'BELT FLAILING INSTRUMENT/CHAIN',
    302: 'BLUNT INSTRUMENT',
    303: 'BRASS KNUCKLES',
    304: 'CLUB/BAT',
    305: 'FIXED OBJECT',
    306: 'ROCK/THROWN OBJECT',
    307: 'VEHICLE',
    308: 'STICK',
    309: 'BOARD',
    310: 'CONCRETE BLOCK/BRICK',
    311: 'HAMMER',
    312: 'PIPE/METAL PIPE',
    400: 'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)',
    500: 'UNKNOWN WEAPON/OTHER WEAPON',
    501: 'BOMB THREAT',
    502: 'BOW AND ARROW',
    503: 'CAUSTIC CHEMICAL/POISON',
     504: 'DEMAND NOTE',
    505: 'EXPLOXIVE DEVICE',
    506: 'FIRE',
    507: 'LIQUOR/DRUGS',
    508: 'MARTIAL ARTS WEAPONS',
    509: 'ROPE/LIGATURE',
    510: 'SCALDING LIQUID',
    511: 'VERBAL THREAT',
    512: 'MACE/PEPPER SPRAY',
    513: 'STUN GUN',
    514: 'TIRE IRON',
    515: 'PHYSICAL PRESENCE',
    516: 'DOG/ANIMAL (SIC ANIMAL ON)'
}

# Define the Victim's Descent mapping
victim_descent_mapping = {
    'Other': {'label': 'O', 'code': 12},
    'Unknown': {'label': 'X', 'code': 18},
    'Hispanic/Latin/Mexican': {'label': 'H', 'code': 7},
    'Black': {'label': 'B', 'code': 2},
    'White': {'label': 'W', 'code': 17},
    'Other Asian': {'label': 'A', 'code': 1},
    'Chinese': {'label': 'C', 'code': 3},
    'Korean': {'label': 'K', 'code': 10},
    'Japanese': {'label': 'J', 'code': 9},
    'Filipino': {'label': 'F', 'code': 5},
    'American Indian/Alaskan Native': {'label': 'I', 'code': 8},
    'Vietnamese': {'label': 'V', 'code': 16},
    'Samoan': {'label': 'S', 'code': 14},
    'Pacific Islander': {'label': 'P', 'code': 13},
    'Asian Indian': {'label': 'Z', 'code': 19},
    'Guamanian': {'label': 'G', 'code': 6},
    'Hawaiian': {'label': 'U', 'code': 15},
    'Cambodian': {'label': 'D', 'code': 4},
    'Laotian': {'label': 'L', 'code': 11},
    'Unknown': {'label': '-', 'code': 0}  # For handling missing values
}

# Define the Streamlit app
def main():
    st.title("Crime Prediction App")

    # Input values for selected columns
    st.subheader("Input Values for Prediction")

    # Given values for the columns specified in colsused
    st.write("Please enter the following information:")

    crm_cd = st.number_input("Crime Code")
    vict_age = st.number_input("Victim's Age")
    vict_sex = st.selectbox("Victim's Sex", ['Female (F)', 'Male (M)', 'Unknown (X)'])
    vict_descent_label = st.selectbox("Victim's Descent", list(victim_descent_mapping.keys()))
    theft = st.radio("Theft Occurred?", ['Yes', 'No'])
    vehicular = st.radio("Vehicular Incident?", ['Yes', 'No'])
    lat = st.number_input("Latitude")
    lon = st.number_input("Longitude")

    # Map the selected victim's sex to the desired values for prediction
    sex_mapping = {'Female (F)': 0, 'Male (M)': 2, 'Unknown (X)': 1}
    input_values = {
        'Crm Cd': crm_cd,
        'Vict Age': vict_age,
        'Vict Sex': sex_mapping[vict_sex],
        'Vict Descent': victim_descent_mapping[vict_descent_label]['code'],  # Using the code here
        'theft': 1 if theft == 'Yes' else 0,
        'vehicular': 1 if vehicular == 'Yes' else 0,
        'LAT': lat,
        'LON': lon
    }

    input_df = pd.DataFrame([input_values])

    # Predict button
    if st.button('Predict'):
        # Predict for crime_type
        st.subheader("Predicted Output")
        st.write("Predicted Crime Type:")
        with st.spinner('Predicting...'):
            prediction = predict_output(input_df, 'crime_type')
            crime_type = "Violent" if prediction[0] == 1 else "Non-Violent"
            st.write(crime_type)

        # Predict for AREA
        st.write("Predicted Area:")
        with st.spinner('Predicting...'):
            area_prediction = predict_output(input_df, 'AREA')
            area_name = area_mapping.get(area_prediction[0], 'Unknown')
            st.write(f"{area_prediction[0]} - {area_name}")

        # Predict for Weapon Used Cd
        st.write("Predicted Weapon Used Code:")
        with st.spinner('Predicting...'):
            weapon_prediction = predict_output(input_df, 'Weapon Used Cd')
            weapon_name = weapon_mapping.get(weapon_prediction[0], 'Unknown')
            st.write(f"{weapon_prediction[0]} - {weapon_name}")

def predict_output(input_df, output_col):
    X = df[input_df.columns]
    y = df[output_col]

    # Model Training
    model =  GradientBoostingClassifier()  # You can choose any other classifier
    model.fit(X, y)

    # Prediction
    return model.predict(input_df)

if __name__ == "__main__":
    main()
