import re
import pandas as pd

def preprocess(data):
    # ✅ Robust pattern for Android + iOS + different locales
    pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4},\s\d{1,2}:\d{2}.*?-'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({
        'user_message': messages,
        'date': dates
    })

    # ✅ Robust datetime parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ DROP rows where date could not be parsed
    df = df.dropna(subset=['date'])

    # ---------------- USER & MESSAGE SPLIT ----------------
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # ---------------- DATE FEATURES (CRITICAL FOR PLOTS) ----------------
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # ---------------- PERIOD (FOR HEATMAPS) ----------------
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour}-{hour + 1}")

    df['period'] = period

    return df
