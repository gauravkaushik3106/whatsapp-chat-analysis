import re
import pandas as pd

def preprocess(data):
    # ✅ EXACT pattern for your WhatsApp format
    pattern = r'\d{2}/\d{2}/\d{2},\s\d{2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Safety check
    if len(messages) == 0 or len(dates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'user_message': messages,
        'date': dates
    })

    # ✅ Robust datetime parsing (24-hour supported)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date'])

    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # ✅ Date features (needed for all plots)
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # ✅ Period column for heatmap
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("23-00")
        else:
            period.append(f"{hour}-{hour + 1}")

    df['period'] = period

    return df
