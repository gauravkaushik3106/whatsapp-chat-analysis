import re
import pandas as pd

def preprocess(data):
    # Normalize line endings
    data = data.replace('\r\n', '\n')

    # EXACT WhatsApp Android format (your file)
    pattern = r'\d{2}/\d{2}/\d{2},\s\d{2}:\d{2}\s-\s'

    dates = re.findall(pattern, data)
    messages = re.split(pattern, data)[1:]

    # If parsing fails, return empty df safely
    if len(dates) == 0 or len(messages) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'date': dates,
        'user_message': messages
    })

    # Convert date (24-hour supported)
    df['date'] = pd.to_datetime(
        df['date'],
        format='%d/%m/%y, %H:%M - ',
        errors='coerce'
    )

    df = df.dropna(subset=['date'])

    users = []
    messages = []

    for msg in df['user_message']:
        split = re.split(r':\s', msg, maxsplit=1)
        if len(split) == 2:
            users.append(split[0])
            messages.append(split[1])
        else:
            users.append('group_notification')
            messages.append(msg)

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Date features (required by helper.py)
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Period column for heatmap
    df['period'] = df['hour'].apply(
        lambda h: f"{h}-00" if h == 23 else f"{h}-{h+1}"
    )

    return df
