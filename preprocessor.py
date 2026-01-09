import re
import pandas as pd

def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)
    dates = re.findall(pattern, data)

    if len(dates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        'message': messages[1:],
        'date': dates
    })

    df['date'] = pd.to_datetime(
        df['date'],
        format='%d/%m/%y, %H:%M - ',
        errors='coerce'
    )

    df = df.dropna(subset=['date'])

    users = []
    texts = []

    for message in df['message']:
        message = message.strip()

        # SAFE split (handles emoji usernames)
        if ": " in message:
            user, text = message.split(": ", 1)
            users.append(user)
            texts.append(text)
        else:
            users.append("group_notification")
            texts.append(message)

    df['user'] = users
    df['message'] = texts

    # Date features
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Hour period
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append("23-00")
        elif hour == 0:
            period.append("00-01")
        else:
            period.append(f"{hour}-{hour+1}")

    df['period'] = period

    return df
