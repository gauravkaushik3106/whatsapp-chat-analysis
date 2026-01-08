import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose WhatsApp chat (.txt)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    try:
        data = bytes_data.decode("utf-16")
    except UnicodeDecodeError:
        data = bytes_data.decode("utf-8", errors="ignore")

    df = preprocessor.preprocess(data)

    if df.empty:
        st.error("Invalid WhatsApp chat format or empty chat file.")
        st.stop()

    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # ---------- TOP STATS ----------
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Media", num_media_messages)
        col4.metric("Links", num_links)

        # ---------- MONTHLY TIMELINE ----------
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ---------- DAILY TIMELINE ----------
        st.title("Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily['only_date'], daily['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ---------- ACTIVITY MAP ----------
        st.title("Activity Map")
        col1, col2 = st.columns(2)

        with col1:
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values)
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # ---------- WEEKLY HEATMAP ----------
        st.title("Weekly Activity Map")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)
        st.pyplot(fig)

        # ---------- WORDCLOUD ----------
        st.title("Wordcloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ---------- EMOJI ANALYSIS ----------
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(
                emoji_df[1].head(),
                labels=emoji_df[0].head(),
                autopct="%0.2f"
            )
            st.pyplot(fig)

        # ======================================================
        # 🔥 CONVERSATION MOOD SHIFT DETECTION (OPTION 3)
        # ======================================================
        st.title("Conversation Mood Shifts")

        hourly_df = helper.hourly_sentiment(selected_user, df)
        mood_shifts = helper.detect_mood_shifts(hourly_df)

        fig, ax = plt.subplots()
        ax.plot(hourly_df['hour_block'], hourly_df['smooth_sentiment'], label='Mood Trend')
        ax.axhline(0, linestyle='--', alpha=0.3)
        plt.xticks(rotation=90)
        ax.legend()
        st.pyplot(fig)

        if not mood_shifts.empty:
            st.subheader("Detected Mood Shift Events")
            st.dataframe(mood_shifts)
        else:
            st.info("No significant mood shifts detected.")
