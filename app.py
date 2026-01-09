import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("Whatsapp Chat Analyzer")

# -----------------------------------------------
# FILE UPLOAD
# -----------------------------------------------
uploaded_file = st.sidebar.file_uploader("Choose WhatsApp chat (.txt)")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # WhatsApp Android exports are usually UTF-16
    try:
        data = bytes_data.decode("utf-16")
    except UnicodeDecodeError:
        data = bytes_data.decode("utf-8", errors="ignore")

    df = preprocessor.preprocess(data)

    if df.empty:
        st.error("Invalid WhatsApp chat format or empty chat file.")
        st.stop()

    # -----------------------------------------------
    # USER SELECTION
    # -----------------------------------------------
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis for", user_list)

    if st.sidebar.button("Show Analysis"):

        # ===============================================
        # TOP STATISTICS
        # ===============================================
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(
            selected_user, df
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", num_messages)
        col2.metric("Words", words)
        col3.metric("Media", num_media_messages)
        col4.metric("Links", num_links)

        # ===============================================
        # MONTHLY TIMELINE
        # ===============================================
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ===============================================
        # DAILY TIMELINE
        # ===============================================
        st.title("Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily['only_date'], daily['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # ===============================================
        # ACTIVITY MAP
        # ===============================================
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

        # ===============================================
        # WEEKLY HEATMAP
        # ===============================================
        st.title("Weekly Activity Map")
        heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(heatmap, ax=ax)
        st.pyplot(fig)

        # ===============================================
        # WORDCLOUD
        # ===============================================
        st.title("Wordcloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        # ===============================================
        # EMOJI ANALYSIS
        # ===============================================
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

        # ==================================================
        # 🔥 EMOTION INTENSITY & CONVERSATION EVENTS (FINAL)
        # ==================================================
        st.title("Emotion Intensity & Conversation Events")

        emotion_df = helper.compute_emotion_intensity(selected_user, df)
        events_df = helper.detect_emotional_events(emotion_df)

        # -------- Z-SCORE EMOTION INTENSITY PLOT --------
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            emotion_df['hour_block'],
            emotion_df['z_intensity'],
            color='black',
            label='Emotion Deviation (Z-score)'
        )

        ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='High Emotional Event')
        ax.axhline(-2, color='green', linestyle='--', alpha=0.5, label='Low Emotional Phase')

        ax.set_xlabel("Time")
        ax.set_ylabel("Z-score Intensity")
        ax.legend()
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # -------- COLORED EVENTS TABLE --------
        st.subheader("Detected Emotional Events")

        def highlight_events(row):
            if "Negative" in row["event_type"]:
                return ["background-color: #ffcccc"] * len(row)
            else:
                return ["background-color: #ccffcc"] * len(row)

        if not events_df.empty:
            styled_df = events_df.style.apply(highlight_events, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No statistically significant emotional events detected.")
