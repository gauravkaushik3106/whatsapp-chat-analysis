import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # ✅ SAFE decoding (fixes UnicodeDecodeError)
    try:
        data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        data = bytes_data.decode("latin-1")

    df = preprocessor.preprocess(data)

    # ✅ SAFETY CHECK: stop if dataframe is empty
    if df.empty:
        st.error("No valid messages found in this chat file.")
        st.stop()

    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list = [user for user in user_list if user != 'group_notification']
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # -------------------- TOP STATS --------------------
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)

        # -------------------- MONTHLY TIMELINE --------------------
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)

        if not timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # -------------------- DAILY TIMELINE --------------------
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)

        if not daily_timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # -------------------- ACTIVITY MAP --------------------
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            if not busy_day.empty:
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            if not busy_month.empty:
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # -------------------- WEEKLY HEATMAP (CRITICAL FIX) --------------------
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)

        if not user_heatmap.empty:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough data to generate Weekly Activity Map")

        # -------------------- MOST BUSY USERS --------------------
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)

            if not x.empty:
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

        # -------------------- WORDCLOUD --------------------
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)

        if df_wc is not None:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            st.pyplot(fig)

        # -------------------- MOST COMMON WORDS --------------------
        st.title('Most common words')
        most_common_df = helper.most_common_words(selected_user, df)

        if not most_common_df.empty:
            fig, ax = plt.subplots()
            ax.barh(most_common_df[0], most_common_df[1])
            st.pyplot(fig)

        # -------------------- EMOJI ANALYSIS --------------------
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
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
