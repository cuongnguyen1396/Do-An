import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px
# from gensim import corpora, models, similarities
# from underthesea import word_tokenize, pos_tag, sent_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# GUI
st.image('GUI_Project2/ttth.png',width=500)
st.title("Đồ Án Tốt Nghiệp Môn Data Science")
st.write("-"*60)
menu = ["Đồ án", "Giới thiệu", "Tìm kiếm thông tin theo Hotel id", "Tìm kiếm thông tin theo Riviwer Id",
         "Tìm kiếm thông tin theo Riviwer Name", "Thống kê"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Đồ án':
    st.write("## Recommendation System")
    st.write("-"*60)
    st.write('### Học viên: Nguyễn Văn Cường, Nguyễn Quí Hiển')
    st.write("-"*60)    
    st.write('#### Xây dựng hệ thống đề xuất để hỗ trợ người dùng nhanh chóng chọn được nơi lưu trú phù hợp trên Agoda ')
    st.write('#### Hệ thống sẽ gồm hai mô hình gợi ý chính:')
    # st.write('##### Content-based filtering')
    # st.write('##### Collaborative filtering')
    st.image('GUI_Project2/filtering.png')
elif choice == 'Giới thiệu':
    st.subheader("Angoda")
    st.image("GUI_Project2/angoda_1.jpg")
    st.write("""
    #####  Agoda là một trang web đặt phòng trực tuyến có trụ sở tại Singapore, được thành lập vào năm 2005, thuộc sở hữu của Booking Holdings Inc,.
    #####  Agoda chuyên cung cấp dịch vụ đặt phòng khách sạn, căn hộ, nhà nghỉ và các loại hình lưu trú trên toàn cầu. Trang web này cho phép người dùng tìm kiếm, so sánh và đặt chỗ ở với mức giá ưu đãi.         
    """)
    st.write("-"*60)
    st.write("""##### Bạn đã có khách sạn cần đặt chưa ?""")
    st.write("""##### Nếu chưa bạn có thể xem các gợi ý ở phần Menu.""")
elif choice == 'Tìm kiếm thông tin theo Hotel id':
    st.subheader("Xem thông tin khách sạn theo mã Hotel_ID")

    # # function cần thiết
    def get_recommendations(df, hotel_id, cosine_sim, nums=5):
        # Get the index of the hotel that matches the hotel_id
        matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
        if not matching_indices:
            print(f"No hotel found with ID: {hotel_id}")
            return pd.DataFrame()  # Return an empty DataFrame if no match
        idx = matching_indices[0]

        # Get the pairwise similarity scores of all hotels with that hotel
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the hotels based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
        sim_scores = sim_scores[1:nums+1]

        # Get the hotel indices
        hotel_indices = [i[0] for i in sim_scores]

        # Return the top n most similar hotels as a DataFrame
        return df.iloc[hotel_indices]

    # Hiển thị đề xuất ra bảng
    def display_recommended_hotels(recommended_hotels, cols=5):
        for i in range(0, len(recommended_hotels), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(recommended_hotels):
                    hotel = recommended_hotels.iloc[i + j]
                    with col:   
                        st.write(hotel['Hotel_Name'])                    
                        expander = st.expander(f"Description")
                        hotel_description = hotel['Hotel_Description']
                        truncated_description = ' '.join(hotel_description.split()[:100]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

    # # Đọc dữ liệu khách sạn
    df_hotels = pd.read_csv('GUI_Project2/hotel_info.csv')
    # Lấy 10 khách sạn
    # random_hotels = df_hotels.head(n=10)
    # print(random_hotels)

    st.session_state.random_hotels = df_hotels

    # Open and read file to cosine_sim_new
    with open('GUI_Project2/cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    ###### Giao diện Streamlit ######
    st.image('GUI_Project2/hotel.jpg', use_column_width=True)

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])

            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_recommendations(df_hotels, st.session_state.selected_hotel_id, cosine_sim=cosine_sim_new, nums=3) 
            display_recommended_hotels(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
elif choice == 'Tìm kiếm thông tin theo Riviwer Id':
    st.subheader("Tìm kiếm thông tin theo Riviwer Id (Surprise)")
    # Đọc dữ liệu 'hotel_info.csv'
    data_info = pd.read_csv('GUI_Project2/hotel_info.csv')
    # Đọc dữ liệu 'data_indexed.csv' đã xử lý tù file 'hotel_comments.csv'
    data_indexed = pd.read_csv('GUI_Project2/data_indexed.csv')
    st.write('#### Bảng thông tin khách sạn và riviewer')
    data_indexed_15=data_indexed.head(15)
    st.dataframe(data_indexed_15)

    # Load  model
    with open('GUI_Project2/surprise.pkl', 'rb') as file:  
        algorithm = pickle.load(file)
    st.write('#### Nhập ID của Khách sạn (Vd: 1, 30, 550, 1735,...)')
    Reviewer_id = st.text_input("Hãy nhập Id khách sạn")
    # Reviewer_id = int(Reviewer_id)
    if Reviewer_id:
        try:
            Reviewer_id = int(Reviewer_id)
            if Reviewer_id in data_indexed['Reviewer_id'].to_list():
                # st.write("Reviewer id is", Reviewer_id)
                df_score = data_indexed[["Hotel_id", "Hotel ID"]]
                df_score['EstimateScore'] = df_score['Hotel_id'].apply(lambda x: algorithm.predict(Reviewer_id, x).est) # est: get EstimateScore
                df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
                df_score = df_score.drop_duplicates()
                df_score = df_score[df_score['EstimateScore'] >= 9.5]
                # st.dataframe(df_score)
                df_recommendations = pd.merge(df_score, data_info, left_on='Hotel ID', right_on='Hotel_ID', how='left')
                df_recommendations = df_recommendations.dropna(subset=['Hotel_Name'])
                df_recommendations = df_recommendations[['Hotel ID','EstimateScore','Hotel_Name','Hotel_Address']].head(10)
                st.write('##### Đây là danh sách các khách sạn có điểm ước lượng cao của khách hàng có Riviewer id:',str(Reviewer_id))
                if Reviewer_id:
                    st.table(df_recommendations)
            else:
                st.write("Tên không tồn tại")
        except ValueError:
            st.write("Vui lòng nhập một số hợp lệ cho Reviewer ID")
elif choice == 'Tìm kiếm thông tin theo Riviwer Name':
    st.subheader("Tìm kiếm thông tin theo tên Reviewer Name (Surprise)")

    # Đọc dữ liệu 'hotel_info.csv'
    data_info = pd.read_csv('GUI_Project2/hotel_info.csv')
    # st.dataframe(data_info)
    # Đọc dữ liệu 'data_indexed.csv' đã xử lý tù file 'hotel_comments.csv'
    data_indexed = pd.read_csv('GUI_Project2/data_indexed.csv')
    st.write('#### Bảng thông tin Id của khách sạng và Riviewer')
    data_indexed_15=data_indexed.head(15)
    st.dataframe(data_indexed_15)

    # Load  model
    with open('GUI_Project2/surprise.pkl', 'rb') as file:  
        algorithm = pickle.load(file)
    
    st.write('#### Nhập ID của Khách sạn (Vd: MARIKO_1, Dang_1, Dang_2, Dieu_1, Minh_2,...)')
    reviewer_name = st.text_input("Hãy nhập tên")
    if reviewer_name:
        try:
            if reviewer_name in data_indexed['Reviewer Name'].to_list():
                # st.write("Reviewer name is", reviewer_name)
                df_score = data_indexed[["Hotel_id", "Hotel ID", "Reviewer Name","Reviewer_id"]][data_indexed['Reviewer Name']==reviewer_name]
                Reviewer_id = df_score['Reviewer_id'].iloc[0]
                df_score['EstimateScore'] = df_score['Hotel_id'].apply(lambda x: algorithm.predict(Reviewer_id, x).est) # est: get EstimateScore
                df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
                df_score = df_score.drop_duplicates()
                df_score = df_score[df_score['EstimateScore'] >= 9.5]
                # st.dataframe(df_score)
                df_recommendations = pd.merge(df_score, data_info, left_on='Hotel ID', right_on='Hotel_ID', how='left')
                df_recommendations = df_recommendations.dropna(subset=['Hotel_Name'])
                df_recommendations = df_recommendations[['Hotel ID','Reviewer Name','EstimateScore','Hotel_Name','Hotel_Address']]
                st.write('##### Đây là danh sách các khách sạn có điểm ước lượng cao của khách hàng có Riviewer id:',reviewer_name)
                if reviewer_name:
                    st.table(df_recommendations)

            else:
                st.write("Tên không tồn tại")
        except ValueError:
            st.write("Vui lòng nhập một số hợp lệ cho Reviewer ID")
elif choice == 'Thống kê':
    st.write("# Biểu đồ thống kê")
    #  Đọc dữ liệu 'hotel_comments.csv'
    data_comments = pd.read_csv('GUI_Project2/data_comments_new.csv')
    # Lấy 10 khách sạn
    # random_hotels = data_comments.head(n=10)
    # st.write('#### Bảng thông tin số liệu khách sạn')
    # st.dataframe(random_hotels)
    # Trưc quan
    # Month
    st.header('Biểu đồ số lượng phân bố khách du lịch theo tháng')
    fig_m = plt.figure(figsize=(15, 6))
    sns.countplot(data=data_comments,x='Month')
    # plt.title('Biểu đồ số lượng phân bố khách du lịch theo tháng')
    plt.xlabel('Tháng')
    plt.ylabel('Số lượng khách du lịch')
    plt.xticks(rotation=0)
    st.pyplot(fig_m)

    # Year
    st.header('Biểu đồ số lượng phân bố khách du lịch theo năm')
    fig_y = plt.figure(figsize=(15, 6))
    # plt.title('Biểu đồ số lượng phân bố khách du lịch theo năm')  
    sns.countplot(data=data_comments,x='Year')
    plt.xlabel('Năm')
    plt.ylabel('Số lượng khách du lịch')
    plt.xticks(rotation=70)
    st.pyplot(fig_y)

    # lưu trú
    st.header('Biểu đồ số lượng phân bố khách du lịch theo hình thức lưu trú')
    fig_l=plt.figure(figsize=(15, 6))
    sns.countplot(data=data_comments,x='Stay Details')  
    plt.title('Biểu đồ Stay Details')
    plt.xlabel('Stay Details')
    plt.ylabel('Số lượng khách')
    plt.xticks(rotation=0)
    st.pyplot(fig_l)
    # Nhóm khách hàng
    st.header('Biểu đồ số lượng phân bố khách du lịch theo nhóm khách hàng')
    fig_nhom=plt.figure(figsize=(15, 6))
    sns.countplot(data=data_comments,x='Group Name')  

    plt.title('Biểu đồ Group Name')
    plt.xlabel('Group Name')
    plt.ylabel('Số lượng khách')
    plt.xticks(rotation=30)
    st.pyplot(fig_nhom)

    # # quốc tich
    # st.header('Biểu đồ số lượng phân bố khách du lịch theo quốc tịch')
    # fig_n=plt.figure(figsize=(15, 6))
    # sns.countplot(data=data_comments,x='Nationality')  

    # plt.title('Biểu đồ Group Name')
    # plt.xlabel('Group Name')
    # plt.ylabel('Số lượng khách')
    # plt.xticks(rotation=30)
    # st.pyplot(fig_n)

    # Điểm đánh giá
    st.header('Biểu đồ số lượng phân bố khách du lịch theo điểm đánh giá')
    fig_s=plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(data_comments['Score'], kde=True, color="purple", binwidth=0.5)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig_s)




    


####################################################
