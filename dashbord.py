import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import joblib
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fungsi untuk halaman 1
def page1():
    st.title("Walmart Sales Analysis and Prediction")
    st.image("https://s.yimg.com/ny/api/res/1.2/1b.cm2M9CF1aByGJ.KpGrg--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyMDA7aD02NzU-/https://media.zenfs.com/en/gobankingrates_644/fe092bf7f2510e24e7144b1762f401a1")
    
    st.write('Walmart adalah perusahaan ritel multinasional Amerika yang mengoperasikan jaringan hipermarket (juga disebut supercenter), department store diskon, dan toko kelontong di Amerika Serikat, yang berkantor pusat di Bentonville, Arkansas .')
    
    
    st.header("Dataset Penjualan Walmart")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    st.write(df)

    st.header('Tujuan Analisis')
    st.write('Tujuan nya adalah untuk memahami bagaimana analisis data penjualan Walmart dapat mendukung pengambilan keputusan bisnis dan strategi pemasaran yang lebih efektif.')

    st.header('Cari Ranking Teratas Srore Walmart')
    sales_per_store = df.groupby('Store')['Weekly_Sales'].sum().reset_index()

    # Mengurutkan store berdasarkan total penjualan secara menurun
    sales_per_store_sorted = sales_per_store.sort_values(by='Weekly_Sales', ascending=False)

    # Menambahkan kolom peringkat
    sales_per_store_sorted['Rank'] = range(1, len(sales_per_store_sorted) + 1)

    num_stores = st.number_input('Jumlah Store yang Ditampilkan:', min_value=1, max_value=len(sales_per_store), value=5, step=1)

    # Menampilkan rekomendasi store berdasarkan peringkat
    st.subheader(f'Rekomendasi {num_stores} Store Teratas Berdasarkan Tingkat Penjualan:')
    st.write(sales_per_store_sorted.head(num_stores))




# Fungsi untuk halaman 2
def page2():
    st.title("Walmart Sales Analysis and Prediction")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    survival_counts = df['Sales_Category'].value_counts()

    st.header('Data Distribution')

    option = st.selectbox(
    'Pilih opsi:',
    ('Unemployment', 'Weekly Sales', 'CPI', 'Fuel Price')
    )

    if option == 'Unemployment':
        fig, ax = plt.subplots()
        hist, bins = np.histogram(df['Unemployment'], bins=30)
        ax.plot(bins[:-1], hist, color='pink', linestyle='-')  # Menentukan warna dan jenis garis
        ax.set_xlabel('Unemployment')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Histogram Distribusi Data')

        # Menampilkan histogram menggunakan Streamlit
        st.pyplot(fig)

        st.caption('Line chart diatas menunjukkan distribusi data Pengganguran')
    
    elif option == 'Weekly Sales':
        fig, ax = plt.subplots()
        hist, bins = np.histogram(df['Weekly_Sales'], bins=30)
        ax.plot(bins[:-1], hist, color='pink', linestyle='-')  # Menentukan warna dan jenis garis
        ax.set_xlabel('Weekly_Sales')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Histogram Distribusi Data')

        st.pyplot(fig)
        st.caption('Line chart diatas menunjukkan distribusi data Penjualan Mingguan')

    elif option == 'CPI':
        fig, ax = plt.subplots()
        hist, bins = np.histogram(df['CPI'], bins=30)
        ax.plot(bins[:-1], hist, color='pink', linestyle='-')  # Menentukan warna dan jenis garis
        ax.set_xlabel('CPI')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Histogram Distribusi Data')

        st.pyplot(fig)
        st.caption('Line chart diatas menunjukkan distribusi data Index Harga Customer')

    elif option == 'Fuel Price':
        fig, ax = plt.subplots()
        hist, bins = np.histogram(df['Fuel_Price'], bins=30)
        ax.plot(bins[:-1], hist, color='pink', linestyle='-')  # Menentukan warna dan jenis garis
        ax.set_xlabel('Fuel_Price')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Histogram Distribusi Data')

        st.pyplot(fig)
        st.caption('Line chart diatas menunjukkan distribusi data harga bahan bakar')

# Fungsi untuk halaman 3
def page3():
    st.title("Walmart Sales Analysis and Prediction")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    st.header('Relationship Analysis')

    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    # Membuat heatmap dengan Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='RdBu', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.write('Heatmap di atas menunjukkan korelasi antara penjualan mingguan, harga bahan bakar, suhu, pengangguran, dan tahun dari Walmart. Korelasi diukur pada skala dari -1 hingga 1, dengan -1 menunjukkan korelasi negatif yang sempurna, 0 menunjukkan tidak ada korelasi, dan 1 menunjukkan korelasi positif yang sempurna.')
    st.write('Heatmap menunjukkan bahwa ada beberapa hubungan antara variabel yang dianalisis. Penjualan mingguan berkorelasi negatif dengan harga bahan bakar dan pengangguran, tetapi berkorelasi positif dengan suhu dan tahun. Harga bahan bakar berkorelasi positif dengan suhu, pengangguran, dan tahun. Suhu berkorelasi positif dengan pengangguran dan tahun. Pengangguran berkorelasi negatif dengan tahun.')

# Fungsi untuk halaman 4
def page4():
    st.title("Walmart Sales Analysis and Prediction")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    st.header('Competition Analysis')

    df_yearly = df[df['Year'].isin([2010, 2011, 2012])]

    # Buat histogram untuk setiap tahun
    fig, ax = plt.subplots()
    sns.histplot(df_yearly, x='Sales_Category', hue='Year', multiple='stack', ax=ax, bins=30)
    ax.set_xlabel('Total Sales')
    ax.set_ylabel('Frequency')
    ax.set_title('Sales Distribution by Year')
    st.pyplot(fig)

    st.write('Diagram di atas menunjukkan data penjualan berdasarkan category (Low, Medium, dan High) dari tahun ke tahun. Dapat dilihat bahwa kebanyakan store walmart memiliki pendapatan redah dari tahun ke tahun, dan paling sedikit ada pada kategoru medium')

# Fungsi untuk halaman 5
def page5():
    st.title("Walmart Sales Analysis and Prediction")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    st.header('Comparison Analysis')

    st.subheader('Unemployment rate per store')
    fig = sns.relplot(x='Store', y='Unemployment', data=df, kind='line', height=8, aspect=2)
    plt.xlabel('Store')
    plt.ylabel('Unemployment')
    plt.title('Tingkat Pengangguran Pada Walmart')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.write('Dapat dilihat bahwa kebanyakan store walmart memiliki pendapatan redah dari tahun ke tahun, dan paling sedikit ada pada kategori medium')

    st.subheader('CPI VS Weekly Sales')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='CPI', hue='Sales_Category', multiple='stack', ax=ax, bins=30)
    ax.set_title('Histogram Weekly Sales Berdasarkan CPI')
    ax.set_xlabel('CPI')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write('Bisa dilihat pada diagram diatas bahwa semakin rendah CPI maka kan semakin rendah juga tingkat penjualannya. Dapat dilihat juga apabila CPI tinggi maka penjualan mingguannya juga akan tinggi')

# Fungsi untuk halaman 6
def page6():
    st.title("Walmart Sales Analysis and Prediction")
    URL = 'Data-Before-Mapping.csv'
    df = pd.read_csv(URL)

    st.title('Sales Prediction')
    year = st.selectbox('Select Year', sorted(df['Year'].unique()), key='year_selectbox')
    unemployment = st.slider('Unemployment Rate', float(df['Unemployment'].min()), float(df['Unemployment'].max()), key='unemployment_slider')
    store = st.selectbox('Select Store', sorted(df['Store'].unique()), key='store_selectbox')

    # Prepare features for prediction
    X = df[['Year', 'Unemployment', 'Store']]
    y = df['Weekly_Sales']

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict using user input
    prediction = model.predict([[year, unemployment, store]])

    st.header('Sales Prediction Result')
    st.subheader(f"The predicted weekly sales for the selected year, unemployment rate, and store is:")
    st.title(f"${prediction[0]:,.2f}")

    if st.button('Predict'):
        st.balloons()  # Adding balloons animation when the button is clicked

# Fungsi utama untuk menavigasi antar halaman
def main():
    pages = {
        "Dashboard": page1,
        "Data Distribution": page2,
        "Relationship Analysis": page3,
        "Competition Analysis": page4,
        "Comparison Analysis": page5,
        "Data Prediction": page6
    }

    st.sidebar.title('Walmart Sales')
    selection = st.sidebar.radio("Select Page", list(pages.keys()))

    # Panggil fungsi halaman yang dipilih
    pages[selection]()

if __name__ == "__main__":
    main()
