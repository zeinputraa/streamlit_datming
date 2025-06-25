import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Ekspor Tanaman Obat & Rempah",
    page_icon="🌿",
    layout="wide"
)

# Judul aplikasi
st.title("🌿 Aplikasi Prediksi Ekspor Tanaman Obat, Aromatik, dan Rempah-Rempah Indonesia")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", 
                           ["📊 Dataset & Analisis", "🔮 Prediksi Ekspor", "📈 Visualisasi Trend", "📤 Upload Data"])

# Sidebar untuk upload data
st.sidebar.markdown("---")
st.sidebar.subheader("📤 Upload Data CSV")
use_custom_data = st.sidebar.checkbox("Gunakan data CSV sendiri")
uploaded_file = None

if use_custom_data:
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV", 
        type=['csv'],
        help="Upload file CSV dengan kolom: Tahun, Negara_Tujuan, Produk, Volume_Ton, Harga_USD_per_kg, Nilai_Ekspor_USD, GDP_Growth_Tujuan, Kurs_IDR_USD"
    )

# Fungsi untuk membuat dataset simulasi
@st.cache_data
def create_dataset():
    """Membuat dataset simulasi ekspor tanaman obat dan rempah berdasarkan pola data BPS"""
    np.random.seed(42)
    
    # Negara tujuan utama ekspor
    countries = ['India', 'China', 'Vietnam', 'USA', 'Netherlands', 'Germany', 
                'Singapore', 'Japan', 'Malaysia', 'Thailand', 'South Korea', 'UAE']
    
    # Jenis produk
    products = ['Jahe', 'Kunyit', 'Kencur', 'Lada', 'Pala', 'Cengkeh', 
               'Kayu Manis', 'Kapulaga', 'Jintan', 'Ketumbar']
    
    # Generate data untuk tahun 2012-2023
    data = []
    for year in range(2012, 2024):
        for country in countries:
            for product in products:
                # Simulasi volume ekspor (ton) dengan trend naik
                base_volume = np.random.normal(150, 50)
                trend_factor = (year - 2012) * 0.1  # Trend naik 10% per tahun
                volume = max(10, base_volume * (1 + trend_factor) + np.random.normal(0, 20))
                
                # Simulasi harga per kg (USD) dengan fluktuasi
                base_price = np.random.uniform(2, 15)
                price_variation = np.random.normal(0, 0.5)
                price = max(1, base_price + price_variation)
                
                # Nilai ekspor (USD)
                value = volume * 1000 * price  # volume dalam ton ke kg
                
                # Faktor tambahan
                gdp_growth = np.random.uniform(3, 7)  # Pertumbuhan GDP negara tujuan
                exchange_rate = np.random.uniform(14000, 16000)  # Kurs IDR/USD
                
                data.append({
                    'Tahun': year,
                    'Negara_Tujuan': country,
                    'Produk': product,
                    'Volume_Ton': round(volume, 2),
                    'Harga_USD_per_kg': round(price, 2),
                    'Nilai_Ekspor_USD': round(value, 2),
                    'GDP_Growth_Tujuan': round(gdp_growth, 2),
                    'Kurs_IDR_USD': round(exchange_rate, 2)
                })
    
    return pd.DataFrame(data)

# Fungsi untuk load data
@st.cache_data
def load_data(uploaded_file=None):
    """Load data dari CSV yang diupload atau gunakan dataset default"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validasi kolom yang diperlukan
            required_columns = ['Tahun', 'Negara_Tujuan', 'Produk', 'Volume_Ton', 
                              'Harga_USD_per_kg', 'Nilai_Ekspor_USD', 'GDP_Growth_Tujuan', 'Kurs_IDR_USD']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_columns)}")
                st.info("📋 Kolom yang diperlukan: " + ", ".join(required_columns))
                return None, True  # Return None dan flag error
            
            # Validasi tipe data
            numeric_columns = ['Tahun', 'Volume_Ton', 'Harga_USD_per_kg', 'Nilai_Ekspor_USD', 'GDP_Growth_Tujuan', 'Kurs_IDR_USD']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for missing values after conversion
            if df[numeric_columns].isnull().any().any():
                st.warning("⚠️ Beberapa data numerik tidak valid, akan diisi dengan median")
                for col in numeric_columns:
                    df[col].fillna(df[col].median(), inplace=True)
            
            st.success(f"✅ Data berhasil dimuat! Total: {len(df)} baris")
            return df, False
            
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            return None, True
    else:
        return create_dataset(), False

# Load dataset
df, data_error = load_data(uploaded_file)

# Cek apakah data berhasil dimuat
if df is None or data_error:
    st.error("⚠️ Data tidak dapat dimuat. Menggunakan dataset default.")
    df = create_dataset()
    data_error = False

# Info sumber data
if uploaded_file is not None and not data_error:
    st.info(f"📂 Menggunakan data dari: **{uploaded_file.name}**")
else:
    st.info("📊 Menggunakan **dataset simulasi** berdasarkan pola ekspor BPS Indonesia")

# Halaman Dataset & Analisis
if page == "📊 Dataset & Analisis":
    st.header("📊 Dataset Ekspor Tanaman Obat & Rempah-Rempah")
    
    # Tampilkan informasi dataset
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Periode", "2012-2023")
    with col3:
        st.metric("Negara Tujuan", df['Negara_Tujuan'].nunique())
    with col4:
        st.metric("Jenis Produk", df['Produk'].nunique())
    
    # Tampilkan sample data
    st.subheader("📋 Sample Dataset")
    st.dataframe(df.head(120), use_container_width=True)
    
    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Analisis missing values
    st.subheader("🔍 Analisis Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Values:**")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            st.success("✅ Tidak ada missing values dalam dataset")
        else:
            st.dataframe(missing_values[missing_values > 0])
    
    with col2:
        st.write("**Distribusi Negara Tujuan:**")
        country_dist = df['Negara_Tujuan'].value_counts().head(5)
        st.bar_chart(country_dist)

# Halaman Prediksi Ekspor
elif page == "🔮 Prediksi Ekspor":
    st.header("🔮 Model Prediksi Nilai Ekspor")
    
    # Preprocessing data
    @st.cache_data
    def preprocess_data(df):
        # Encode categorical variables
        le_country = LabelEncoder()
        le_product = LabelEncoder()
        
        df_processed = df.copy()
        df_processed['Negara_Encoded'] = le_country.fit_transform(df['Negara_Tujuan'])
        df_processed['Produk_Encoded'] = le_product.fit_transform(df['Produk'])
        
        return df_processed, le_country, le_product
    
    df_processed, le_country, le_product = preprocess_data(df)
    
    # Pilih fitur untuk model
    features = ['Tahun', 'Volume_Ton', 'Harga_USD_per_kg', 'GDP_Growth_Tujuan', 
               'Kurs_IDR_USD', 'Negara_Encoded', 'Produk_Encoded']
    target = 'Nilai_Ekspor_USD'
    
    X = df_processed[features]
    y = df_processed[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    @st.cache_resource
    def train_model(X_train_scaled, y_train):
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        return model
    
    model = train_model(X_train_scaled, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Tampilkan hasil evaluasi
    st.subheader("📊 Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:,.2f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Nilai Aktual (USD)')
    ax.set_ylabel('Nilai Prediksi (USD)')
    ax.set_title('Actual vs Predicted Values')
    st.pyplot(fig)
    
    # Form prediksi
    st.subheader("🎯 Prediksi Nilai Ekspor Baru")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tahun_pred = st.number_input("Tahun", min_value=2024, max_value=2030, value=2024)
        volume_pred = st.number_input("Volume (Ton)", min_value=1.0, max_value=1000.0, value=100.0)
        harga_pred = st.number_input("Harga per kg (USD)", min_value=1.0, max_value=50.0, value=5.0)
    
    with col2:
        negara_pred = st.selectbox("Negara Tujuan", df['Negara_Tujuan'].unique())
        produk_pred = st.selectbox("Produk", df['Produk'].unique())
    
    with col3:
        gdp_pred = st.slider("GDP Growth Negara Tujuan (%)", 1.0, 10.0, 5.0)
        kurs_pred = st.number_input("Kurs IDR/USD", min_value=10000.0, max_value=20000.0, value=15000.0)
    
    if st.button("🔮 Prediksi Nilai Ekspor", type="primary"):
        # Encode input
        negara_encoded = le_country.transform([negara_pred])[0]
        produk_encoded = le_product.transform([produk_pred])[0]
        
        # Prepare input
        input_data = np.array([[tahun_pred, volume_pred, harga_pred, gdp_pred, 
                               kurs_pred, negara_encoded, produk_encoded]])
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        
        # Tampilkan hasil
        st.success(f"💰 Prediksi Nilai Ekspor: **${prediction:,.2f} USD**")
        
        # Konversi ke IDR
        nilai_idr = prediction * kurs_pred
        st.info(f"🇮🇩 Setara dengan: **Rp {nilai_idr:,.0f}**")

# Halaman Visualisasi Trend
elif page == "📈 Visualisasi Trend":
    st.header("📈 Visualisasi Trend Ekspor")
    
    # Trend ekspor per tahun
    st.subheader("📊 Trend Total Ekspor per Tahun")
    yearly_export = df.groupby('Tahun')['Nilai_Ekspor_USD'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_export['Tahun'], yearly_export['Nilai_Ekspor_USD'], 
            marker='o', linewidth=2, markersize=8, color='green')
    ax.set_title('Trend Total Nilai Ekspor Tanaman Obat & Rempah (2012-2023)', fontsize=14)
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Nilai Ekspor (USD)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Top 5 negara tujuan
    st.subheader("🌍 Top 5 Negara Tujuan Ekspor")
    top_countries = df.groupby('Negara_Tujuan')['Nilai_Ekspor_USD'].sum().sort_values(ascending=False).head(5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top_countries.index, top_countries.values, color='skyblue')
    ax.set_title('Top 5 Negara Tujuan Ekspor (Total 2012-2023)')
    ax.set_ylabel('Total Nilai Ekspor (USD)')
    plt.xticks(rotation=45)
    
    # Tambahkan nilai di atas bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Heatmap ekspor per produk dan negara
    st.subheader("🔥 Heatmap Ekspor per Produk dan Negara")
    
    # Aggregate data untuk heatmap
    heatmap_data = df.groupby(['Produk', 'Negara_Tujuan'])['Nilai_Ekspor_USD'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Produk', columns='Negara_Tujuan', values='Nilai_Ekspor_USD')
    heatmap_pivot = heatmap_pivot.fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_pivot, annot=False, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Nilai Ekspor (USD)'})
    ax.set_title('Heatmap Nilai Ekspor per Produk dan Negara Tujuan')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Halaman Upload Data
elif page == "📤 Upload Data":
    st.header("📤 Upload Data CSV Anda")
    
    st.markdown("""
    ### 📋 Format Data yang Diperlukan
    
    File CSV Anda harus memiliki kolom-kolom berikut:
    
    | Kolom | Deskripsi | Contoh |
    |-------|-----------|--------|
    | `Tahun` | Tahun ekspor | 2023 |
    | `Negara_Tujuan` | Negara tujuan ekspor | India, China, USA |
    | `Produk` | Jenis produk | Jahe, Lada, Pala |
    | `Volume_Ton` | Volume dalam ton | 150.5 |
    | `Harga_USD_per_kg` | Harga per kg dalam USD | 5.25 |
    | `Nilai_Ekspor_USD` | Total nilai ekspor USD | 791250.0 |
    | `GDP_Growth_Tujuan` | Pertumbuhan GDP negara tujuan (%) | 5.2 |
    | `Kurs_IDR_USD` | Nilai tukar IDR ke USD | 15000.0 |
    """)
    
    # Template download
    st.subheader("📥 Download Template CSV")
    
    # Buat template data
    template_data = {
        'Tahun': [2023, 2023, 2024],
        'Negara_Tujuan': ['India', 'China', 'USA'],
        'Produk': ['Jahe', 'Lada', 'Pala'],
        'Volume_Ton': [100.0, 150.5, 200.0],
        'Harga_USD_per_kg': [5.0, 8.5, 12.0],
        'Nilai_Ekspor_USD': [500000.0, 1279250.0, 2400000.0],
        'GDP_Growth_Tujuan': [6.1, 5.2, 2.8],
        'Kurs_IDR_USD': [15000.0, 15100.0, 15200.0]
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Convert to CSV
    csv_template = template_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_template,
        file_name="template_ekspor_rempah.csv",
        mime="text/csv",
        help="Download template CSV untuk panduan format data"
    )
    
    st.dataframe(template_df, use_container_width=True)
    
    # Upload section
    st.subheader("📤 Upload File CSV")
    
    if uploaded_file is not None:
        if not data_error:
            st.success("✅ File berhasil diupload dan divalidasi!")
            
            # Tampilkan preview data
            st.subheader("👀 Preview Data yang Diupload")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistik singkat
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Baris", len(df))
            with col2:
                st.metric("Periode", f"{df['Tahun'].min()}-{df['Tahun'].max()}")
            with col3:
                st.metric("Negara Tujuan", df['Negara_Tujuan'].nunique())
            
            st.info("✨ Data Anda siap digunakan! Silakan navigasi ke halaman lain untuk analisis.")
            
        else:
            st.error("❌ Terjadi error saat memproses file. Silakan periksa format data.")
    
    else:
        st.info("📁 Silakan upload file CSV Anda menggunakan sidebar di kiri.")
        
        # Tips untuk data yang baik
        st.subheader("💡 Tips untuk Data yang Baik")
        st.markdown("""
        - **Minimal 100 baris** untuk hasil analisis yang optimal
        - **Data numerik** harus dalam format angka, bukan teks
        - **Nama negara dan produk** konsisten (tidak ada typo)
        - **Tidak ada nilai kosong** di kolom penting
        - **Encoding file** sebaiknya UTF-8
        - **Separator** menggunakan koma (,)
        """)
    
    # Contoh data visualization jika ada data
    if uploaded_file is not None and not data_error:
        st.subheader("📊 Quick Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries
            top_countries = df.groupby('Negara_Tujuan')['Nilai_Ekspor_USD'].sum().sort_values(ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            top_countries.plot(kind='bar', ax=ax, color='lightcoral')
            ax.set_title('Top 5 Negara Berdasarkan Nilai Ekspor')
            ax.set_ylabel('Nilai Ekspor (USD)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Top products
            top_products = df.groupby('Produk')['Nilai_Ekspor_USD'].sum().sort_values(ascending=False).head(5)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            top_products.plot(kind='bar', ax=ax, color='lightgreen')
            ax.set_title('Top 5 Produk Berdasarkan Nilai Ekspor')
            ax.set_ylabel('Nilai Ekspor (USD)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Distribusi harga per produk
    st.subheader("💰 Distribusi Harga per Produk")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    df.boxplot(column='Harga_USD_per_kg', by='Produk', ax=ax, rot=45)
    ax.set_title('Distribusi Harga per Produk')
    ax.set_xlabel('Produk')
    ax.set_ylabel('Harga per kg (USD)')
    plt.suptitle('')  # Remove automatic title
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**📝 Catatan:**
- Dataset default adalah simulasi berdasarkan pola data ekspor BPS Indonesia
- Anda dapat menggunakan data CSV sendiri melalui fitur upload
- Model menggunakan Linear Regression untuk prediksi nilai ekspor
- Aplikasi ini dibuat untuk keperluan pembelajaran dan demo

**📋 Format CSV yang Diperlukan:**
`Tahun, Negara_Tujuan, Produk, Volume_Ton, Harga_USD_per_kg, Nilai_Ekspor_USD, GDP_Growth_Tujuan, Kurs_IDR_USD`
""")

st.markdown("*Dikembangkan dengan ❤️ menggunakan Streamlit*")