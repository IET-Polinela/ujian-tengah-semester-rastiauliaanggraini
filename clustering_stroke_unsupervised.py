
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print("Dataset Loaded")
print(df.head())
print("\nInfo Dataset:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Imputasi missing values pada kolom bmi dengan rata-rata
mean_bmi = df['bmi'].mean()
df['bmi'].fillna(mean_bmi, inplace=True)
print("\nMissing Values Setelah Imputasi:")
print(df.isnull().sum())

# Drop 'id' column
df = df.drop(columns=['id'])

# Encode categorical variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = label_encoder.fit_transform(df[col])

# Scale the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Elbow Method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(df_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Metode Elbow")
plt.xlabel("Jumlah Klaster")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_method.png")  # Save elbow plot
plt.show()

# Clustering dengan KMeans (k=3 berdasarkan elbow)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = kmeans.fit_predict(df_scaled)
df['Cluster'] = labels

# Evaluasi dengan silhouette score
score = silhouette_score(df_scaled, labels)
print("\nSilhouette Score:", round(score, 3))

# Visualisasi PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}", alpha=0.6, s=60, edgecolor='k')
plt.title("Visualisasi PCA dari Klaster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.savefig("pca_clusters.png")  # Save PCA scatter plot
plt.show()

# Tampilkan rata-rata fitur tiap klaster
cluster_means = df.groupby('Cluster').mean(numeric_only=True)
print("\nRata-rata Fitur per Klaster:")
print(cluster_means)
