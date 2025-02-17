import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np


st.image("image.png", width=300)
st.markdown("""
    <style>
    .centered-title {
        font-size: 48px;
        color: #4b72fa;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="centered-title">✨ Bienvenido al Recomendador de Productos Skincare ✨</h1>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    reviews = pd.read_csv('reviews_final.csv', dtype={'author_id': str})
    products = pd.read_csv('products_final.csv')
    products_from_clusters = pd.read_csv('products_recommended.csv')
    return reviews, products, products_from_clusters


@st.cache_data
def process_reviews_data(reviews):
    reviews_dummies = pd.get_dummies(
        reviews,
        columns = ['skin_tone', 'skin_type'],
        drop_first = True
    )

    reviews_customers = reviews_dummies.groupby('author_id').agg({
        'rating': 'mean',  
        'price_usd': 'mean',
        'skin_tone_light_to_medium': 'mean',
        'skin_tone_medium_to_tan': 'mean',
        'skin_tone_very_light': 'mean',
        'skin_type_dry': 'mean',
        'skin_type_normal': 'mean',
        'skin_type_oily': 'mean'
    
    }).reset_index()
        
    reviews_cluster_proportions = reviews_dummies.groupby('author_id')['cluster_product'].value_counts(normalize=True).unstack(fill_value=0)
    
    reviews_cluster_proportions['most_frequent_cluster'] = reviews_cluster_proportions.idxmax(axis=1)
    reviews_customers = reviews_customers.merge(reviews_cluster_proportions, on='author_id', how='left')
    return reviews_customers

@st.cache_data
def process_clustering(reviews_customers):
    variables_cluster = [
        'rating',
        'price_usd',
        'skin_tone_light_to_medium',
        'skin_tone_medium_to_tan',
        'skin_tone_very_light',
        'skin_type_dry',
        'skin_type_normal',
        'skin_type_oily',
        'most_frequent_cluster'
    ] 

    reviews_cluster = reviews_customers[variables_cluster]
    reviews_cluster.columns = reviews_cluster.columns.astype(str)
    scaler = StandardScaler()
    reviews_scaled = scaler.fit_transform(reviews_cluster)
    
    pca = PCA(n_components=2) 
    reviews_pca = pca.fit_transform(reviews_scaled)
    kmeans_cluster_user = KMeans(n_clusters=8, random_state=42)
    kmeans_cluster_user.fit(reviews_pca)
    
    reviews_customers['cluster_user'] = kmeans_cluster_user.labels_
    return reviews_customers, scaler, kmeans_cluster_user, variables_cluster

@st.cache_data
def train_neighbor_model(pivot_table):
    model_neighbor = NearestNeighbors(
        n_neighbors= 5,
        metric = "cosine",
    )
    model_neighbor.fit(pivot_table)
    return model_neighbor

reviews, products, products_from_clusters = load_data()

reviews_customers = process_reviews_data(reviews)
reviews_customers, scaler, kmeans_cluster_user, variables_cluster = process_clustering(reviews_customers)

pivot_table = reviews.pivot_table(
    values="rating",
    index="author_id",
    columns="product_id",
    fill_value=0
)

model_neighbor = train_neighbor_model(pivot_table)

customer_features = reviews_customers[variables_cluster]
customer_features.columns = customer_features.columns.astype(str)
target = reviews_customers['cluster_user']

knn_classifier_user = KNeighborsClassifier(n_neighbors=5)
knn_classifier_user.fit(customer_features, target)

def get_cluster_newuser(new_user_data, knn_classifier_user, scaler, features_columns):
    new_user_dummies = pd.get_dummies(new_user_data)
    new_user_dummies = new_user_dummies.reindex(columns=features_columns, fill_value=0)
    new_user_scaled = scaler.transform(new_user_dummies)
    predicted_cluster = knn_classifier_user.predict(new_user_scaled)[0]
    return predicted_cluster

def get_clusterproducts_users (nb_user, new_user_data=None):
    if nb_user in reviews['author_id'].values:
        clusters = reviews.loc[reviews['author_id'] == nb_user, 'cluster_product'].values
        if len(clusters) > 0:
            return list(clusters)
    elif nb_user == '0' and new_user_data is not None:
        predicted_cluster = get_cluster_newuser(new_user_data, knn_classifier, scaler, X.columns)
        return [predicted_cluster]
        return list(clusters)
    else:
        print('User not found')
        
def get_neighbors_newuser(user_id,pivot_table, model_neighbor, n_neighbors=5):
    pivot_table.columns = pivot_table.columns.astype(str) 
    if user_id not in pivot_table.index:
        return []
    pivot_table.columns = pivot_table.columns.astype(str)
    new_user_ratings = pd.DataFrame([0] * pivot_table.shape[1]).T
    new_user_ratings.columns = pivot_table.columns
    pivot_with_new_user = pd.concat([pivot_table, new_user_ratings], ignore_index=True)
    pivot_with_new_user_np = pivot_with_new_user.to_numpy()
    _, vecinos = model_neighbor.kneighbors([pivot_with_new_user_np[-1]])  # Última fila = nuevo usuario
    neighbor_ids = vecinos[0, 1:]
    return neighbor_ids.tolist()

def get_neighbors_existing_user(user_id, pivot_table, n_neighbors=5):
    pivot_table.columns = pivot_table.columns.astype(str)
    if user_id not in pivot_table.index:
        st.error(f"El usuario {user_id} no existe en la base de datos.")
        return []
    user_row = pivot_table.loc[user_id].values.reshape(1, -1)  
    modelo = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    modelo.fit(pivot_table)
    _, vecinos = modelo.kneighbors(user_row)   
    neighbor_ids = pivot_table.index[vecinos[0, 1:]] 
    return neighbor_ids.tolist()

def get_top_products_by_cluster(clusters, products_for_cluster, top_n=3):
    clusters = [int(cluster) for cluster in clusters]
    top_products = {}
    for cluster in clusters:
        cluster_products = products_for_cluster[products_for_cluster['cluster_product'] == cluster].sort_values(by= 'rating', ascending = False)
        top_cluster_products = cluster_products.head(top_n)
        top_products[cluster] = top_cluster_products
    return top_products


def recommend_from_neighbors(user_id, pivot_table, n_neighbors=5):
    modelo = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    modelo.fit(pivot_table.values)
    user_row = pivot_table.loc[user_id].values.reshape(1, -1) 
    _, vecinos = modelo.kneighbors(user_row)
    author_ids_vecinos = pivot_table.index[vecinos[0]]
    only_rating = pivot_table.iloc[vecinos[0], :].replace(0, np.nan).dropna(how='all', axis=1).dropna(how='all', axis=0)
    media_productos = only_rating.mean(axis=0)
    tabla_pedro = pivot_table.loc[user_id].to_frame(name='rating_real')
    tabla_pedro['recomendacion'] = media_productos
    recommendation_users = tabla_pedro[tabla_pedro['rating_real'] == 0].sort_values("recomendacion", ascending=False).head(10)
    
    return recommendation_users.index.tolist()


def recommend_products_for_user(user_id, pivot_table, new_user_data, n_neighbors, model_neighbor, knn_classifier_user, scaler):
    w = []
    user_id = str(user_id) 
    pivot_table.index = pivot_table.index.astype(str)
    if user_id == '0':
        if new_user_data is None:
            raise ValueError("Se requiere 'new_user_data' para un nuevo usuario.")
        predicted_cluster = get_cluster_newuser(new_user_data, knn_classifier_user, scaler, customer_features.columns)
        neighbor_ids = get_neighbors_newuser(user_id,pivot_table, model_neighbor, n_neighbors)
        new_user_ratings = pd.Series([0] * pivot_table.shape[1], index=pivot_table.columns)
        pivot_table.loc['new_user'] = new_user_ratings
        from_neighbors = recommend_from_neighbors('new_user', pivot_table, n_neighbors=n_neighbors)
        w.extend(from_neighbors)
        pivot_table = pivot_table.drop(index='new_user') 
    else:
        new_user_data = None
        neighbor_ids = get_neighbors_existing_user(user_id, pivot_table, n_neighbors=n_neighbors)
        from_neighbors = recommend_from_neighbors(user_id, pivot_table, n_neighbors=n_neighbors)
        w.extend(from_neighbors)
        for i in neighbor_ids:
            cluster_i = get_clusterproducts_users(i, new_user_data)
            if cluster_i:
                for j in cluster_i:
                    recommend_from_cluster = get_top_products_by_cluster([j], products_from_clusters, top_n=3)
                    for cluster, productss in recommend_from_cluster.items():
                        w.extend(productss['product_id'].tolist())  
    unique_w = list(set(w))
    if user_id != '0':
        productos_comprados = pivot_table.loc[user_id]
        productos_comprados = productos_comprados[productos_comprados > 0].index.tolist() 
        productos_no_comprados = [product for product in unique_w if product not in productos_comprados] 
    else:   
        productos_no_comprados = unique_w
    best = products[products['product_id'].isin(productos_no_comprados)][['product_id','product_name', 'rating','cluster_product','secondary_category']].sort_values(by='rating', ascending=False)
    best = best[best['rating'] >= 4]
    return best



# Interfaz con Streamlit

st.title("Recomendador de Productos Skincare")

if "best_products" not in st.session_state:
    st.session_state["best_products"] = None

st.markdown("""
<div style="padding: 10px; border-radius: 10px;">
    <h4 style="color: #28a745;">¿Eres cliente? 👤</h4>
</div>
""", unsafe_allow_html=True)

es_cliente = st.radio("", ("Sí", "No"))
if es_cliente == "Sí":
    st.markdown("**¡Déjanos conocerte mejor! 📲✨ Introduciendo tu número de cliente, recibirás recomendaciones personalizadas. ¡Descubre productos increíbles que parecen hechos justo para ti!**")
    user_id = st.text_input("Introduce tu número de cliente:")
    if st.button('Obtener recomendaciones'):
        best_products = recommend_products_for_user(user_id, pivot_table, None, 5, model_neighbor, knn_classifier_user, scaler)
        st.session_state["best_products"] = best_products
else:      
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px;">
        <h4 style="color: #4b72fa;">¿No sabes cuál es tu tono de piel? 🌞</h4>
        <ul>
            <li><strong>✨ Very Light:</strong> Si tu piel es extremadamente clara, te quemas al sol con facilidad y casi nunca te bronceas, tu tono es very light.</li>
            <li><strong>🌸 Light to Medium:</strong> Si tu piel se quema con el sol, pero puedes lograr un bronceado suave, tu tono de piel es light to medium.</li>
            <li><strong>🌤️ Medium to Tan:</strong> Si te bronceas fácilmente y te quemas solo después de una exposición prolongada al sol, tu tono es medium to tan.</li>
            <li><strong>🌑 Dark:</strong> Si tu piel se broncea con facilidad y rara vez te quemas, entonces tienes un tono de piel oscuro.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)   
    skin_tone = st.radio("Selecciona tu tono de piel", ['very_light', 'medium_to_tan', 'light_to_medium', 'dark'])
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px;">
        <h4 style="color: #2b8a8a;">¿No sabes cuál es tu tipo de piel? 💧</h4>
        <ul>
            <li><strong>💦 Oily:</strong> Tu piel tiende a brillar, especialmente en la zona T (frente, nariz, mentón), y puedes sentir que produce mucho sebo.</li>
            <li><strong>🌵 Dry:</strong> Si sientes tu piel tirante, áspera o incluso escamosa, y necesita hidratación constante, tu piel es seca.</li>
            <li><strong>🌿 Normal:</strong> Si tu piel no es ni muy grasa ni muy seca, no se irrita fácilmente, y se siente equilibrada, entonces tienes un tipo de piel normal.</li>
            <li><strong>🌗 Combination:</strong> Tu piel es mixta, suele tener zonas grasas (generalmente la zona T) y otras zonas secas o normales.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True) 
    skin_type = st.radio("Selecciona tu tipo de piel", ['oily', 'dry', 'normal', 'combination']) 
    if st.button('Obtener recomendaciones'):
        new_user_data = {'skin_tone': skin_tone, 'skin_type': skin_type}
        best_products = recommend_products_for_user('0', pivot_table, new_user_data, 5, model_neighbor, knn_classifier_user, scaler)
        st.session_state["best_products"] = best_products

if st.session_state["best_products"] is not None:
    best_products = st.session_state["best_products"]

    if isinstance(best_products, pd.DataFrame) and 'secondary_category' in best_products.columns:
        st.write("Filtros para refinar los productos recomendados:")
        categorias = st.multiselect("Tipo de Tratamiento", 
                                    options=best_products['secondary_category'].unique(),
                                    default=best_products['secondary_category'].unique())
     
        productos_filtrados = best_products[(best_products['secondary_category'].isin(categorias))]
    productos_filtrados = productos_filtrados.rename(columns={
    'product_name': 'Nombre de Producto', 
    'rating': 'Rating', 
    'secondary_category': 'Tipo de Tratamiento'})
    st.write("Productos recomendados:")
    st.dataframe(productos_filtrados[['Nombre de Producto', 'Rating', 'Tipo de Tratamiento']])






