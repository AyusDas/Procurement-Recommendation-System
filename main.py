import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request

# Database Connection (Modify with your credentials)
DB_URI = "postgresql://username:password@localhost:5432/nowpurchase"
engine = create_engine(DB_URI)

app = Flask(__name__)

def load_data():
    query = "SELECT customer_id, material_id, quantity FROM procurement_transactions;"
    df = pd.read_sql(query, engine)
    return df

# Data Preprocessing
def preprocess_data(df):
    pivot_table = df.pivot_table(index='customer_id', columns='material_id', values='quantity', fill_value=0)
    return pivot_table

# Compute Similarity
def compute_similarity(pivot_table):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(pivot_table)
    similarity_matrix = cosine_similarity(normalized_data)
    return pd.DataFrame(similarity_matrix, index=pivot_table.index, columns=pivot_table.index)

# Generate Recommendations
def recommend_materials(customer_id, pivot_table, similarity_df, top_n=3):
    if customer_id not in similarity_df.index:
        return "Customer not found"
    
    similar_customers = similarity_df[customer_id].sort_values(ascending=False)[1:top_n+1]
    recommended_materials = set()
    
    for similar_customer in similar_customers.index:
        customer_materials = pivot_table.loc[similar_customer]
        top_materials = customer_materials[customer_materials > 0].index.tolist()
        recommended_materials.update(top_materials)
    
    return list(recommended_materials)

@app.route('/recommend', methods=['GET'])
def recommend():
    customer_id = request.args.get('customer_id', type=int)
    if customer_id is None:
        return jsonify({"error": "Missing customer_id parameter"}), 400
    
    df = load_data()
    pivot_table = preprocess_data(df)
    similarity_df = compute_similarity(pivot_table)
    recommendations = recommend_materials(customer_id, pivot_table, similarity_df)
    
    return jsonify({"customer_id": customer_id, "recommended_materials": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
