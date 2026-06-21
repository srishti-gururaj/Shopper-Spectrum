# Shopper-Spectrum

​# 🛒 Shopper Spectrum: E-Commerce Segmentation & Recommendation

## 📖 Project Overview
This project is a complete, end-to-end Machine Learning pipeline designed to analyze e-commerce transaction data, segment customers based on purchasing behavior, and recommend products using collaborative filtering. 

## ✨ Key Features & Deliverables
* **Data Preprocessing & EDA:** Cleaned raw transaction data and generated 15+ comprehensive visualizations (Lollipop charts, Hexbins, distributions) to uncover business insights.
* **RFM Analysis:** Engineered Recency, Frequency, and Monetary metrics to accurately evaluate customer value.
* **Customer Segmentation (Machine Learning):** Trained and compared multiple clustering models including **K-Means, Agglomerative Hierarchical, and DBSCAN**. K-Means was selected as the optimal model using the Elbow Method and Silhouette Scores.
* **Statistical Validation:** Validated the statistical significance of the customer segments mathematically using **ANOVA Hypothesis Testing**, calculating exact P-values to prove the models were not based on random chance.
* **Recommendation Engine:** Built an item-based collaborative filtering system using **Cosine Similarity** to suggest complementary products based on user selection.
* **Interactive Web Dashboard:** Deployed a fully functional **Streamlit** web application allowing users to input live data for real-time segmentation predictions and product recommendations.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation & Math:** Pandas, NumPy, SciPy (Hypothesis Testing)
* **Machine Learning:** Scikit-Learn (K-Means, Hierarchical, DBSCAN, Cosine Similarity)
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit, LocalTunnel

## 📂 Repository Structure
* `shopper_spectrum_analysis.ipynb`: The main Jupyter Notebook containing all data cleaning, EDA, model training, and mathematical evaluation.
* `app.py`: The Streamlit web application executable script.
* `kmeans_model.pkl`: The exported, trained K-Means clustering model.
* `scaler.pkl`: The data scaler used to standardize user inputs for the app.
* `item_similarity.csv`: The Cosine Similarity matrix powering the recommendation engine.
* `online_retail.zip`: The raw dataset used to train the models.

## 🚀 How to Run the Web App Locally
1. Download all repository files into a single folder.
2. Open your terminal or command prompt in that folder.
3. Install the required dependencies by running: `pip install streamlit pandas scikit-learn numpy`
4. Launch the application by running: `streamlit run app.py`
