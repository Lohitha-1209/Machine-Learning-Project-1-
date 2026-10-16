
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

st.set_page_config(
    page_title="ML Capstone Explorer",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

@st.cache_data
def load_data(filename):
    return pd.read_csv(DATA_DIR / filename)

@st.cache_data
def prepare_amazon():
    df = load_data("amazon.csv").copy()

    def to_number(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

    for col in [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]:
        df[col] = to_number(df[col])

    return df

@st.cache_resource
def train_regression():
    df = load_data("Crop Yeild Data.csv")
    target = "Yield"

    X = df.drop(columns=[target, "Production"], errors="ignore")
    y = pd.to_numeric(df[target], errors="coerce")

    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return model, X, y, {
        "R²": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred)
    }

@st.cache_resource
def train_classification():
    df = load_data("cars.csv")
    target = "Vehicle Size"

    df = df.dropna(subset=[target]).copy()
    X = df.drop(columns=[target])
    y = df[target]

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Weighted F1": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        )["weighted avg"]["f1-score"]
    }

    return model, X, y, metrics, y_test, pred

@st.cache_resource
def train_clustering():
    df = prepare_amazon()

    features = [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]

    numeric = df[features].copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    sample_size = min(1500, len(scaled))
    sample_idx = np.random.RandomState(42).choice(
        len(scaled), size=sample_size, replace=False
    )
    sample = scaled[sample_idx]

    hierarchical = AgglomerativeClustering(n_clusters=4)
    hierarchical_labels = hierarchical.fit_predict(sample)

    return df, features, scaler, scaled, kmeans, labels, sample_idx, hierarchical_labels

st.title("📊 Machine Learning Capstone Explorer")
st.caption("Interactive Regression · Classification · Clustering")

page = st.sidebar.radio(
    "Choose a track",
    ["Home", "Regression", "Classification", "Amazon Clustering"]
)

if page == "Home":
    st.subheader("Explore all three Machine Learning tracks")

    c1, c2, c3 = st.columns(3)
    c1.metric("ML tracks", "3")
    c2.metric("Datasets", "3")
    c3.metric("Interface", "Interactive")

    st.markdown("""
    **Regression:** predict crop yield from agricultural conditions.

    **Classification:** predict vehicle size from car specifications.

    **Clustering:** group Amazon products using price, discount, rating
    and review-count features.
    """)

    st.info("Select a track from the sidebar to begin.")

elif page == "Regression":
    st.header("🌾 Crop Yield Prediction")

    with st.spinner("Preparing the regression model..."):
        model, X, y, metrics = train_regression()

    st.write("Enter the agricultural conditions below.")
    input_row = {}

    c1, c2 = st.columns(2)
    for i, col in enumerate(X.columns):
        with (c1 if i % 2 == 0 else c2):
            if pd.api.types.is_numeric_dtype(X[col]):
                median = float(X[col].median())
                minimum = float(X[col].min())
                maximum = float(X[col].max())
                if not np.isfinite(median):
                    median = 0.0
                if not np.isfinite(minimum) or not np.isfinite(maximum):
                    minimum, maximum = 0.0, 100.0
                if minimum == maximum:
                    input_row[col] = st.number_input(
                        col, value=median
                    )
                else:
                    input_row[col] = st.number_input(
                        col, min_value=minimum, max_value=maximum,
                        value=min(max(median, minimum), maximum)
                    )
            else:
                choices = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )
                input_row[col] = st.selectbox(col, choices)

    if st.button("Predict Crop Yield", type="primary"):
        sample = pd.DataFrame([input_row], columns=X.columns)
        prediction = model.predict(sample)[0]
        st.success(f"Predicted Yield: {prediction:.4f}")

        a, b, c = st.columns(3)
        a.metric("Test R²", f"{metrics['R²']:.3f}")
        b.metric("Test RMSE", f"{metrics['RMSE']:.3f}")
        c.metric("Test MAE", f"{metrics['MAE']:.3f}")

        st.caption("Metrics are calculated on a held-out test set.")

    st.subheader("Model evaluation")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

elif page == "Classification":
    st.header("🚗 Vehicle Size Classification")

    with st.spinner("Preparing the classification model..."):
        model, X, y, metrics, y_test, pred = train_classification()

    st.write("Enter the vehicle specifications.")
    input_row = {}
    c1, c2 = st.columns(2)

    for i, col in enumerate(X.columns):
        with (c1 if i % 2 == 0 else c2):
            if pd.api.types.is_numeric_dtype(X[col]):
                median = float(X[col].median())
                minimum, maximum = float(X[col].min()), float(X[col].max())
                if not np.isfinite(median):
                    median = 0.0
                if not np.isfinite(minimum) or not np.isfinite(maximum):
                    minimum, maximum = 0.0, 100.0
                if minimum == maximum:
                    input_row[col] = st.number_input(
                        col, value=median, key=f"class_{col}"
                    )
                else:
                    input_row[col] = st.number_input(
                        col, min_value=minimum, max_value=maximum,
                        value=min(max(median, minimum), maximum),
                        key=f"class_{col}"
                    )
            else:
                choices = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )
                input_row[col] = st.selectbox(
                    col, choices, key=f"class_{col}"
                )

    if st.button("Predict Vehicle Size", type="primary"):
        sample = pd.DataFrame([input_row], columns=X.columns)
        result = model.predict(sample)[0]
        st.success(f"Predicted Vehicle Size: {result}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            classes = model.named_steps["model"].classes_
            prob_df = pd.DataFrame({
                "Vehicle Size": classes,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Vehicle Size"))

    a, b = st.columns(2)
    a.metric("Test Accuracy", f"{metrics['Accuracy']:.3f}")
    b.metric("Weighted F1", f"{metrics['Weighted F1']:.3f}")

    st.subheader("Confusion Matrix")
    labels = sorted(pd.Series(y_test).unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    st.dataframe(
        pd.DataFrame(cm, index=labels, columns=labels),
        use_container_width=True
    )

elif page == "Amazon Clustering":
    st.header("🛒 Amazon Product Explorer")

    with st.spinner("Preparing product clusters..."):
        df, features, scaler, scaled, kmeans, labels, sample_idx, h_labels = train_clustering()

    df = df.copy()
    df["Cluster"] = labels

    search = st.text_input("Search for a product by name")
    matches = df[
        df["product_name"].astype(str).str.contains(
            search, case=False, na=False
        )
    ] if search.strip() else df

    if matches.empty:
        st.warning("No matching products found. Try another search.")
    else:
        options = matches["product_name"].drop_duplicates().tolist()
        selected = st.selectbox("Select a product", options)
        product = df[df["product_name"] == selected].iloc[0]

        st.subheader("Product details")
        st.write(product["product_name"])

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Discounted price",
            f"₹{product['discounted_price']:,.2f}"
            if pd.notna(product["discounted_price"]) else "N/A"
        )
        c2.metric(
            "Actual price",
            f"₹{product['actual_price']:,.2f}"
            if pd.notna(product["actual_price"]) else "N/A"
        )
        c3.metric(
            "Rating",
            f"{product['rating']:.1f}/5"
            if pd.notna(product["rating"]) else "N/A"
        )

        st.metric("K-Means Cluster", int(product["Cluster"]))

        st.subheader("Similar products in the same cluster")
        similar = df[
            (df["Cluster"] == product["Cluster"]) &
            (df["product_name"] != selected)
        ].head(10)

        st.dataframe(
            similar[[
                "product_name", "category", "discounted_price",
                "actual_price", "discount_percentage", "rating"
            ]],
            use_container_width=True
        )

    st.subheader("Product price distribution")
    price_df = df.dropna(subset=["discounted_price"])
    if not price_df.empty:
        fig, ax = plt.subplots()
        ax.hist(price_df["discounted_price"], bins=30)
        ax.set_title("Distribution of Discounted Prices")
        ax.set_xlabel("Discounted Price (₹)")
        ax.set_ylabel("Number of Products")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Elbow Method")
    inertias = []
    k_values = range(2, 9)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(list(k_values), inertias, marker="o")
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("PCA Cluster Visualization")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap="tab10", s=15, alpha=0.7
    )
    ax.set_title("Amazon Products: K-Means Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Clustering Evaluation")
    sample_labels = labels[sample_idx]
    st.write({
        "Silhouette Score": round(
            silhouette_score(scaled, labels), 4
        ),
        "Davies-Bouldin Index": round(
            davies_bouldin_score(scaled, labels), 4
        ),
        "Calinski-Harabasz Index": round(
            calinski_harabasz_score(scaled, labels), 2
        )
    })

    st.caption(
        "Clustering uses recorded product attributes, not historical "
        "sales data. It identifies product groups, not time-based trends."
    )

