
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
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ML Capstone Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
}
.section-note {
    padding: 12px;
    border-radius: 10px;
    background: var(--secondary-background-color);
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 2. DATA LOADING AND PREPROCESSING
# ==================================================

@st.cache_data
def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        files = (
            [p.name for p in DATA_DIR.rglob("*") if p.is_file()]
            if DATA_DIR.exists() else []
        )
        raise FileNotFoundError(
            f"Missing dataset: {filename}. "
            f"Files found in data folder: {files}"
        )

    return pd.read_csv(path)


def to_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ))

    return ColumnTransformer(transformers)


def get_feature_names(model, X):
    """Get feature names after preprocessing."""
    try:
        return model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return np.array(X.columns)


# ==================================================
# 3. REGRESSION MODELS
# ==================================================

@st.cache_resource
def train_regression():
    df = load_data("Crop Yield Data.csv")

    target = next(
        (c for c in df.columns if c.strip().lower() == "yield"),
        None
    )
    if target is None:
        raise ValueError(
            f"Yield column not found. Columns: {df.columns.tolist()}"
        )

    y = to_numeric(df[target])
    X = df.drop(columns=[target, "Production"], errors="ignore")

    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].copy()

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if len(X) < 10 or X.shape[1] == 0:
        raise ValueError("Insufficient usable data for regression.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    fitted_models = {}

    for name, estimator in [
        ("Random Forest Regressor",
         RandomForestRegressor(
             n_estimators=100, random_state=42, n_jobs=-1
         )),
        ("Gradient Boosting Regressor",
         GradientBoostingRegressor(random_state=42))
    ]:
        model = Pipeline([
            ("preprocess", build_preprocessor(X)),
            ("model", estimator)
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "R²": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))
        }

        fitted_models[name] = model

    return X, y, X_test, y_test, fitted_models, results


# ==================================================
# 4. CLASSIFICATION MODELS
# ==================================================

@st.cache_resource
def train_classification():
    df = load_data("cars.csv")

    target = next(
        (c for c in df.columns
         if c.strip().lower() == "vehicle size"),
        None
    )

    if target is None:
        raise ValueError(
            f"Vehicle Size column not found. Columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[target]).copy()
    X = df.drop(columns=[target])
    y = df[target].astype(str)

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if len(X) < 10 or y.nunique() < 2:
        raise ValueError("Insufficient data for classification.")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    results = {}
    fitted_models = {}
    predictions = {}

    for name, estimator in [
        ("Random Forest Classifier",
         RandomForestClassifier(
             n_estimators=100,
             random_state=42,
             n_jobs=-1,
             class_weight="balanced"
         )),
        ("Gradient Boosting Classifier",
         GradientBoostingClassifier(random_state=42))
    ]:
        model = Pipeline([
            ("preprocess", build_preprocessor(X)),
            ("model", estimator)
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "Recall (weighted)": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "F1 (weighted)": f1_score(
                y_test, pred, average="weighted", zero_division=0
            )
        }

        fitted_models[name] = model
        predictions[name] = pred

    return X, y, X_test, y_test, fitted_models, results, predictions


# ==================================================
# 5. AMAZON DATA AND CLUSTERING
# ==================================================

@st.cache_data
def prepare_amazon():
    df = load_data("amazon.csv").copy()
    df.columns = [c.strip() for c in df.columns]

    lookup = {c.lower(): c for c in df.columns}

    for col in [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]:
        actual_col = lookup.get(col)
        if actual_col:
            df[actual_col] = to_numeric(df[actual_col])

    return df


@st.cache_resource
def train_clustering():
    df = prepare_amazon()
    lookup = {c.strip().lower(): c for c in df.columns}

    required = [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]

    missing = [c for c in required if c not in lookup]
    if missing:
        raise ValueError(
            f"Missing Amazon columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    features = [lookup[c] for c in required]
    numeric = df[features].copy().replace(
        [np.inf, -np.inf], np.nan
    )

    features = [c for c in features if numeric[c].notna().any()]
    numeric = numeric[features]

    if len(df) < 5 or len(features) < 2:
        raise ValueError("Insufficient data for clustering.")

    numeric = numeric.fillna(numeric.median()).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    k = min(4, len(df) - 1)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(scaled)

    # Agglomerative clustering is computed on a reproducible sample
    # to avoid excessive memory usage on large datasets.
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(
        len(df), size=min(len(df), 1500), replace=False
    )
    sample_scaled = scaled[sample_idx]

    agglomerative = AgglomerativeClustering(n_clusters=k)
    agg_labels = agglomerative.fit_predict(sample_scaled)

    return (
        df, features, scaled, kmeans,
        kmeans_labels, sample_idx, agg_labels
    )


# ==================================================
# 6. SIDEBAR AND HOME PAGE
# ==================================================

with st.sidebar:
    st.title("📊 ML Studio")
    st.caption("Review 2 · Machine Learning Project")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Regression", "Classification", "Clustering"]
    )
    st.divider()
    st.caption("Models are trained from the datasets in your repository.")

if page == "Overview":
    st.title("Machine Learning Capstone Studio")
    st.write(
        "An interactive dashboard for model predictions, "
        "algorithm comparison and evaluation."
    )

    a, b, c = st.columns(3)
    a.metric("ML Tasks", "3")
    b.metric("Supervised Learning", "2")
    c.metric("Unsupervised Learning", "1")

    st.divider()
    st.subheader("Explore the project")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🌾 Regression")
        st.write("Crop yield prediction.")
        st.caption("Random Forest · Gradient Boosting")

    with c2:
        st.markdown("### 🚗 Classification")
        st.write("Vehicle size prediction.")
        st.caption("Random Forest · Gradient Boosting")

    with c3:
        st.markdown("### 🛒 Clustering")
        st.write("Amazon product segmentation.")
        st.caption("K-Means · Agglomerative Clustering")

    st.info(
        "Choose a task from the sidebar to view predictions, "
        "evaluation metrics and visualizations."
    )


# ==================================================
# 7. REGRESSION DASHBOARD
# ==================================================

elif page == "Regression":
    st.title("🌾 Crop Yield Regression")
    st.caption("Supervised learning · Continuous target prediction")

    try:
        with st.spinner("Training and evaluating regression models..."):
            X, y, X_test, y_test, models, results = train_regression()

        selected = st.selectbox(
            "Select regression algorithm",
            list(models.keys())
        )

        model = models[selected]
        pred = model.predict(X_test)
        metrics = results[selected]

        st.markdown(
            f'<div class="section-note"><b>Algorithm:</b> {selected}'
            f'<br><b>Evaluation:</b> 80/20 train-test split; '
            f'Random State = 42</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{metrics['R²']:.4f}")
        m2.metric("MAE", f"{metrics['MAE']:.4f}")
        m3.metric("RMSE", f"{metrics['RMSE']:.4f}")

        st.subheader("Compare regression algorithms")
        st.dataframe(
            pd.DataFrame(results).T,
            use_container_width=True
        )

        tab1, tab2, tab3 = st.tabs([
            "Actual vs Predicted",
            "Residual Analysis",
            "Feature Importance"
        ])

        with tab1:
            st.caption(
                f"Algorithm: {selected}. Points closer to the diagonal "
                "represent more accurate predictions."
            )
            fig, ax = plt.subplots()
            ax.scatter(y_test, pred, alpha=0.65)
            low = min(float(y_test.min()), float(pred.min()))
            high = max(float(y_test.max()), float(pred.max()))
            ax.plot([low, high], [low, high], linestyle="--")
            ax.set_xlabel("Actual Yield")
            ax.set_ylabel("Predicted Yield")
            ax.set_title(f"Actual vs Predicted — {selected}")
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            residuals = y_test.to_numpy() - pred
            st.caption(
                f"Algorithm: {selected}. Residual = actual yield "
                "minus predicted yield. Values near zero are desirable."
            )
            fig, ax = plt.subplots()
            ax.scatter(pred, residuals, alpha=0.65)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted Yield")
            ax.set_ylabel("Residual")
            ax.set_title(f"Residual Plot — {selected}")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.caption(
                f"Algorithm: {selected}. Feature importance indicates "
                "which processed input features contribute most to "
                "this fitted tree-based model."
            )
            estimator = model.named_steps["model"]
            names = get_feature_names(model, X)

            importance = pd.DataFrame({
                "Feature": names,
                "Importance": estimator.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            importance = importance.sort_values("Importance")
            fig, ax = plt.subplots()
            ax.barh(importance["Feature"], importance["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {selected}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Predict crop yield")
        st.caption(
            "Enter feature values. Defaults use typical values "
            "from the available dataset."
        )

        inputs = {}
        left, right = st.columns(2)

        for i, col in enumerate(X.columns):
            with (left if i % 2 == 0 else right):
                if pd.api.types.is_numeric_dtype(X[col]):
                    vals = pd.to_numeric(X[col], errors="coerce").dropna()
                    default = float(vals.median()) if not vals.empty else 0.0
                    inputs[col] = st.number_input(
                        col, value=default, key=f"reg_{col}"
                    )
                else:
                    choices = sorted(X[col].dropna().astype(str).unique())
                    inputs[col] = (
                        st.selectbox(col, choices, key=f"reg_{col}")
                        if choices else ""
                    )

        if st.button("Predict Yield", type="primary"):
            sample = pd.DataFrame([inputs], columns=X.columns)
            result = model.predict(sample)[0]
            st.success(
                f"Predicted Yield using {selected}: {result:.4f}"
            )

    except Exception as e:
        st.error("Regression section encountered an error.")
        st.code(str(e))


# ==================================================
# 8. CLASSIFICATION DASHBOARD
# ==================================================

elif page == "Classification":
    st.title("🚗 Vehicle Size Classification")
    st.caption("Supervised learning · Categorical target prediction")

    try:
        with st.spinner("Training and evaluating classifiers..."):
            (
                X, y, X_test, y_test,
                models, results, predictions
            ) = train_classification()

        selected = st.selectbox(
            "Select classification algorithm",
            list(models.keys())
        )
        model = models[selected]
        pred = predictions[selected]
        metrics = results[selected]

        st.markdown(
            f'<div class="section-note"><b>Algorithm:</b> {selected}'
            f'<br><b>Evaluation:</b> 80/20 train-test split; '
            f'Random State = 42</div>',
            unsafe_allow_html=True
        )

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        b.metric("Precision", f"{metrics['Precision (weighted)']:.4f}")
        c.metric("Recall", f"{metrics['Recall (weighted)']:.4f}")
        d.metric("Weighted F1", f"{metrics['F1 (weighted)']:.4f}")

        st.subheader("Compare classification algorithms")
        st.dataframe(
            pd.DataFrame(results).T,
            use_container_width=True
        )

        tab1, tab2, tab3 = st.tabs([
            "Confusion Matrix",
            "Class-wise Metrics",
            "Feature Importance"
        ])

        classes = sorted(pd.Series(y_test).unique())

        with tab1:
            st.caption(
                f"Algorithm: {selected}. Rows are actual classes; "
                "columns are predicted classes. The diagonal shows "
                "correct predictions."
            )
            cm = confusion_matrix(y_test, pred, labels=classes)

            fig, ax = plt.subplots()
            image = ax.imshow(cm, interpolation="nearest")
            ax.set_xticks(range(len(classes)), labels=classes, rotation=45)
            ax.set_yticks(range(len(classes)), labels=classes)
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Actual Class")
            ax.set_title(f"Confusion Matrix — {selected}")
            fig.colorbar(image, ax=ax)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.caption(
                f"Algorithm: {selected}. Precision, recall and F1-score "
                "are calculated from held-out test predictions."
            )
            report = classification_report(
                y_test, pred, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.round(4), use_container_width=True)

        with tab3:
            st.caption(
                f"Algorithm: {selected}. Feature importance is extracted "
                "from the selected fitted tree-based classifier."
            )
            estimator = model.named_steps["model"]
            names = get_feature_names(model, X)

            importance = pd.DataFrame({
                "Feature": names,
                "Importance": estimator.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)
            importance = importance.sort_values("Importance")

            fig, ax = plt.subplots()
            ax.barh(importance["Feature"], importance["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {selected}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Predict vehicle size")

        inputs = {}
        left, right = st.columns(2)

        for i, col in enumerate(X.columns):
            with (left if i % 2 == 0 else right):
                if pd.api.types.is_numeric_dtype(X[col]):
                    vals = pd.to_numeric(X[col], errors="coerce").dropna()
                    default = float(vals.median()) if not vals.empty else 0.0
                    inputs[col] = st.number_input(
                        col, value=default, key=f"cls_{col}"
                    )
                else:
                    choices = sorted(X[col].dropna().astype(str).unique())
                    inputs[col] = (
                        st.selectbox(col, choices, key=f"cls_{col}")
                        if choices else ""
                    )

        if st.button("Predict Vehicle Size", type="primary"):
            sample = pd.DataFrame([inputs], columns=X.columns)
            result = model.predict(sample)[0]
            st.success(
                f"Predicted Vehicle Size using {selected}: {result}"
            )

            probabilities = model.predict_proba(sample)[0]
            probability_df = pd.DataFrame({
                "Class": model.named_steps["model"].classes_,
                "Probability": probabilities
            }).set_index("Class")

            st.caption(
                f"Class probabilities from {selected}; these are "
                "model estimates, not guaranteed real-world probabilities."
            )
            st.bar_chart(probability_df)

    except Exception as e:
        st.error("Classification section encountered an error.")
        st.code(str(e))


# ==================================================
# 9. CLUSTERING DASHBOARD
# ==================================================

elif page == "Clustering":
    st.title("🛒 Amazon Product Clustering")
    st.caption("Unsupervised learning · Product segmentation")

    try:
        with st.spinner("Calculating product clusters..."):
            (
                df, features, scaled, kmeans,
                kmeans_labels, sample_idx, agg_labels
            ) = train_clustering()

        algorithm = st.selectbox(
            "Select clustering algorithm",
            ["K-Means", "Agglomerative Clustering"]
        )

        if algorithm == "K-Means":
            labels = kmeans_labels
            plot_data = scaled
            display_df = df.copy()
        else:
            labels = agg_labels
            plot_data = scaled[sample_idx]
            display_df = df.iloc[sample_idx].copy()

        display_df["Cluster"] = labels

        st.markdown(
            f'<div class="section-note"><b>Algorithm:</b> {algorithm}'
            f'<br><b>Features:</b> {", ".join(features)}'
            '<br><b>Preprocessing:</b> median imputation and StandardScaler'
            '</div>',
            unsafe_allow_html=True
        )

        a, b, c = st.columns(3)
        a.metric("Products Analyzed", f"{len(display_df):,}")
        b.metric("Clusters", str(len(np.unique(labels))))
        c.metric("Features", str(len(features)))

        if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
            sil = silhouette_score(plot_data, labels)
            dbi = davies_bouldin_score(plot_data, labels)
            chi = calinski_harabasz_score(plot_data, labels)

            a, b, c = st.columns(3)
            a.metric("Silhouette Score", f"{sil:.4f}")
            b.metric("Davies-Bouldin Index", f"{dbi:.4f}")
            c.metric("Calinski-Harabasz", f"{chi:.2f}")

            st.caption(
                "Algorithm: selected clustering model. Higher silhouette "
                "and Calinski-Harabasz scores are generally better; a "
                "lower Davies-Bouldin score is generally better."
            )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Cluster Visualization",
            "Cluster Profiles",
            "Elbow Method",
            "Product Search"
        ])

        with tab1:
            st.caption(
                f"Clustering algorithm: {algorithm}. PCA projects the "
                "standardized features into two dimensions for display; "
                "PCA does not create the cluster labels."
            )

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(plot_data)

            fig, ax = plt.subplots()
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=labels, cmap="tab10", s=18, alpha=0.7
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"PCA View — {algorithm}")
            fig.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                f"Explained variance by the two PCA components: "
                f"{pca.explained_variance_ratio_.sum():.1%}"
            )

        with tab2:
            st.caption(
                f"Algorithm: {algorithm}. Profiles show the average "
                "standardized feature values within each cluster."
            )

            profile = display_df.groupby("Cluster")[features].mean()
            st.dataframe(profile.round(3), use_container_width=True)

            counts = display_df["Cluster"].value_counts().sort_index()

            fig, ax = plt.subplots()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of Products")
            ax.set_title(f"Cluster Sizes — {algorithm}")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.caption(
                "Algorithm: K-Means. Inertia measures within-cluster "
                "sum of squared distances. Use the elbow as a guide "
                "to choosing K, not as a definitive answer."
            )

            k_values = list(range(2, min(9, len(df))))
            inertias = []

            for k in k_values:
                km = KMeans(
                    n_clusters=k, random_state=42, n_init=10
                )
                km.fit(scaled)
                inertias.append(km.inertia_)

            if k_values:
                fig, ax = plt.subplots()
                ax.plot(k_values, inertias, marker="o")
                ax.set_xlabel("Number of Clusters (K)")
                ax.set_ylabel("Inertia")
                ax.set_title("K-Means Elbow Curve")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough rows to calculate the elbow curve.")

        with tab4:
            lookup = {c.strip().lower(): c for c in df.columns}
            name_col = lookup.get("product_name")

            if name_col is None:
                st.warning(
                    "Product name column not found. "
                    f"Available columns: {df.columns.tolist()}"
                )
            else:
                query = st.text_input("Search product name")

                matches = df[
                    df[name_col].astype(str).str.contains(
                        query, case=False, na=False
                    )
                ] if query.strip() else df

                if matches.empty:
                    st.warning("No products found.")
                else:
                    product_name = st.selectbox(
                        "Choose product",
                        matches[name_col].drop_duplicates().tolist()
                    )

                    selected_row = df[
                        df[name_col] == product_name
                    ].iloc[0]

                    # Use the selected algorithm's cluster labels.
                    if algorithm == "K-Means":
                        selected_cluster = int(
                            kmeans_labels[selected_row.name]
                        )
                        source_df = df.copy()
                        source_df["Cluster"] = kmeans_labels
                    else:
                        source_df = display_df
                        matched = source_df[
                            source_df[name_col] == product_name
                        ]
                        if matched.empty:
                            st.info(
                                "This product is outside the sample used "
                                "for Agglomerative Clustering. Search for "
                                "another product or choose K-Means."
                            )
                            st.stop()
                        selected_cluster = int(
                            matched.iloc[0]["Cluster"]
                        )

                    st.write(product_name)
                    st.metric("Cluster", selected_cluster)

                    price_col = lookup.get("discounted_price")
                    actual_col = lookup.get("actual_price")
                    rating_col = lookup.get("rating")

                    c1, c2, c3 = st.columns(3)

                    if price_col:
                        v = selected_row[price_col]
                        c1.metric(
                            "Discounted Price",
                            f"₹{v:,.2f}" if pd.notna(v) else "N/A"
                        )

                    if actual_col:
                        v = selected_row[actual_col]
                        c2.metric(
                            "Actual Price",
                            f"₹{v:,.2f}" if pd.notna(v) else "N/A"
                        )

                    if rating_col:
                        v = selected_row[rating_col]
                        c3.metric(
                            "Rating",
                            f"{v:.1f}/5" if pd.notna(v) else "N/A"
                        )

                    st.caption(
                        f"Products below are assigned to the same "
                        f"{algorithm} cluster."
                    )

                    similar = source_df[
                        (source_df["Cluster"] == selected_cluster) &
                        (source_df[name_col] != product_name)
                    ]

                    st.dataframe(similar.head(10), use_container_width=True)

        st.info(
            "Important: this Amazon dataset provides product attributes, "
            "not a time series of sales. The dashboard identifies product "
            "segments; it does not claim to forecast product trends over time."
        )

    except Exception as e:
        st.error("Clustering section encountered an error.")
        st.code(str(e))
