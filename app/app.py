
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Studio",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #315b82);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {color: white;}
div[data-testid="stMetric"] {
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        available = (
            [p.name for p in DATA_DIR.glob("*.csv")]
            if DATA_DIR.exists() else []
        )
        raise FileNotFoundError(
            f"Cannot find {filename}. CSV files found: {available}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{filename} is empty. Upload the original CSV to the data folder."
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"{filename} contains no readable columns or rows. "
            "Check the file in GitHub."
        )

    if df.empty or len(df.columns) == 0:
        raise ValueError(f"{filename} contains no usable data.")

    df.columns = df.columns.astype(str).str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def make_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore", sparse=False
        )


def make_preprocessor(X, scale=True):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []

    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric", Pipeline(steps), numeric
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_encoder())
            ]),
            categorical
        ))

    return ColumnTransformer(transformers, remainder="drop")


def numeric_inputs(X, prefix):
    """Create input fields from the dataset's actual feature columns."""
    values = {}

    left, right = st.columns(2)

    for i, col in enumerate(X.columns):
        with (left if i % 2 == 0 else right):
            series = X[col]

            if pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(series, errors="coerce").dropna()
                default = float(clean.median()) if not clean.empty else 0.0

                values[col] = st.number_input(
                    col.replace("_", " "),
                    value=default,
                    key=f"{prefix}_{col}"
                )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())

                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " "),
                        options,
                        key=f"{prefix}_{col}"
                    )
                else:
                    values[col] = ""

    return pd.DataFrame([values], columns=X.columns)


def note(message):
    st.info(message)


# --------------------------------------------------
# REGRESSION: KNN REGRESSOR
# --------------------------------------------------
@st.cache_resource
def train_regression():
    filename = next(
        (
            p.name for p in DATA_DIR.glob("*.csv")
            if "".join(c.lower() for c in p.name if c.isalnum())
            in {"cropyeilddatacsv", "cropyielddatacsv"}
        ),
        None
    )

    if filename is None:
        raise FileNotFoundError(
            "Crop yield CSV not found in data/. Check its filename."
        )

    df = load_csv(filename)

    if "Yield" not in df.columns:
        raise ValueError(
            f"Target 'Yield' not found. Columns: {df.columns.tolist()}"
        )

    df["Yield"] = clean_numeric(df["Yield"])
    df = df.drop_duplicates().dropna(subset=["Yield"]).copy()

    # Exclude Production to avoid target leakage.
    X = df.drop(
        columns=[c for c in ["Yield", "Production"] if c in df.columns]
    ).copy()
    y = df["Yield"].astype(float)

    for col, area in [
        ("Fertilizer", "Fertilizer_per_Area"),
        ("Pesticide", "Pesticide_per_Area")
    ]:
        if col in X.columns and "Area" in X.columns:
            X[area] = X[col] / X["Area"].replace(0, np.nan)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if len(X) < 5:
        raise ValueError("Not enough valid rows to train regression.")

    stratify = (
        X["Season"]
        if "Season" in X.columns
        and X["Season"].value_counts().min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        ("preprocess", make_preprocessor(X, scale=True)),
        ("model", KNeighborsRegressor(n_neighbors=min(5, len(X_train))))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MAE": mean_absolute_error(y_test, predictions)
    }

    return X, X_train, X_test, y_train, y_test, model, predictions, metrics


def regression_page():
    st.markdown(
        '<div class="hero"><h1>🌾 Crop Yield Prediction</h1>'
        '<p>K-Nearest Neighbors Regression</p></div>',
        unsafe_allow_html=True
    )

    X, X_train, X_test, y_train, y_test, model, pred, metrics = (
        train_regression()
    )

    a, b, c = st.columns(3)
    a.metric("R² score", f"{metrics['R²']:.4f}")
    b.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c.metric("MAE", f"{metrics['MAE']:.3f}")

    st.subheader("1. Enter crop conditions")
    sample = numeric_inputs(X, "crop")

    if st.button("Predict crop yield", type="primary"):
        result = float(model.predict(sample)[0])
        st.session_state["crop_prediction"] = result

    if "crop_prediction" in st.session_state:
        st.success(
            f"Estimated crop yield: "
            f"{st.session_state['crop_prediction']:,.3f}"
        )

    tabs = st.tabs([
        "Actual vs Predicted", "Residuals",
        "Model Calculation", "Evaluation"
    ])

    with tabs[0]:
        fig, ax = plt.subplots()
        ax.scatter(y_test, pred, alpha=0.6)
        lo = min(float(y_test.min()), float(np.min(pred)))
        hi = max(float(y_test.max()), float(np.max(pred)))
        ax.plot([lo, hi], [lo, hi], "--")
        ax.set_xlabel("Actual yield")
        ax.set_ylabel("Predicted yield")
        ax.set_title("Actual vs Predicted Yield")
        st.pyplot(fig)
        plt.close(fig)

        note(
            "Dots closer to the dashed line indicate more accurate "
            "predictions. Distance from the line indicates prediction error."
        )

    with tabs[1]:
        residuals = np.asarray(y_test) - pred
        fig, ax = plt.subplots()
        ax.scatter(pred, residuals, alpha=0.6)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted yield")
        ax.set_ylabel("Residual: actual − predicted")
        ax.set_title("Residual Analysis")
        st.pyplot(fig)
        plt.close(fig)

        note(
            "Positive residuals mean the model predicted too low; "
            "negative residuals mean it predicted too high."
        )

    with tabs[2]:
        st.markdown("### How KNN calculates the prediction")

        processed_train = model.named_steps["preprocess"].transform(X_train)
        processed_sample = model.named_steps["preprocess"].transform(sample)

        distances = np.sqrt(
            np.sum((processed_train - processed_sample) ** 2, axis=1)
        )
        nearest = np.argsort(distances)[:model.named_steps["model"].n_neighbors]

        details = pd.DataFrame({
            "Neighbor": np.arange(1, len(nearest) + 1),
            "Distance": distances[nearest],
            "Actual training yield": y_train.iloc[nearest].to_numpy()
        })

        st.dataframe(details.round(4), use_container_width=True)

        st.latex(
            r"\hat{y}=\frac{1}{k}\sum_{i=1}^{k} y_i"
        )
        st.write(
            "KNN finds the closest training examples after preprocessing. "
            "Their yields are averaged to produce the prediction."
        )

    with tabs[3]:
        st.dataframe(
            pd.DataFrame([metrics]).round(4),
            use_container_width=True
        )
        note(
            "R² measures explained variation. RMSE penalizes large errors "
            "more heavily, while MAE gives the average absolute error."
        )


# --------------------------------------------------
# CLASSIFICATION: GRADIENT BOOSTING
# --------------------------------------------------
@st.cache_resource
def train_classification():
    df = load_csv("cars.csv").drop_duplicates().copy()

    if "Vehicle Size" not in df.columns:
        raise ValueError(
            f"Target 'Vehicle Size' not found. Columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["Vehicle Size"])
    y = df["Vehicle Size"].astype(str)

    X = df.drop(
        columns=[c for c in ["Vehicle Size", "Model"] if c in df.columns]
    )
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if y.nunique() < 2:
        raise ValueError("Classification needs at least two classes.")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        ("preprocess", make_preprocessor(X, scale=False)),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
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

    return X, X_test, y_test, model, pred, metrics


def classification_page():
    st.markdown(
        '<div class="hero"><h1>🚘 Vehicle Classification</h1>'
        '<p>Gradient Boosting Classifier</p></div>',
        unsafe_allow_html=True
    )

    X, X_test, y_test, model, pred, metrics = train_classification()

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    b.metric("Precision", f"{metrics['Precision (weighted)']:.3f}")
    c.metric("Recall", f"{metrics['Recall (weighted)']:.3f}")
    d.metric("Weighted F1", f"{metrics['F1 (weighted)']:.3f}")

    st.subheader("1. Enter vehicle details")
    sample = numeric_inputs(X, "car")

    if st.button("Predict vehicle size", type="primary"):
        st.session_state["car_prediction"] = model.predict(sample)[0]
        try:
            st.session_state["car_probabilities"] = model.predict_proba(sample)[0]
        except Exception:
            st.session_state["car_probabilities"] = None

    if "car_prediction" in st.session_state:
        st.success(
            f"Predicted vehicle size: {st.session_state['car_prediction']}"
        )

        probabilities = st.session_state.get("car_probabilities")
        if probabilities is not None:
            classes = model.named_steps["model"].classes_
            st.bar_chart(
                pd.DataFrame({
                    "Vehicle class": classes,
                    "Probability": probabilities
                }).set_index("Vehicle class")
            )

    tabs = st.tabs([
        "Confusion Matrix", "Class Metrics",
        "Feature Importance", "How it Works"
    ])

    with tabs[0]:
        classes = model.named_steps["model"].classes_
        cm = confusion_matrix(y_test, pred, labels=classes)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), labels=classes, rotation=25)
        ax.set_yticks(range(len(classes)), labels=classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        note(
            "Diagonal cells are correct predictions. Off-diagonal cells "
            "show which vehicle categories the model confused."
        )

    with tabs[1]:
        report = classification_report(
            y_test, pred, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3))

    with tabs[2]:
        model_step = model.named_steps["model"]
        transformed = model.named_steps["preprocess"]
        try:
            names = transformed.get_feature_names_out()
            importance = model_step.feature_importances_

            fi = pd.DataFrame({
                "Feature": names,
                "Importance": importance
            }).sort_values("Importance", ascending=False).head(15)

            st.bar_chart(fi.set_index("Feature"))
            st.dataframe(fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

    with tabs[3]:
        st.markdown("""
        **Gradient Boosting calculation**

        1. The preprocessing pipeline fills missing values and encodes
           categorical inputs.
        2. A sequence of decision trees learns patterns in the training data.
        3. Each successive tree helps correct errors made by earlier trees.
        4. The combined model produces the predicted vehicle-size class.

        **Accuracy** measures the fraction of correct predictions.
        **Precision** measures prediction reliability.
        **Recall** measures how many actual examples were identified.
        **F1** balances precision and recall.
        """)


# --------------------------------------------------
# CLUSTERING: AMAZON PRODUCTS
# --------------------------------------------------
@st.cache_resource
def prepare_amazon():
    df = load_csv("amazon.csv").copy()

    required = [
        "product_name", "category", "discounted_price",
        "actual_price", "discount_percentage", "rating", "rating_count"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing Amazon columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in [
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]:
        df[col] = clean_numeric(df[col])

    features = pd.DataFrame({
        "actual_price": np.log1p(df["actual_price"].clip(lower=0)),
        "discounted_price": np.log1p(df["discounted_price"].clip(lower=0)),
        "discount_percentage": df["discount_percentage"],
        "rating": df["rating"],
        "rating_count": np.log1p(df["rating_count"].clip(lower=0))
    })

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return df, features, scaled, scaler


@st.cache_resource
def train_clusters():
    df, features, scaled, scaler = prepare_amazon()

    # Cap K-Means training size to keep Streamlit Cloud responsive.
    rng = np.random.RandomState(42)
    sample_size = min(len(df), 5000)
    idx = np.sort(rng.choice(len(df), sample_size, replace=False))
    sample_X = scaled[idx]

    if len(sample_X) < 4:
        raise ValueError("Not enough Amazon products for clustering.")

    results = []
    models = {}

    for k in [3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(sample_X)

        models[f"K-Means (k={k})"] = {
            "model": km, "labels": labels, "indices": idx,
            "X": sample_X
        }

        results.append({
            "Algorithm": f"K-Means (k={k})",
            "Silhouette": silhouette_score(sample_X, labels),
            "Davies-Bouldin": davies_bouldin_score(sample_X, labels),
            "Calinski-Harabasz": calinski_harabasz_score(sample_X, labels)
        })

    # Agglomerative clustering is fitted on the same representative sample.
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    agg_labels = agg.fit_predict(sample_X)

    models["Agglomerative (Ward, k=3)"] = {
        "model": agg, "labels": agg_labels, "indices": idx,
        "X": sample_X
    }

    results.append({
        "Algorithm": "Agglomerative (Ward, k=3)",
        "Silhouette": silhouette_score(sample_X, agg_labels),
        "Davies-Bouldin": davies_bouldin_score(sample_X, agg_labels),
        "Calinski-Harabasz": calinski_harabasz_score(sample_X, agg_labels)
    })

    return df, features, scaled, scaler, models, pd.DataFrame(results)


def clustering_page():
    st.markdown(
        '<div class="hero"><h1>🛍️ Amazon Product Segmentation</h1>'
        '<p>Search products, explore clusters, and understand their profiles.</p></div>',
        unsafe_allow_html=True
    )

    df, features, scaled, scaler, models, metrics = train_clusters()

    selected = st.selectbox(
        "Select clustering algorithm",
        list(models.keys())
    )
    info = models[selected]
    labels = info["labels"]
    idx = info["indices"]
    sample_df = df.iloc[idx].copy()
    sample_df["Cluster"] = labels

    row = metrics.set_index("Algorithm").loc[selected]

    a, b, c = st.columns(3)
    a.metric("Silhouette", f"{row['Silhouette']:.3f}")
    b.metric("Davies–Bouldin", f"{row['Davies-Bouldin']:.3f}")
    c.metric("Calinski–Harabasz", f"{row['Calinski-Harabasz']:.1f}")

    note(
        "Silhouette: higher is generally better. "
        "Davies–Bouldin: lower is generally better. "
        "Calinski–Harabasz: higher is generally better. "
        "These metrics are guides, not proof of a perfect clustering."
    )

    tabs = st.tabs([
        "Product Search", "PCA Visualization",
        "Cluster Profiles", "Algorithm Comparison", "How it Works"
    ])

    with tabs[0]:
        query = st.text_input("Search Amazon product name")

        matches = df[
            df["product_name"].astype(str).str.contains(
                query, case=False, na=False
            )
        ] if query.strip() else df.head(100)

        if matches.empty:
            st.warning("No matching products found.")
        else:
            product = st.selectbox(
                "Choose product",
                matches["product_name"].drop_duplicates().tolist()
            )

            selected_row = df[df["product_name"] == product].iloc[0]
            st.subheader(product)

            cols = st.columns(4)
            cols[0].metric(
                "Selling price",
                f"{selected_row['discounted_price']:,.2f}"
                if pd.notna(selected_row["discounted_price"]) else "N/A"
            )
            cols[1].metric(
                "Actual price",
                f"{selected_row['actual_price']:,.2f}"
                if pd.notna(selected_row["actual_price"]) else "N/A"
            )
            cols[2].metric(
                "Discount",
                f"{selected_row['discount_percentage']}%"
            )
            cols[3].metric(
                "Rating",
                str(selected_row["rating"])
            )

            locations = np.where(
                sample_df["product_name"].to_numpy() == product
            )[0]

            if len(locations):
                cluster = int(labels[locations[0]])
                st.success(f"Cluster assigned by {selected}: {cluster}")

                similar = sample_df[
                    sample_df["Cluster"] == cluster
                ]
                similar = similar[
                    similar["product_name"] != product
                ]

                st.subheader("Products in the same cluster")
                st.dataframe(
                    similar[[
                        "product_name", "category", "discounted_price",
                        "actual_price", "discount_percentage",
                        "rating", "rating_count"
                    ]].head(15),
                    use_container_width=True
                )
            else:
                st.info(
                    "This product was not included in the clustering sample. "
                    "Try another product from the sample or reduce dataset size."
                )

    with tabs[1]:
        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(info["X"])

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=labels, cmap="tab10", alpha=0.65, s=16
        )
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"PCA Cluster Map: {selected}")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
        plt.close(fig)

        st.write(
            f"The first two PCA components preserve "
            f"{pca.explained_variance_ratio_.sum() * 100:.1f}% "
            "of the variance in the standardized features."
        )
        note(
            "Each dot represents a product. Nearby points have similar "
            "feature profiles in this 2D projection. PCA visualizes the "
            "clusters; it does not create the cluster labels."
        )

    with tabs[2]:
        cols = [
            "discounted_price", "actual_price",
            "discount_percentage", "rating", "rating_count"
        ]
        profile = sample_df.groupby("Cluster")[cols].median()
        profile["Product count"] = sample_df.groupby("Cluster").size()

        st.subheader("Median feature values per cluster")
        st.dataframe(profile.round(2), use_container_width=True)

        st.bar_chart(sample_df["Cluster"].value_counts().sort_index())

        note(
            "Compare median prices, discounts, ratings, and review counts "
            "to describe the type of products in each cluster."
        )

    with tabs[3]:
        st.dataframe(metrics.round(4), use_container_width=True)

        fig, ax = plt.subplots()
        ax.bar(metrics["Algorithm"], metrics["Silhouette"])
        ax.set_ylabel("Silhouette score")
        ax.set_title("Clustering configuration comparison")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig)
        plt.close(fig)

    with tabs[4]:
        st.markdown("""
        **Features used:** log-transformed actual price, selling price,
        discount percentage, rating, and log-transformed review count.

        **Preprocessing:** missing values are filled using feature medians,
        then features are standardized.

        **K-Means:** starts with cluster centers and assigns products to
        the nearest center. It updates the centers and repeats until
        assignments stabilize.

        **Agglomerative Clustering:** starts with individual products
        and repeatedly merges nearby groups using Ward linkage.

        **Important limitation:** this dataset describes product attributes
        and review counts. It does not contain historical measurements
        by date, so these clusters do not predict future product trends.
        """)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.title("📊 ML Capstone Studio")
    st.caption("Review 2 · Interactive dashboard")
    page = st.radio(
        "Choose a module",
        ["Overview", "Regression", "Classification", "Amazon Clustering"]
    )

if page == "Overview":
    st.markdown(
        '<div class="hero"><h1>Machine Learning Capstone Studio</h1>'
        '<p>Predict, classify, cluster, and understand the calculations.</p></div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)
    a.metric("ML tasks", "3")
    b.metric("Supervised tasks", "2")
    c.metric("Unsupervised tasks", "1")

    st.subheader("Explore the dashboard")
    st.write("🌾 **Regression:** estimate crop yield using KNN.")
    st.write("🚘 **Classification:** classify vehicle size using Gradient Boosting.")
    st.write("🛍️ **Clustering:** explore Amazon product segments.")
    st.info("Select a task in the sidebar to enter details and view predictions.")

elif page == "Regression":
    try:
        regression_page()
    except Exception as e:
        st.error("Regression could not be loaded.")
        st.exception(e)

elif page == "Classification":
    try:
        classification_page()
    except Exception as e:
        st.error("Classification could not be loaded.")
        st.exception(e)

else:
    try:
        clustering_page()
    except Exception as e:
        st.error("Amazon clustering could not be loaded.")
        st.exception(e)
