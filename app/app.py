
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

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ML Capstone Studio",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #14213d, #315b82);
    color: white;
    margin-bottom: 22px;
}
.hero h1 { color: white; }
div[data-testid="stMetric"] {
    border: 1px solid #d9e2ec;
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SHARED UTILITIES
# =========================================================
@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        available = (
            [p.name for p in DATA_DIR.glob("*.csv")]
            if DATA_DIR.exists() else []
        )
        raise FileNotFoundError(
            f"Cannot find {filename} in {DATA_DIR}. "
            f"Available CSV files: {available}"
        )

    if path.stat().st_size == 0:
        raise ValueError(f"{filename} is empty.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{filename} has no data rows.")

    df.columns = df.columns.astype(str).str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor(X, scale=True):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:
        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if scale:
            steps.append(("scaler", StandardScaler()))

        transformers.append((
            "numeric", Pipeline(steps), numeric_cols
        ))

    if categorical_cols:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent"
                )),
                ("encoder", encoder())
            ]),
            categorical_cols
        ))

    return ColumnTransformer(
        transformers,
        remainder="drop"
    )


def input_form(X, prefix):
    values = {}
    left, right = st.columns(2)

    for i, col in enumerate(X.columns):
        with (left if i % 2 == 0 else right):
            series = X[col]

            if pd.api.types.is_numeric_dtype(series):
                clean = pd.to_numeric(
                    series, errors="coerce"
                ).dropna()

                default = (
                    float(clean.median())
                    if not clean.empty else 0.0
                )

                values[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=default,
                    key=f"{prefix}_{col}"
                )
            else:
                options = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if options:
                    values[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options,
                        key=f"{prefix}_{col}"
                    )
                else:
                    values[col] = ""

    return pd.DataFrame([values], columns=X.columns)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


# =========================================================
# REGRESSION: CROP YIELD
# =========================================================
@st.cache_resource
def train_regression():
    candidates = [
        p.name for p in DATA_DIR.glob("*.csv")
        if "".join(
            ch.lower() for ch in p.name if ch.isalnum()
        ) in {
            "cropyeilddatacsv",
            "cropyielddatacsv"
        }
    ]

    if not candidates:
        raise FileNotFoundError(
            "Crop yield CSV not found in data/. "
            "Check the exact filename."
        )

    df = load_csv(candidates[0])

    if "Yield" not in df.columns:
        raise ValueError(
            f"Expected target 'Yield'. Found: {df.columns.tolist()}"
        )

    df["Yield"] = numeric(df["Yield"])
    df = df.dropna(subset=["Yield"]).drop_duplicates()

    # Production can directly reveal yield and cause target leakage.
    drop_cols = [
        c for c in ["Yield", "Production"]
        if c in df.columns
    ]

    X = df.drop(columns=drop_cols).copy()
    y = df["Yield"].astype(float)

    # Recreate the derived features if those were used in training.
    for source, derived in [
        ("Fertilizer", "Fertilizer_per_Area"),
        ("Pesticide", "Pesticide_per_Area")
    ]:
        if source in X.columns and "Area" in X.columns:
            X[derived] = (
                numeric(X[source]) /
                numeric(X["Area"]).replace(0, np.nan)
            )

    X = X.drop(
        columns=[
            c for c in ["Fertilizer", "Pesticide"]
            if c in X.columns
        ],
        errors="ignore"
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if len(X) < 5:
        raise ValueError("Not enough rows for regression.")

    stratify = None
    if "Season" in X.columns:
        if X["Season"].value_counts().min() >= 2:
            stratify = X["Season"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    k = min(5, len(X_train))

    model = Pipeline([
        ("preprocess", make_preprocessor(X, scale=True)),
        ("model", KNeighborsRegressor(n_neighbors=k))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred)
    }

    return X, X_train, X_test, y_train, y_test, model, pred, metrics


def regression_page():
    hero(
        "🌾 Crop Yield Prediction",
        "Estimate crop yield and understand the result."
    )

    (
        X, X_train, X_test, y_train, y_test,
        model, predictions, metrics
    ) = train_regression()

    st.subheader("Enter crop details")
    sample = input_form(X, "crop")

    if st.button("Predict Crop Yield", type="primary"):
        st.session_state["crop_result"] = float(
            model.predict(sample)[0]
        )
        st.session_state["crop_input"] = sample.copy()

    if "crop_result" in st.session_state:
        st.success(
            f"Estimated crop yield: "
            f"{st.session_state['crop_result']:,.3f}"
        )

        st.caption(
            "Calculated using the K-Nearest Neighbors Regressor. "
            "It finds similar training examples and combines "
            "their known yields."
        )

        st.markdown("### How was the prediction calculated?")

        chosen_input = st.session_state["crop_input"]
        preprocess = model.named_steps["preprocess"]

        train_matrix = preprocess.transform(X_train)
        input_matrix = preprocess.transform(chosen_input)

        distances = np.sqrt(
            np.sum(
                (train_matrix - input_matrix) ** 2,
                axis=1
            )
        )

        k = model.named_steps["model"].n_neighbors
        nearest = np.argsort(distances)[:k]

        details = pd.DataFrame({
            "Neighbor": np.arange(1, len(nearest) + 1),
            "Distance": distances[nearest],
            "Training yield": y_train.iloc[nearest].to_numpy()
        })

        st.dataframe(
            details.round(4),
            use_container_width=True
        )

        st.latex(r"\hat{y}=\frac{1}{k}\sum_{i=1}^{k}y_i")

        st.write(
            "The input is transformed using the same preprocessing "
            "as the training data. The model finds the closest "
            "training samples and averages their yields."
        )

    st.divider()
    st.subheader("Model evaluation")

    a, b, c = st.columns(3)
    a.metric("R² Score", f"{metrics['R²']:.4f}")
    b.metric("RMSE", f"{metrics['RMSE']:.3f}")
    c.metric("MAE", f"{metrics['MAE']:.3f}")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Actual vs Predicted Yield")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.65)

        low = min(float(y_test.min()), float(predictions.min()))
        high = max(float(y_test.max()), float(predictions.max()))

        ax.plot([low, high], [low, high], "--")
        ax.set_xlabel("Actual yield")
        ax.set_ylabel("Predicted yield")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Points closer to the dashed line represent "
            "more accurate predictions."
        )

    with right:
        st.markdown("#### Residual Analysis")

        residuals = np.asarray(y_test) - predictions

        fig, ax = plt.subplots()
        ax.scatter(predictions, residuals, alpha=0.65)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted yield")
        ax.set_ylabel("Actual − predicted")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive residuals indicate underprediction; "
            "negative residuals indicate overprediction."
        )

    st.markdown("#### Evaluation metric interpretation")
    st.write(
        "**R²:** closer to 1 is generally better. "
        "**RMSE:** penalizes larger errors. "
        "**MAE:** average absolute prediction error. "
        "RMSE and MAE are expressed in the target's units."
    )


# =========================================================
# CLASSIFICATION: VEHICLE SIZE
# =========================================================
@st.cache_resource
def train_classification():
    df = load_csv("cars.csv").drop_duplicates().copy()

    target = "Vehicle Size"

    if target not in df.columns:
        raise ValueError(
            f"Expected target '{target}'. "
            f"Found: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[target])
    y = df[target].astype(str)

    X = df.drop(
        columns=[
            c for c in [target, "Model"]
            if c in df.columns
        ]
    ).copy()

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if y.nunique() < 2:
        raise ValueError("Classification needs at least two classes.")

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        ("preprocess", make_preprocessor(X, scale=False)),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions,
            average="weighted", zero_division=0
        )
    }

    return X, X_test, y_test, model, predictions, metrics


def classification_page():
    hero(
        "🚘 Vehicle Classification",
        "Enter vehicle details to estimate its category."
    )

    X, X_test, y_test, model, predictions, metrics = (
        train_classification()
    )

    st.subheader("Enter vehicle details")
    sample = input_form(X, "vehicle")

    if st.button("Predict Vehicle Category", type="primary"):
        result = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]

        st.session_state["vehicle_result"] = result
        st.session_state["vehicle_probabilities"] = probabilities
        st.session_state["vehicle_input"] = sample.copy()

    if "vehicle_result" in st.session_state:
        st.success(
            f"Predicted vehicle category: "
            f"{st.session_state['vehicle_result']}"
        )

        st.caption(
            "Calculated using the Gradient Boosting Classifier. "
            "The model combines successive decision trees to "
            "produce the final class prediction."
        )

        classes = model.named_steps["model"].classes_
        probabilities = st.session_state["vehicle_probabilities"]

        st.markdown("### Prediction probabilities")

        prob_df = pd.DataFrame({
            "Category": classes,
            "Probability": probabilities
        }).set_index("Category")

        st.bar_chart(prob_df)

        st.caption(
            "Probabilities indicate the model's estimated support "
            "for each class; they are not guarantees of correctness."
        )

    st.divider()
    st.subheader("Model evaluation")

    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

    left, right = st.columns(2)

    classes = model.named_steps["model"].classes_
    cm = confusion_matrix(y_test, predictions, labels=classes)

    with left:
        st.markdown("#### Confusion Matrix")

        fig, ax = plt.subplots()
        image = ax.imshow(cm, cmap="Blues")

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "The diagonal contains correct predictions. "
            "Other cells indicate misclassified vehicles."
        )

    with right:
        st.markdown("#### Feature Importance")

        fitted = model.named_steps["model"]
        names = model.named_steps[
            "preprocess"
        ].get_feature_names_out()

        importance = pd.DataFrame({
            "Feature": names,
            "Importance": fitted.feature_importances_
        }).sort_values(
            "Importance", ascending=False
        ).head(12)

        st.bar_chart(
            importance.set_index("Feature")
        )

        st.caption(
            "Higher feature importance means the feature contributed "
            "more to the fitted model's decisions. It does not prove "
            "that the feature causes the outcome."
        )

    st.markdown("#### Detailed classification report")

    report = classification_report(
        y_test, predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )


# =========================================================
# AMAZON PRODUCT CLUSTERING
# =========================================================
AMAZON_FEATURES = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count"
]


@st.cache_resource
def train_amazon():
    df = load_csv("amazon.csv").copy()

    required = [
        "product_name", "category",
        "discounted_price", "actual_price",
        "discount_percentage", "rating", "rating_count"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Amazon CSV is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for col in AMAZON_FEATURES:
        df[col] = numeric(df[col])

    df = df.reset_index(drop=True)

    # Retain rows with usable product identity and feature values.
    df = df.dropna(
        subset=["product_name", "actual_price", "discounted_price"],
    ).reset_index(drop=True)

    if len(df) < 4:
        raise ValueError("Not enough usable Amazon products.")

    # Log-transform skewed price and review-count columns.
    features = pd.DataFrame({
        "Actual price": np.log1p(
            df["actual_price"].clip(lower=0)
        ),
        "Discounted price": np.log1p(
            df["discounted_price"].clip(lower=0)
        ),
        "Discount percentage": df["discount_percentage"],
        "Rating": df["rating"],
        "Review count": np.log1p(
            df["rating_count"].clip(lower=0)
        )
    })

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Fit one deployable K-Means model on ALL usable products.
    # This means every product can be assigned to a cluster.
    k = min(4, max(2, int(np.sqrt(len(df) / 2))))
    k = min(k, len(df) - 1)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=5
    )

    labels = kmeans.fit_predict(scaled)

    df["Cluster"] = labels

    # Compare algorithms on a representative sample for metrics.
    rng = np.random.RandomState(42)
    sample_size = min(len(df), 2000)

    sample_indices = np.sort(
        rng.choice(len(df), sample_size, replace=False)
    )

    sample_X = scaled[sample_indices]
    results = []
    cluster_models = {}

    # Score K-Means on the sample using its own fit for fair comparison.
    for sample_k in sorted(set([2, min(3, sample_size - 1), min(4, sample_size - 1)])):
        if sample_k < 2 or sample_k >= sample_size:
            continue

        km = KMeans(
            n_clusters=sample_k,
            random_state=42,
            n_init=5
        )

        sample_labels = km.fit_predict(sample_X)

        if len(np.unique(sample_labels)) < 2:
            continue

        results.append({
            "Algorithm": f"K-Means (k={sample_k})",
            "Silhouette": silhouette_score(sample_X, sample_labels),
            "Davies-Bouldin": davies_bouldin_score(
                sample_X, sample_labels
            ),
            "Calinski-Harabasz": calinski_harabasz_score(
                sample_X, sample_labels
            )
        })

        cluster_models[f"K-Means (k={sample_k})"] = km

    # Agglomerative evaluation uses the same sample.
    agg_k = min(3, sample_size - 1)

    if agg_k >= 2:
        agg = AgglomerativeClustering(
            n_clusters=agg_k,
            linkage="ward"
        )

        agg_labels = agg.fit_predict(sample_X)

        if len(np.unique(agg_labels)) >= 2:
            results.append({
                "Algorithm": f"Agglomerative (k={agg_k})",
                "Silhouette": silhouette_score(
                    sample_X, agg_labels
                ),
                "Davies-Bouldin": davies_bouldin_score(
                    sample_X, agg_labels
                ),
                "Calinski-Harabasz": calinski_harabasz_score(
                    sample_X, agg_labels
                )
            })

    return (
        df, features, scaled, scaler,
        kmeans, labels, sample_X, pd.DataFrame(results)
    )


def clustering_page():
    hero(
        "🛍️ Amazon Product Explorer",
        "Search a product, discover its segment, and explore the data."
    )

    (
        df, features, scaled, scaler,
        kmeans, labels, sample_X, comparison
    ) = train_amazon()

    st.subheader("Find a product")

    query = st.text_input(
        "Search by product name",
        placeholder="Type a product name..."
    )

    if query.strip():
        matches = df[
            df["product_name"].astype(str).str.contains(
                query, case=False, na=False
            )
        ]

        if matches.empty:
            st.warning(
                "No product matched that search. Try a shorter keyword."
            )
            selected_product = None
        else:
            selected_product = st.selectbox(
                "Choose a matching product",
                matches["product_name"].drop_duplicates().tolist()
            )
    else:
        selected_product = None
        st.info("Enter a product name to explore its analysis.")

    if selected_product:
        product_row = df[
            df["product_name"] == selected_product
        ].iloc[0]

        product_index = product_row.name
        cluster = int(labels[product_index])

        st.divider()
        st.subheader("Product analysis")

        st.success(
            f"This product belongs to product segment {cluster + 1}."
        )

        st.caption(
            "The segment was calculated using K-Means clustering. "
            "Products are grouped by their price, discount, rating, "
            "and review-count patterns."
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Selling price",
            f"{product_row['discounted_price']:,.2f}"
        )
        b.metric(
            "Actual price",
            f"{product_row['actual_price']:,.2f}"
        )
        c.metric(
            "Discount",
            f"{product_row['discount_percentage']:.1f}%"
            if pd.notna(product_row["discount_percentage"])
            else "N/A"
        )
        d.metric(
            "Rating",
            f"{product_row['rating']:.1f}"
            if pd.notna(product_row["rating"])
            else "N/A"
        )

        # Similar products from the same cluster.
        similar = df[
            (df["Cluster"] == cluster) &
            (df["product_name"] != selected_product)
        ].copy()

        st.markdown("### Similar products")

        display_cols = [
            "product_name", "category",
            "actual_price", "discounted_price",
            "discount_percentage", "rating", "rating_count"
        ]

        st.dataframe(
            similar[display_cols].head(10),
            use_container_width=True
        )

        st.caption(
            "These products share a cluster, which means their selected "
            "attributes are similar. This is not necessarily a measure "
            "of semantic similarity or purchase intent."
        )

        st.divider()
        st.subheader("Visualizations")

        # Price distribution.
        left, right = st.columns(2)

        with left:
            st.markdown("#### Selling price distribution")

            fig, ax = plt.subplots()
            prices = df["discounted_price"].dropna()

            ax.hist(prices, bins=25)
            ax.axvline(
                product_row["discounted_price"],
                linestyle="--",
                label="Selected product"
            )

            ax.set_xlabel("Discounted price")
            ax.set_ylabel("Number of products")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "The dashed line shows where this product's selling "
                "price falls in the dataset."
            )

        with right:
            st.markdown("#### Rating distribution")

            fig, ax = plt.subplots()
            ratings = df["rating"].dropna()

            ax.hist(ratings, bins=10)
            ax.axvline(
                product_row["rating"],
                linestyle="--",
                label="Selected product"
            )

            ax.set_xlabel("Rating")
            ax.set_ylabel("Number of products")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "This compares the product's rating with other "
                "products' ratings."
            )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Discount distribution")

            fig, ax = plt.subplots()
            discounts = df["discount_percentage"].dropna()

            ax.hist(discounts, bins=15)
            ax.axvline(
                product_row["discount_percentage"],
                linestyle="--",
                label="Selected product"
            )

            ax.set_xlabel("Discount percentage")
            ax.set_ylabel("Number of products")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("#### Product counts by segment")

            counts = df["Cluster"].value_counts().sort_index()
            counts.index = [
                f"Segment {i + 1}" for i in counts.index
            ]

            st.bar_chart(counts)

        # PCA uses all scaled product rows and the labels from K-Means.
        st.markdown("#### Product segment map")

        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(scaled)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            cmap="tab10",
            alpha=0.55,
            s=15
        )

        selected_xy = coordinates[product_index]

        ax.scatter(
            selected_xy[0],
            selected_xy[1],
            marker="*",
            s=250,
            edgecolors="black",
            label="Selected product"
        )

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Product Segments Visualized with PCA")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        explained = pca.explained_variance_ratio_.sum() * 100

        st.caption(
            f"The first two PCA components preserve {explained:.1f}% "
            "of the variance. Each point represents a product; the star "
            "marks the selected product."
        )

        # Cluster profile.
        st.markdown("#### Segment characteristics")

        profile_cols = [
            "actual_price", "discounted_price",
            "discount_percentage", "rating", "rating_count"
        ]

        profile = df.groupby("Cluster")[profile_cols].median()
        profile["Product count"] = df.groupby("Cluster").size()

        profile.index = [
            f"Segment {i + 1}" for i in profile.index
        ]

        st.dataframe(
            profile.round(2),
            use_container_width=True
        )

        st.markdown("#### How was the segment calculated?")

        st.write(
            "1. Price and review-count values are log-transformed to "
            "reduce skew from very large values."
        )
        st.write(
            "2. Missing feature values are filled using column medians."
        )
        st.write(
            "3. Features are standardized so values with large numeric "
            "ranges do not dominate the distance calculation."
        )
        st.write(
            "4. K-Means assigns products to the nearest cluster center "
            "and updates the centers to reduce within-cluster distances."
        )

        st.latex(
            r"J = \sum_{i=1}^{n}"
            r"\left\|x_i-\mu_{c_i}\right\|^2"
        )

        st.caption(
            "The objective minimizes the sum of squared distances "
            "between each product and its assigned cluster center."
        )

    # Always show clustering comparison, even before product search.
    st.divider()
    st.subheader("Clustering evaluation")

    if not comparison.empty:
        st.dataframe(
            comparison.round(4),
            use_container_width=True
        )

        st.caption(
            "Metrics are calculated on a representative sample. "
            "Higher Silhouette and Calinski–Harabasz scores and lower "
            "Davies–Bouldin scores are generally preferable, but they "
            "should be interpreted together."
        )

        fig, ax = plt.subplots()
        ax.bar(
            comparison["Algorithm"],
            comparison["Silhouette"]
        )
        ax.set_ylabel("Silhouette score")
        ax.set_title("Clustering configuration comparison")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Unable to calculate clustering comparison metrics.")


# =========================================================
# APP NAVIGATION
# =========================================================
with st.sidebar:
    st.title("📊 ML Capstone Studio")
    st.caption("Review 2 · Interactive ML Dashboard")

    page = st.radio(
        "Explore",
        [
            "Overview",
            "Regression",
            "Classification",
            "Amazon Product Analysis"
        ]
    )

if page == "Overview":
    hero(
        "Machine Learning Capstone Studio",
        "Explore predictions, understand calculations, and visualize data."
    )

    a, b, c = st.columns(3)
    a.metric("ML Tasks", "3")
    b.metric("Supervised Learning", "2")
    c.metric("Unsupervised Learning", "1")

    st.markdown("### Explore your project")

    st.write(
        "**🌾 Crop Yield Prediction** — Estimate crop yield "
        "from the selected input features."
    )
    st.write(
        "**🚘 Vehicle Classification** — Predict a vehicle category "
        "and inspect classification performance."
    )
    st.write(
        "**🛍️ Amazon Product Analysis** — Find products, inspect "
        "their segments, and explore price, rating, and discount patterns."
    )

    st.info(
        "Choose a section from the sidebar to begin."
    )

elif page == "Regression":
    try:
        regression_page()
    except Exception as e:
        st.error("Unable to load regression.")
        st.exception(e)

elif page == "Classification":
    try:
        classification_page()
    except Exception as e:
        st.error("Unable to load classification.")
        st.exception(e)

else:
    try:
        clustering_page()
    except Exception as e:
        st.error("Unable to load Amazon product analysis.")
        st.exception(e)
