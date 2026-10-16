
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# =========================================================
# CONFIGURATION AND UI
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
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.2);
}
.hero {
    padding: 24px;
    border-radius: 16px;
    background: linear-gradient(115deg,#14213d,#254f75);
    color: white;
    margin-bottom: 20px;
}
.note {
    padding: 12px;
    border-left: 4px solid #3b82f6;
    background: var(--secondary-background-color);
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# COMMON HELPERS
# =========================================================

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        normalized = "".join(
            c.lower() for c in filename if c.isalnum()
        )
        candidates = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []

        matches = [
            p for p in candidates
            if "".join(
                c.lower() for c in p.name if c.isalnum()
            ) == normalized
        ]

        if not matches:
            raise FileNotFoundError(
                f"Cannot find {filename} in {DATA_DIR}. "
                f"Available CSVs: {[p.name for p in candidates]}"
            )

        path = matches[0]

    if path.stat().st_size == 0:
        raise ValueError(f"{path.name} is empty.")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if df.empty:
        raise ValueError(f"{path.name} has no data rows.")

    return df


def numeric_clean(series):
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
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor(X, scale=True):
    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

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
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_encoder())
            ]),
            categorical_cols
        ))

    return ColumnTransformer(transformers, remainder="drop")


def note(message):
    st.markdown(
        f'<div class="note">{message}</div>',
        unsafe_allow_html=True
    )


def input_form(X, prefix):
    values = {}
    left, right = st.columns(2)

    for i, col in enumerate(X.columns):
        with (left if i % 2 == 0 else right):
            if pd.api.types.is_numeric_dtype(X[col]):
                nums = pd.to_numeric(
                    X[col], errors="coerce"
                ).dropna()
                default = float(nums.median()) if not nums.empty else 0.0

                values[col] = st.number_input(
                    col.replace("_", " "),
                    value=default,
                    key=f"{prefix}_{col}"
                )
            else:
                choices = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )
                values[col] = (
                    st.selectbox(
                        col.replace("_", " "),
                        choices,
                        key=f"{prefix}_{col}"
                    )
                    if choices else ""
                )

    return pd.DataFrame([values], columns=X.columns)


# =========================================================
# REGRESSION DATA AND MODELS
# =========================================================

@st.cache_data
def regression_data():
    candidates = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    names = {p.name.lower().replace("_", " "): p.name for p in candidates}

    filename = next(
        (
            names[key] for key in [
                "crop yeild data.csv",
                "crop yield data.csv"
            ] if key in names
        ),
        "Crop Yeild Data.csv"
    )

    df = load_csv(filename)

    if "Yield" not in df.columns:
        raise ValueError(
            f"Crop CSV must contain 'Yield'. Columns: {df.columns.tolist()}"
        )

    df["Yield"] = numeric_clean(df["Yield"])
    df = df.dropna(subset=["Yield"]).drop_duplicates().copy()

    # Avoid production leakage when predicting yield.
    X = df.drop(
        columns=[
            c for c in ["Yield", "Production"]
            if c in df.columns
        ]
    ).copy()

    if {"Fertilizer", "Area"}.issubset(X.columns):
        X["Fertilizer_per_Area"] = (
            X["Fertilizer"] / X["Area"].replace(0, np.nan)
        )

    if {"Pesticide", "Area"}.issubset(X.columns):
        X["Pesticide_per_Area"] = (
            X["Pesticide"] / X["Area"].replace(0, np.nan)
        )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    return X, df["Yield"].astype(float)


@st.cache_resource
def fit_regression():
    X, y = regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "ElasticNet Regression": ElasticNet(
            alpha=0.01, max_iter=10000
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=18, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Support Vector Regressor": SVR()
    }

    fitted, predictions, rows = {}, {}, []

    for name, estimator in estimators.items():
        pipe = Pipeline([
            ("preprocess", make_preprocessor(X)),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        fitted[name] = pipe
        predictions[name] = pred

        rows.append({
            "Algorithm": name,
            "R2": r2_score(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred)
        })

    # Polynomial regression where the feature count is manageable.
    try:
        prep = make_preprocessor(X)
        transformed = prep.fit_transform(X_train)

        if transformed.shape[1] <= 25:
            poly = Pipeline([
                ("preprocess", make_preprocessor(X)),
                ("polynomial", PolynomialFeatures(
                    degree=2, include_bias=False
                )),
                ("model", LinearRegression())
            ])

            poly.fit(X_train, y_train)
            pred = poly.predict(X_test)
            name = "Polynomial Regression"

            fitted[name] = poly
            predictions[name] = pred

            rows.append({
                "Algorithm": name,
                "R2": r2_score(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "MAE": mean_absolute_error(y_test, pred)
            })
    except Exception:
        pass

    results = pd.DataFrame(rows).sort_values(
        "R2", ascending=False
    )

    return X, y, X_test, y_test, fitted, predictions, results


def regression_page():
    st.markdown(
        '<div class="hero"><h1>🌾 Crop Yield Regression</h1>'
        '<p>Predict yield and understand model performance.</p></div>',
        unsafe_allow_html=True
    )

    try:
        X, y, X_test, y_test, models, preds, results = fit_regression()

        selected = st.selectbox(
            "Choose regression algorithm",
            results["Algorithm"].tolist()
        )

        row = results.set_index("Algorithm").loc[selected]

        st.caption(
            f"Algorithm: {selected} | Target: Yield | "
            "80/20 train-test split"
        )

        a, b, c = st.columns(3)
        a.metric("R² score", f"{row['R2']:.4f}")
        b.metric("RMSE", f"{row['RMSE']:.4f}")
        c.metric("MAE", f"{row['MAE']:.4f}")

        st.subheader("Enter crop conditions")
        sample = input_form(X, "reg")

        if st.button("Predict crop yield", type="primary"):
            st.session_state["crop_prediction"] = float(
                models[selected].predict(sample)[0]
            )
            st.session_state["crop_model"] = selected

        if "crop_prediction" in st.session_state:
            st.success(
                f"Estimated yield: "
                f"{st.session_state['crop_prediction']:,.4f}"
            )
            st.caption(
                f"Model: {st.session_state['crop_model']}. "
                "This is an estimate, not a guarantee."
            )

        tabs = st.tabs([
            "Actual vs predicted", "Residuals",
            "Model comparison", "Explanation"
        ])

        pred = preds[selected]

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_test, pred, alpha=0.6)

            low = min(float(y_test.min()), float(np.min(pred)))
            high = max(float(y_test.max()), float(np.max(pred)))

            ax.plot(
                [low, high], [low, high],
                linestyle="--", label="Perfect prediction"
            )
            ax.set_xlabel("Actual yield")
            ax.set_ylabel("Predicted yield")
            ax.set_title(f"Actual vs predicted: {selected}")
            ax.legend()
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)
            note(
                "Dots close to the dashed line indicate predictions "
                "close to actual yield."
            )

        with tabs[1]:
            residuals = np.asarray(y_test) - np.asarray(pred)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(pred, residuals, alpha=0.6)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted yield")
            ax.set_ylabel("Residual: actual − predicted")
            ax.set_title("Residual analysis")
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)
            note(
                "Positive residuals mean underprediction. Negative "
                "residuals mean overprediction. Random scatter around "
                "zero is generally desirable."
            )

        with tabs[2]:
            st.dataframe(results.round(4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 4))
            ordered = results.sort_values("R2")
            ax.barh(ordered["Algorithm"], ordered["R2"])
            ax.set_xlabel("R²")
            ax.set_title("Regression algorithm comparison")
            ax.grid(axis="x", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)
            note(
                "Higher R² is generally better. Lower RMSE and MAE "
                "indicate smaller prediction errors."
            )

        with tabs[3]:
            st.markdown(f"""
            **Algorithm:** {selected}

            - **R²:** How well predictions explain target variation.
            - **RMSE:** Penalizes larger errors more strongly.
            - **MAE:** Average absolute prediction error.
            - **Test set:** Data not used to fit the model.
            """)

    except Exception as e:
        st.error("Regression failed to load.")
        st.code(str(e))


# =========================================================
# CLASSIFICATION DATA AND MODELS
# =========================================================

@st.cache_data
def classification_data():
    df = load_csv("cars.csv").drop_duplicates().copy()

    if "Vehicle Size" not in df.columns:
        raise ValueError(
            f"Expected 'Vehicle Size'. Columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=["Vehicle Size"])
    y = df["Vehicle Size"].astype(str)

    X = df.drop(
        columns=[
            c for c in ["Vehicle Size", "Model"]
            if c in df.columns
        ]
    ).copy()

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]

    return X, y


@st.cache_resource
def fit_classification():
    X, y = classification_data()

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=stratify
    )

    estimators = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, random_state=42
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, random_state=42, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "Bagging": BaggingClassifier(
            n_estimators=50, random_state=42, n_jobs=-1
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=400,
            random_state=42, early_stopping=True
        )
    }

    fitted, predictions, rows = {}, {}, []

    for name, estimator in estimators.items():
        try:
            pipe = Pipeline([
                ("preprocess", make_preprocessor(X)),
                ("model", estimator)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            fitted[name] = pipe
            predictions[name] = pred

            rows.append({
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(
                    y_test, pred, average="weighted",
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test, pred, average="weighted",
                    zero_division=0
                ),
                "F1": f1_score(
                    y_test, pred, average="weighted",
                    zero_division=0
                )
            })
        except Exception:
            continue

    if not fitted:
        raise ValueError("No classification models could be trained.")

    results = pd.DataFrame(rows).sort_values(
        "F1", ascending=False
    )

    return X, y, X_test, y_test, fitted, predictions, results


def classification_page():
    st.markdown(
        '<div class="hero"><h1>🚘 Vehicle Classification</h1>'
        '<p>Predict vehicle size and understand classification errors.</p></div>',
        unsafe_allow_html=True
    )

    try:
        X, y, X_test, y_test, models, preds, results = (
            fit_classification()
        )

        selected = st.selectbox(
            "Choose classification algorithm",
            results["Algorithm"].tolist()
        )

        row = results.set_index("Algorithm").loc[selected]

        st.caption(
            f"Algorithm: {selected} | Target: Vehicle Size | "
            "80/20 train-test split"
        )

        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{row['Accuracy']:.3f}")
        b.metric("Precision", f"{row['Precision']:.3f}")
        c.metric("Recall", f"{row['Recall']:.3f}")
        d.metric("F1 score", f"{row['F1']:.3f}")

        st.subheader("Enter vehicle details")
        sample = input_form(X, "car")

        if st.button("Predict vehicle size", type="primary"):
            st.session_state["vehicle_prediction"] = (
                models[selected].predict(sample)[0]
            )
            st.session_state["vehicle_model"] = selected

        if "vehicle_prediction" in st.session_state:
            st.success(
                f"Predicted vehicle size: "
                f"{st.session_state['vehicle_prediction']}"
            )
            st.caption(
                f"Model: {st.session_state['vehicle_model']}"
            )

        tabs = st.tabs([
            "Confusion matrix", "Class metrics",
            "Probabilities", "Model comparison", "Explanation"
        ])

        pred = preds[selected]
        classes = sorted(y_test.unique())

        with tabs[0]:
            matrix = confusion_matrix(
                y_test, pred, labels=classes
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            image = ax.imshow(matrix, cmap="Blues")

            ax.set_xticks(
                range(len(classes)), labels=classes, rotation=25
            )
            ax.set_yticks(range(len(classes)), labels=classes)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("Actual class")
            ax.set_title(f"Confusion matrix: {selected}")

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j, i, matrix[i, j],
                        ha="center", va="center"
                    )

            fig.colorbar(image, ax=ax)
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)
            note(
                "The diagonal contains correct predictions. "
                "Off-diagonal cells show misclassified examples."
            )

        with tabs[1]:
            report = classification_report(
                y_test, pred, output_dict=True,
                zero_division=0
            )
            st.dataframe(
                pd.DataFrame(report).T.round(3),
                use_container_width=True
            )
            note(
                "Precision measures the reliability of positive "
                "class predictions. Recall measures how many actual "
                "examples were identified. F1 balances both."
            )

        with tabs[2]:
            try:
                probabilities = models[selected].predict_proba(sample)[0]
                model_classes = models[selected].named_steps[
                    "model"
                ].classes_

                probability_df = pd.DataFrame({
                    "Class": model_classes,
                    "Probability": probabilities
                }).set_index("Class")

                st.bar_chart(probability_df)
                note(
                    "Probabilities are model estimates for this input, "
                    "not guarantees of correctness."
                )
            except Exception:
                st.info("Probabilities are unavailable for this model.")

        with tabs[3]:
            st.dataframe(results.round(4), use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 4))
            ordered = results.sort_values("F1")
            ax.barh(ordered["Algorithm"], ordered["F1"])
            ax.set_xlabel("Weighted F1")
            ax.set_title("Classification algorithm comparison")
            ax.grid(axis="x", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)
            note(
                "Compare accuracy and weighted F1 together, "
                "especially if classes have different frequencies."
            )

        with tabs[4]:
            st.markdown(f"""
            **Algorithm:** {selected}

            - **Input:** Vehicle attributes.
            - **Output:** A vehicle-size class.
            - **Accuracy:** Fraction of test examples predicted correctly.
            - **Precision:** Reliability of class predictions.
            - **Recall:** Fraction of actual examples found.
            - **F1:** Balance between precision and recall.
            """)

    except Exception as e:
        st.error("Classification failed to load.")
        st.code(str(e))


# =========================================================
# AMAZON CLUSTERING
# =========================================================

@st.cache_data
def amazon_data():
    df = load_csv("amazon.csv").copy()

    required = [
        "product_name", "category", "actual_price",
        "discounted_price", "discount_percentage",
        "rating", "rating_count"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Amazon CSV missing columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    for col in [
        "actual_price", "discounted_price",
        "discount_percentage", "rating", "rating_count"
    ]:
        df[col] = numeric_clean(df[col])

    df["main_category"] = (
        df["category"].astype(str).str.split("|").str[0]
    )

    feature_columns = [
        "actual_price", "discounted_price",
        "discount_percentage", "rating", "rating_count"
    ]

    X = df[feature_columns].copy()
    X["actual_price"] = np.log1p(
        X["actual_price"].clip(lower=0)
    )
    X["discounted_price"] = np.log1p(
        X["discounted_price"].clip(lower=0)
    )
    X["rating_count"] = np.log1p(
        X["rating_count"].clip(lower=0)
    )

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    return df, X


@st.cache_resource
def fit_clustering():
    df, X = amazon_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X_scaled)

    models = {}
    rows = []

    for k in [3, 4, 5]:
        model = KMeans(
            n_clusters=k, random_state=42, n_init=10
        )
        labels = model.fit_predict(X_scaled)
        name = f"K-Means (k={k})"

        models[name] = {
            "labels": labels,
            "model": model,
            "X": X_scaled,
            "full_data": True
        }

        rows.append({
            "Algorithm": name,
            "Clusters": k,
            "Silhouette": silhouette_score(X_scaled, labels),
            "Davies-Bouldin": davies_bouldin_score(
                X_scaled, labels
            ),
            "Calinski-Harabasz": calinski_harabasz_score(
                X_scaled, labels
            )
        })

    # Hierarchical clustering on a sample to avoid excessive memory.
    rng = np.random.RandomState(42)
    sample_indices = np.arange(len(df))

    if len(df) > 1500:
        sample_indices = np.sort(
            rng.choice(len(df), 1500, replace=False)
        )

    X_sample = X_scaled[sample_indices]

    for method in ["ward", "complete", "average"]:
        model = AgglomerativeClustering(
            n_clusters=3, linkage=method
        )
        labels = model.fit_predict(X_sample)
        name = f"Agglomerative ({method}, k=3)"

        models[name] = {
            "labels": labels,
            "model": model,
            "X": X_sample,
            "full_data": False,
            "sample_indices": sample_indices
        }

        rows.append({
            "Algorithm": name,
            "Clusters": len(np.unique(labels)),
            "Silhouette": silhouette_score(X_sample, labels),
            "Davies-Bouldin": davies_bouldin_score(
                X_sample, labels
            ),
            "Calinski-Harabasz": calinski_harabasz_score(
                X_sample, labels
            )
        })

    return (
        df, X, X_scaled, coordinates, pca,
        models, pd.DataFrame(rows), sample_indices
    )


def clustering_page():
    st.markdown(
        '<div class="hero"><h1>🛍️ Amazon Product Explorer</h1>'
        '<p>Discover product groups using clustering algorithms.</p></div>',
        unsafe_allow_html=True
    )

    try:
        (
            df, X, X_scaled, coordinates, pca,
            models, metrics, sample_indices
        ) = fit_clustering()

        selected = st.selectbox(
            "Choose clustering algorithm",
            list(models.keys())
        )

        info = models[selected]
        labels = info["labels"]

        metric = metrics.set_index("Algorithm").loc[selected]

        st.caption(
            f"Algorithm: {selected} | Features: price, discount, "
            "rating and review count | StandardScaler preprocessing"
        )

        a, b, c = st.columns(3)
        a.metric("Products clustered", f"{len(labels):,}")
        b.metric("Silhouette", f"{metric['Silhouette']:.3f}")
        c.metric(
            "Davies–Bouldin",
            f"{metric['Davies-Bouldin']:.3f}"
        )

        tabs = st.tabs([
            "Search product", "PCA cluster map",
            "Cluster profiles", "Compare algorithms",
            "Elbow & dendrogram"
        ])

        # Product search and assignment
        with tabs[0]:
            query = st.text_input(
                "Search Amazon product name",
                placeholder="Type a product name..."
            )

            matches = df[
                df["product_name"].astype(str).str.contains(
                    query, case=False, na=False
                )
            ] if query.strip() else df

            if matches.empty:
                st.warning("No products matched that search.")
            else:
                names = matches["product_name"].drop_duplicates().tolist()
                product = st.selectbox("Select product", names)

                row = df[df["product_name"] == product].iloc[0]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Selling price",
                    f"{row['discounted_price']:,.0f}"
                    if pd.notna(row["discounted_price"]) else "N/A"
                )
                c2.metric(
                    "Actual price",
                    f"{row['actual_price']:,.0f}"
                    if pd.notna(row["actual_price"]) else "N/A"
                )
                c3.metric(
                    "Discount",
                    f"{row['discount_percentage']:.1f}%"
                    if pd.notna(row["discount_percentage"]) else "N/A"
                )
                c4.metric(
                    "Rating",
                    f"{row['rating']:.1f}/5"
                    if pd.notna(row["rating"]) else "N/A"
                )

                if info["full_data"]:
                    cluster_id = int(labels[row.name])

                    st.success(
                        f"**{product}** belongs to Cluster {cluster_id} "
                        f"using {selected}."
                    )

                    similar = df.loc[
                        np.where(labels == cluster_id)[0]
                    ].copy()

                else:
                    lookup = np.where(
                        info["sample_indices"] == row.name
                    )[0]

                    if len(lookup) == 0:
                        st.info(
                            "This product is outside the hierarchical "
                            "clustering sample. Select K-Means for "
                            "a full-catalog cluster assignment."
                        )
                        similar = pd.DataFrame()
                    else:
                        cluster_id = int(labels[lookup[0]])

                        st.success(
                            f"**{product}** belongs to Cluster "
                            f"{cluster_id} using {selected}."
                        )

                        similar = df.iloc[
                            info["sample_indices"][
                                np.where(labels == cluster_id)[0]
                            ]
                        ].copy()

                if not similar.empty:
                    similar = similar[
                        similar["product_name"] != product
                    ]

                    st.subheader("Other products in this cluster")

                    st.dataframe(
                        similar[[
                            "product_name", "main_category",
                            "actual_price", "discounted_price",
                            "discount_percentage", "rating",
                            "rating_count"
                        ]].head(15),
                        use_container_width=True
                    )

                    note(
                        "Products in the same cluster have similar "
                        "values for the selected features. This does "
                        "not establish that they are direct substitutes "
                        "or have the same sales trend."
                    )

        # PCA map
        with tabs[1]:
            if info["full_data"]:
                plot_coords = coordinates
            else:
                plot_coords = PCA(
                    n_components=2, random_state=42
                ).fit_transform(info["X"])

            fig, ax = plt.subplots(figsize=(8, 5))

            points = ax.scatter(
                plot_coords[:, 0], plot_coords[:, 1],
                c=labels, cmap="tab10",
                s=18, alpha=0.65
            )

            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"PCA cluster map: {selected}")
            fig.colorbar(points, ax=ax, label="Cluster")
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

            if info["full_data"]:
                st.write(
                    f"The first two PCA components explain "
                    f"{pca.explained_variance_ratio_.sum() * 100:.1f}% "
                    "of standardized feature variance."
                )

            note(
                "Each dot is a product. Dots close together have "
                "similar feature profiles in this two-dimensional "
                "projection. PCA displays clusters; it does not "
                "create the cluster assignments."
            )

        # Cluster profiles
        with tabs[2]:
            if info["full_data"]:
                prof = df.copy()
            else:
                prof = df.iloc[info["sample_indices"]].copy()

            prof["Cluster"] = labels

            profile = prof.groupby("Cluster")[[
                "actual_price", "discounted_price",
                "discount_percentage", "rating", "rating_count"
            ]].median()

            profile["Product count"] = (
                prof.groupby("Cluster").size()
            )

            st.dataframe(profile.round(2), use_container_width=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            counts = prof["Cluster"].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of products")
            ax.set_title("Cluster sizes")

            st.pyplot(fig)
            plt.close(fig)

            note(
                "Compare the median price, discount, rating and "
                "review count to understand each cluster. Cluster "
                "numbers are arbitrary and may change between algorithms."
            )

        # Algorithm comparison
        with tabs[3]:
            st.dataframe(metrics.round(4), use_container_width=True)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            for ax, column, title, ascending in [
                (axes[0], "Silhouette", "Silhouette ↑", True),
                (axes[1], "Davies-Bouldin", "Davies–Bouldin ↓", False),
                (axes[2], "Calinski-Harabasz", "Calinski–Harabasz ↑", True)
            ]:
                ordered = metrics.sort_values(
                    column, ascending=ascending
                )
                ax.barh(
                    ordered["Algorithm"], ordered[column]
                )
                ax.set_title(title, fontsize=10)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="x", alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            note(
                "Silhouette and Calinski–Harabasz are generally better "
                "when higher. Davies–Bouldin is generally better when "
                "lower. Compare metrics with caution across different "
                "samples and cluster counts."
            )

        # Elbow curve and dendrogram
        with tabs[4]:
            st.subheader("Elbow method")

            k_values = list(range(2, 9))
            inertias = []

            for k in k_values:
                km = KMeans(
                    n_clusters=k, random_state=42, n_init=10
                )
                km.fit(X_scaled)
                inertias.append(km.inertia_)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(k_values, inertias, marker="o")
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("K-Means elbow curve")
            ax.grid(alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

            note(
                "Inertia measures the within-cluster squared distance "
                "to centroids. Look for a bend where increasing k "
                "provides less improvement."
            )

            st.subheader("Hierarchical dendrogram")

            tree_data = X_scaled[
                sample_indices[:min(400, len(sample_indices))]
            ]

            tree = linkage(tree_data, method="ward")

            fig, ax = plt.subplots(figsize=(8, 4))
            dendrogram(
                tree, truncate_mode="lastp",
                p=25, ax=ax
            )
            ax.set_xlabel("Merged sample groups")
            ax.set_ylabel("Ward distance")
            ax.set_title("Hierarchical clustering dendrogram")

            st.pyplot(fig)
            plt.close(fig)

            note(
                "The dendrogram shows which groups merge and at what "
                "distance. Higher merge heights generally indicate "
                "more dissimilar groups."
            )

        st.info(
            "Your Amazon dataset contains product attributes rather "
            "than dated sales records. This app identifies product "
            "segments; it does not forecast future sales trends."
        )

    except Exception as e:
        st.error("Clustering failed to load.")
        st.code(str(e))


# =========================================================
# NAVIGATION AND HOME PAGE
# =========================================================

with st.sidebar:
    st.title("📊 ML Studio")
    st.caption("Review 2 · Interactive ML project")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Regression",
            "Classification",
            "Amazon Clustering"
        ]
    )

    st.divider()
    st.caption("Data-driven predictions and visualizations")


if page == "Overview":
    st.markdown(
        '<div class="hero"><h1>Machine Learning Capstone Studio</h1>'
        '<p>Explore three ML approaches through interactive predictions '
        'and clear visualizations.</p></div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)
    a.metric("Learning tracks", 3)
    b.metric("Supervised tasks", 2)
    c.metric("Unsupervised tasks", 1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌾 Regression")
        st.write(
            "Predict crop yield and analyze errors "
            "with regression metrics and plots."
        )

    with col2:
        st.subheader("🚘 Classification")
        st.write(
            "Predict vehicle size and understand "
            "classification performance."
        )

    with col3:
        st.subheader("🛍️ Clustering")
        st.write(
            "Search Amazon products, inspect clusters "
            "and compare product profiles."
        )

    st.info("Choose a track from the sidebar to begin.")

elif page == "Regression":
    regression_page()

elif page == "Classification":
    classification_page()

else:
    clustering_page()
