import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Student Course Completion Prediction
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score, f1_score

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    import numpy as np

    from catboost import CatBoostClassifier, Pool
    return (
        CatBoostClassifier,
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        Pipeline,
        Pool,
        RandomForestClassifier,
        StandardScaler,
        f1_score,
        np,
        pd,
        precision_score,
        recall_score,
        train_test_split,
    )


@app.cell
def _():
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset (clean)
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('Course_Completion_Prediction.csv')
    return (df,)


@app.cell
def _(df):
    for col in ["Student_ID", "Name", "Enrollment_Date", "Course_ID"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    TARGET_COL = "Completed"   

    df.head()
    return (TARGET_COL,)


@app.cell
def _(df):
    print("Колонки:", df.columns.tolist())
    return


@app.cell
def _(TARGET_COL, df):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    if y.dtype == "O":
        y = y.astype("category").cat.codes 

    print("Распределение классов (0/1):")
    print(y.value_counts(normalize=True))
    return X, y


@app.cell
def _(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print("Категориальные признаки:", categorical_cols)
    print()
    print("Числовые признаки:", numeric_cols)
    return categorical_cols, numeric_cols


@app.cell
def _(mo):
    test_size_slide = mo.ui.slider(start = 10 , stop = 60, step = 5, value = 30, label='Test size(%)')
    return (test_size_slide,)


@app.cell
def _(test_size_slide):
    test_size_slide_ber = test_size_slide.value / 100
    return (test_size_slide_ber,)


@app.cell
def _(X, test_size_slide_ber, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size_slide_ber,
        stratify=y,
        random_state=42
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell
def _(f1_score, precision_score, recall_score):
    def print_metrics(model_name, y_true, y_pred):
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"{model_name}:")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1-score:  {f1:.3f}")
        print("-" * 40)
    return (print_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo Slider Show
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Модель 1 — Logistic Regression
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    LogisticRegression,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    X_test,
    X_train,
    categorical_cols,
    numeric_cols,
    print_metrics,
    y_test,
    y_train,
):
    _numeric_transformer = StandardScaler()
    _categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocess_lr = ColumnTransformer(transformers=[('num', _numeric_transformer, numeric_cols), ('cat', _categorical_transformer, categorical_cols)])
    log_reg_model = Pipeline(steps=[('preprocess', preprocess_lr), ('model', LogisticRegression(max_iter=1000, n_jobs=-1))])
    log_reg_model.fit(X_train, y_train)
    y_pred_lr = log_reg_model.predict(X_test)
    print_metrics('Logistic Regression', y_test, y_pred_lr)
    return (y_pred_lr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Модель 2 — CatBoostClassifier
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    Pool,
    X_test,
    X_train,
    categorical_cols,
    print_metrics,
    y_test,
    y_train,
):
    cat_features = categorical_cols

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

    cat_model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        iterations=500,
        loss_function="Logloss",
        eval_metric="F1",
        verbose=100,
        random_state=42
    )

    cat_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    y_pred_cat = cat_model.predict(test_pool)
    y_pred_cat = y_pred_cat.astype(int) 

    print_metrics("CatBoost", y_test, y_pred_cat)
    return (y_pred_cat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Модель 3 — RandomForestClassifier
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    RandomForestClassifier,
    X_test,
    X_train,
    categorical_cols,
    numeric_cols,
    print_metrics,
    y_test,
    y_train,
):
    preprocess_rf = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocess_rf),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"  
            )),
        ]
    )

    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    print_metrics("Random Forest", y_test, y_pred_rf)
    return (y_pred_rf,)


@app.cell
def _(
    f1_score,
    precision_score,
    recall_score,
    y_pred_cat,
    y_pred_lr,
    y_pred_rf,
    y_test,
):
    # Logistic Regression
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr    = recall_score(y_test, y_pred_lr)
    f1_lr        = f1_score(y_test, y_pred_lr)

    # CatBoost
    precision_cat = precision_score(y_test, y_pred_cat)
    recall_cat    = recall_score(y_test, y_pred_cat)
    f1_cat        = f1_score(y_test, y_pred_cat)

    # Random Forest
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf    = recall_score(y_test, y_pred_rf)
    f1_rf        = f1_score(y_test, y_pred_rf)
    return (
        f1_cat,
        f1_lr,
        f1_rf,
        precision_cat,
        precision_lr,
        precision_rf,
        recall_cat,
        recall_lr,
        recall_rf,
    )


@app.cell
def _(
    f1_cat,
    f1_lr,
    f1_rf,
    pd,
    precision_cat,
    precision_lr,
    precision_rf,
    recall_cat,
    recall_lr,
    recall_rf,
):
    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "CatBoost", "Random Forest"],
        "Precision": [precision_lr, precision_cat, precision_rf],
        "Recall":    [recall_lr, recall_cat, recall_rf],
        "F1":        [f1_lr, f1_cat, f1_rf]
    })

    metrics_df
    return (metrics_df,)


@app.cell
def _(metrics_df, np, plt):
    models = metrics_df["Model"]
    precision = metrics_df["Precision"]
    recall = metrics_df["Recall"]
    f1 = metrics_df["F1"]

    x = np.arange(len(models))
    width = 0.25

    plt.figure()

    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Score")
    plt.title("Сравнение моделей по метрикам")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(test_size_slide):
    test_size_slide
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Кластеризация
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k-means
    """)
    return


@app.cell
def _():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    return KMeans, PCA, plt, silhouette_score


@app.cell
def _(ColumnTransformer, OneHotEncoder, StandardScaler, df):
    TARGET_COL_1 = "Completed"

    X_clust = df.drop(columns=[TARGET_COL_1])

    categorical_cols_1 = X_clust.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols_1     = X_clust.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print("Категориальные признаки:", categorical_cols_1)
    print("Числовые признаки:", numeric_cols_1)

    _numeric_transformer = StandardScaler()
    _categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,  
    )

    preprocess_clust = ColumnTransformer(
        transformers=[
            ("num", _numeric_transformer, numeric_cols_1),
            ("cat", _categorical_transformer, categorical_cols_1),
        ]
    )

    X_prepared_1 = preprocess_clust.fit_transform(X_clust)
    X_prepared_1.shape
    return X_prepared_1, numeric_cols_1


@app.cell
def _(mo):
    k_slider = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=3,
        label="Число кластеров K",
        show_value=True,
    )

    k_slider  
    return (k_slider,)


@app.cell
def _(KMeans, PCA, X_prepared_1, df, k_slider, mo, numeric_cols_1, plt):
    n_clusters = int(k_slider.value)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_labels = kmeans.fit_predict(X_prepared_1)

    df["cluster"] = cluster_labels

    print("Распределение по кластерам:")
    print(df["cluster"].value_counts(), "\n")

    cluster_summary = df.groupby("cluster")[numeric_cols_1].mean().round(2)
    cluster_summary

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_prepared_1)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=df["cluster"],
        alpha=0.6,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Кластеры студентов (PCA 2D), K = {n_clusters}")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Кластер")

    plt.tight_layout()

    mo.mpl.interactive(fig)
    return cluster_labels, cluster_summary


@app.cell
def _(X_prepared_1, cluster_labels, mo, silhouette_score):
    sil_score = silhouette_score(X_prepared_1, cluster_labels)

    mo.md(f"""
    ### 📏 Silhouette Score
    **Silhouette Score = {sil_score:.3f}**

    Интерпретация:
    - 0.5–1.0 → хорошие, чёткие кластеры  
    - 0.2–0.5 → допустимые, средние  
    - ниже 0.2 → слабая сегментация  
    """)
    return


@app.cell
def _(cluster_summary, df, mo):
    descriptions = []

    selected_features = [
        "Login_Frequency",
        "Average_Session_Duration_Min",
        "Quiz_Score_Avg",
        "Assignments_Submitted",
        "Progress_Percentage",
        "Satisfaction_Rating"
    ]

    for c in sorted(df["cluster"].unique()):
        cluster_data = cluster_summary.loc[c]

        desc = f"### Кластер {c}\n"

        desc += "#### 📊 Ключевые показатели\n"
        for col1 in selected_features:
            if col1 in cluster_summary.columns:
                desc += f"- **{col1}**: {cluster_data[col1]:.2f}\n"

        if ("Progress_Percentage" in cluster_summary.columns) and \
           ("Quiz_Score_Avg" in cluster_summary.columns) and \
           ("Project_Grade" in cluster_summary.columns):

            avg_score = cluster_data["Quiz_Score_Avg"]
            submitted = cluster_data["Assignments_Submitted"]
            progress = cluster_data["Progress_Percentage"]
        
            # высокоуспевающие
            if avg_score >= 75 and submitted >= 6 and progress >= 60:
                desc += "→ 🟢 **Высокоуспевающие студенты**: отличные оценки, активность и прогресс.\n"
        
            # группа риска
            elif progress < 45 or avg_score < 65 or submitted < 4:
                desc += "→ 🔴 **Группа риска**: низкий прогресс, мало выполненных заданий или слабые оценки.\n"
        
            # средний сегмент
            else:
                desc += "→ 🟡 **Средний сегмент**: учатся стабильно, есть потенциал роста.\n"


        descriptions.append(desc)

    mo.md("\n\n".join(descriptions))

    return


@app.cell
def _(mo):
    recommendations = """
    ## 🎯 Рекомендации по сегментам

    - **Кластеры с низким прогрессом**
      - отправить автоматические напоминания,
      - предложить наставника,
      - короткий обзорный материал для возврата в курс.

    - **Кластеры со средним прогрессом**
      - персональные рекомендации по темам,
      - поощрения за регулярность.

    - **Высокоуспевающие**
      - приглашение в продвинутые модули,
      - участие в соревнованиях / бонусах,
      - использование их паттернов для обучения других.

    """

    mo.md(recommendations)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
