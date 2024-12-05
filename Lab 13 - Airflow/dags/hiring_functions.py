import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score


# 1. Crear carpetas
def create_folders(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    folder_name = f"data_{execution_date}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'models'), exist_ok=True)
    print(f"Carpetas creadas en {folder_name}")


# 2. División de datos
def split_data(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    raw_path = os.path.join(f'data_{execution_date}', 'raw', 'data_1.csv')
    splits_path = os.path.join(f'data_{execution_date}', 'splits')

    df = pd.read_csv(raw_path)
    X = df.drop(columns=['HiringDecision'])
    y = df['HiringDecision']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=99
    )

    X_train.to_csv(os.path.join(splits_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(splits_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(splits_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(splits_path, 'y_test.csv'), index=False)
    print("Datos divididos y guardados en la carpeta splits")


# 3. Preprocesamiento y entrenamiento
def preprocess_and_train(**kwargs):
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    splits_path = os.path.join(f'data_{execution_date}', 'splits')
    models_path = os.path.join(f'data_{execution_date}', 'models')

    X_train = pd.read_csv(os.path.join(splits_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(splits_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(splits_path, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(splits_path, 'y_test.csv')).squeeze()

    numerical_features = ['Age', 'ExperienceYears', 'PreviousCompanies',
                           'DistanceFromCompany', 'InterviewScore',
                           'SkillScore', 'PersonalityScore']
    categorical_features = ['Gender', 'EducationLevel', 'RecruitmentStrategy']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=99))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Score: {f1:.2f}")

    joblib.dump(clf, os.path.join(models_path, 'hiring_model.joblib'))
    print("Modelo entrenado y guardado")


# 4. Predicción
def predict(file, model_path):
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La predicción es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface(**kwargs):
    import gradio as gr
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    model_path = os.path.join(f'data_{execution_date}', 'models', 'hiring_model.joblib')

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Nico será contratado o no."
    )
    interface.launch(share=True)