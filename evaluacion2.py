import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression

# Crear una conexión usando SQLAlchemy
engine = create_engine("postgresql+psycopg2://randomn3_owner:T78ZeKnFpBsI@ep-late-violet-a56fkacj.us-east-2.aws.neon.tech:5432/kumaretail")

# Leer la consulta y cargarla en un DataFrame
query = "SELECT * FROM student_performance;"
df = pd.read_sql_query(query, engine)

# Seleccionar las variables predictoras (X) y la variable objetivo (y)
X = df[['study_hours', 'class_attendance', 'previous_gpa', 'social_activities', 'library_visits', 'is_scholarship_holder']]
y = df['current_gpa']

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Obtener el intercepto y los coeficientes (pendientes)
intercept = model.intercept_
coefficients = model.coef_

# Mostrar los resultados
print("Intercepto (β0):", intercept)
for i, col_name in enumerate(X.columns):
    print(f"Coeficiente para {col_name} (β{i+1}):", coefficients[i])


# Definir los coeficientes
intercept = 3.8208
coef_study_hours = 0.03598
coef_class_attendance = 0.19803
coef_previous_gpa = -0.00508
coef_social_activities = -0.00473
coef_library_visits = 0.01903
coef_is_scholarship_holder = -0.0038

# Calcular el predicted_gpa usando la fórmula de regresión
df['predicted_gpa'] = (
    intercept
    + (coef_study_hours * df['study_hours'])
    + (coef_class_attendance * df['class_attendance'])
    + (coef_previous_gpa * df['previous_gpa'])
    + (coef_social_activities * df['social_activities'])
    + (coef_library_visits * df['library_visits'])
    + (coef_is_scholarship_holder * df['is_scholarship_holder'])
)

# Mostrar los primeros resultados
print(df[['student_id', 'predicted_gpa']].head())



# Cerrar la conexión
engine.dispose()
