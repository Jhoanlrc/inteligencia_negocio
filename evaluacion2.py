import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
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


#Pregunta 1 : ¿Cuál es el impacto de aumentar las horas de estudio en el promedio de calificaciones? 


# Gráfico de dispersión de study_hours vs. predicted_gpa con línea de regresión
# plt.figure(figsize=(10, 6))
# sns.regplot(x='study_hours', y='predicted_gpa', data=df, line_kws={"color": "red"})
# plt.title("Impacto de las Horas de Estudio en el Promedio de Calificaciones")
# plt.xlabel("Horas de Estudio por Semana")
# plt.ylabel("Predicted GPA")
# plt.show()



#Pregunta 2 : ¿Cuánto influye el rendimiento previo en el semestre actual, en comparación con factores como la asistencia a clases o las visitas a la biblioteca?
# Gráfico de barras de los coeficientes

# plt.figure(figsize=(8, 5))
# coef_data = {
#     'Factor': ['Rendimiento Previo', 'Asistencia a Clases', 'Visitas a la Biblioteca'],
#     'Coeficiente': [0.00508, 0.19803, 0.01903]
# }
# coef_df = pd.DataFrame(coef_data)
# sns.barplot(x='Factor', y='Coeficiente', data=coef_df)
# plt.title("Influencia de los Factores en el Promedio de Calificaciones")
# plt.ylabel("Coeficiente de Regresión")
# plt.show()

# # Gráficos de dispersión con líneas de regresión para cada variable
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Rendimiento Previo
# sns.regplot(x='previous_gpa', y='predicted_gpa', data=df, ax=axes[0], line_kws={"color": "red"})
# axes[0].set_title("Rendimiento Previo vs. Predicted GPA")
# axes[0].set_xlabel("Previous GPA")
# axes[0].set_ylabel("Predicted GPA")

# # Asistencia a Clases
# sns.regplot(x='class_attendance', y='predicted_gpa', data=df, ax=axes[1], line_kws={"color": "red"})
# axes[1].set_title("Asistencia a Clases vs. Predicted GPA")
# axes[1].set_xlabel("Class Attendance")

# # Visitas a la Biblioteca
# sns.regplot(x='library_visits', y='predicted_gpa', data=df, ax=axes[2], line_kws={"color": "red"})
# axes[2].set_title("Visitas a la Biblioteca vs. Predicted GPA")
# axes[2].set_xlabel("Library Visits")

# plt.tight_layout()
# plt.show()


# Pregunta 3 :¿Qué tan importante es la situación de beca en el rendimiento académico de los estudiantes?

# Gráfico de barras para el promedio de predicted_gpa según situación de beca
plt.figure(figsize=(8, 5))
sns.barplot(x='is_scholarship_holder', y='predicted_gpa', data=df, ci=None)
plt.xticks([0, 1], ['No Becado', 'Becado'])
plt.title("Promedio de Predicted GPA según Situación de Beca")
plt.xlabel("Situación de Beca")
plt.ylabel("Promedio de Predicted GPA")
plt.show()

# Gráfico de barras del coeficiente de situación de beca comparado con otros factores
plt.figure(figsize=(8, 5))
coef_data = {
    'Factor': ['Rendimiento Previo', 'Asistencia a Clases', 'Visitas a la Biblioteca', 'Situación de Beca'],
    'Coeficiente': [0.00508, 0.19803, 0.01903, -0.0038]
}
coef_df = pd.DataFrame(coef_data)
sns.barplot(x='Factor', y='Coeficiente', data=coef_df)
plt.title("Comparación de Coeficientes de Regresión para Factores Clave")
plt.ylabel("Coeficiente de Regresión")
plt.show()

# Cerrar la conexión
engine.dispose()
