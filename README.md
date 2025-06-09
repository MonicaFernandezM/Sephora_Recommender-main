# Sephora_Recommender-main

Presentation:
https://www.canva.com/design/DAGObkJMmmw/SJxybdr8ZfKIiumnPbOmbQ/edit


🧴 Skincare Product Recommender - Sephora Dataset

Este proyecto desarrolla un sistema de recomendación de productos de skincare basado en las características del usuario como tipo y color de piel, utilizando un dataset de productos de Sephora. Se aplican técnicas de análisis exploratorio, limpieza de datos y machine learning para sugerir productos adecuados a cada perfil.

⸻

📊 Objetivo

El objetivo es construir un modelo que ayude a los usuarios a encontrar productos de cuidado de la piel que se ajusten a sus necesidades individuales, basándose en datos reales.

⸻

🛠️ Tecnologías utilizadas
	•	Lenguaje: Python
	•	Librerías: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
	•	Entorno: Jupyter Notebook

⸻

🔍 Dataset
	•	Fuente: Kaggle - Sephora Skincare dataset
	•	Atributos clave:
	•	Tipo de piel (Skin Type)
	•	Color de piel (Skin Tone)
	•	Calificación de producto (Rating)
	•	Ingredientes
	•	Precio

⸻

📌 Proceso
	1.	Cargar y limpiar datos:
	•	Manejo de valores nulos
	•	Codificación de variables categóricas
	2.	Análisis exploratorio:
	•	Distribuciones por tipo y tono de piel
	•	Preferencias de productos por perfil
	3.	Modelo de recomendación:
	•	Filtrado basado en características del usuario
	•	Ranking de productos según calificación promedio
	4.	Evaluación:
	•	Métricas de precisión en recomendaciones
	•	Validación cruzada (si aplica)

⸻

✅ Resultados
	•	Se generaron recomendaciones personalizadas de skincare basadas en el perfil del usuario.
	•	El sistema filtra productos con mejor rendimiento según usuarios con piel y tono similares.
