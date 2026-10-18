import os
import sqlite3
import io
import base64
import json
import re
import traceback
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from pyngrok import ngrok
    HAS_NGROK = True
except ImportError:
    HAS_NGROK = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

app = Flask(__name__)

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            tipo TEXT NOT NULL,
            identificador TEXT NOT NULL,
            parametros TEXT NOT NULL,
            resultado_nn TEXT NOT NULL,
            analisis_ia TEXT,
            foto_data TEXT,
            eliminado INTEGER DEFAULT 0,
            justificacion_eliminado TEXT DEFAULT ''
        )
    ''')
    
    cursor.execute("PRAGMA table_info(registros)")
    columnas = [col[1] for col in cursor.fetchall()]
    if "eliminado" not in columnas:
        cursor.execute("ALTER TABLE registros ADD COLUMN eliminado INTEGER DEFAULT 0")
    if "justificacion_eliminado" not in columnas:
        cursor.execute("ALTER TABLE registros ADD COLUMN justificacion_eliminado TEXT DEFAULT ''")

    conn.commit()
    conn.close()

def guardar_registro(registro):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO registros (fecha, tipo, identificador, parametros, resultado_nn, analisis_ia, foto_data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        registro["fecha"],
        registro["tipo"],
        registro["identificador"],
        registro["parametros"],
        registro["resultado_nn"],
        registro["analisis_ia"],
        registro["foto_data"]
    ))
    conn.commit()
    conn.close()

def obtener_sujetos_unicos():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT identificador, tipo FROM registros ORDER BY identificador ASC")
    filas = cursor.fetchall()
    conn.close()
    return [{"identificador": f[0], "tipo": f[1]} for f in filas]

def obtener_historial_sujeto(identificador=None):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if identificador:
        cursor.execute("SELECT * FROM registros WHERE identificador = ? ORDER BY id DESC", (identificador,))
    else:
        cursor.execute("SELECT * FROM registros ORDER BY id DESC")
    filas = cursor.fetchall()
    conn.close()
    return [dict(f) for f in filas]

# --- MODELO DE CULTIVOS (AHORA EVALÚA TAMAÑO/ALTURA EN CM Y TEMPERATURA) ---
torch.manual_seed(42)
np.random.seed(42)

class CultivationNN(nn.Module):
    def __init__(self):
        super(CultivationNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# Entrenar modelo de cultivo con Altura (cm) y Temperatura (°C)
X_cultivo = np.random.rand(100, 2)
X_cultivo[:, 0] = X_cultivo[:, 0] * 150 + 10  # Altura de planta de 10cm a 160cm
X_cultivo[:, 1] = X_cultivo[:, 1] * 25 + 15   # Temperatura de 15°C a 40°C
y_cultivo = (X_cultivo[:, 0]/160.0 * 0.7 + X_cultivo[:, 1]/40.0 * 0.3).reshape(-1, 1)

modelo_cultivo = CultivationNN()
optimizer_c = optim.Adam(modelo_cultivo.parameters(), lr=0.01)
criterion_c = nn.MSELoss()
for _ in range(120):
    optimizer_c.zero_grad()
    loss = criterion_c(modelo_cultivo(torch.tensor(X_cultivo, dtype=torch.float32)), torch.tensor(y_cultivo, dtype=torch.float32))
    loss.backward()
    optimizer_c.step()

# --- MODELO PORCINO ---
class PigHealthNN(nn.Module):
    def __init__(self):
        super(PigHealthNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

X_cerdos = np.random.rand(100, 2)
X_cerdos[:, 0] = X_cerdos[:, 0] * 100 + 50
X_cerdos[:, 1] = X_cerdos[:, 1] * 80 + 80
y_cerdos = np.array([[1.0] if (p > 95.0 and l > 110.0) else [0.0] for p, l in X_cerdos])

modelo_cerdos = PigHealthNN()
optimizer_p = optim.Adam(modelo_cerdos.parameters(), lr=0.01)
criterion_p = nn.BCELoss()
for _ in range(150):
    optimizer_p.zero_grad()
    loss_p = criterion_p(modelo_cerdos(torch.tensor(X_cerdos, dtype=torch.float32)), torch.tensor(y_cerdos, dtype=torch.float32))
    loss_p.backward()
    optimizer_p.step()

# --- RUTAS ---

@app.route("/", methods=["GET", "POST"])
def home():
    resultado = False
    resultado_nn = ""
    analisis_ia = ""
    identificador = ""
    error_msg = None
    sujeto_seleccionado = request.args.get("sujeto")

    if request.method == "POST":
        try:
            tipo = request.form.get("tipo")
            foto_base64 = None

            if tipo == "cultivo":
                identificador = request.form.get("nombre_producto", "Cultivo Genérico").strip()
                altura = float(request.form.get("altura", 0))  # Nuevo parámetro: Altura de la planta
                temp = float(request.form.get("temp", 0))
                entrada = torch.tensor([[altura, temp]], dtype=torch.float32)
                pred = modelo_cultivo(entrada).item()
                resultado_nn = f"Índice de Crecimiento/Salud: {pred:.2f} (Escala 0-1)"
                parametros = f"Altura: {altura}cm, Temp: {temp}°C"
                tipo_str = "Cultivo"
            else:
                identificador = request.form.get("identificador_porcino", "Animal Genérico").strip()
                peso = float(request.form.get("peso", 0))
                largo = float(request.form.get("largo", 0))
                entrada = torch.tensor([[peso, largo]], dtype=torch.float32)
                pred = modelo_cerdos(entrada).item()
                estado = "Apto para venta/reproducción" if pred > 0.5 else "En etapa de crecimiento"
                resultado_nn = f"Estado: {estado} (Confianza: {pred:.2f})"
                parametros = f"Peso: {peso}kg, Largo: {largo}cm"
                tipo_str = "Porcino"

            resultado = True

            file = request.files.get("foto")
            if file and file.filename != "":
                file_bytes = file.read()
                foto_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                api_key = request.form.get("manual_api_key") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                
                if api_key and HAS_GENAI:
                    try:
                        img = Image.open(io.BytesIO(file_bytes))
                        client = genai.Client(api_key=api_key)
                        prompt = f"Analiza detalladamente el crecimiento en tamaño/altura o salud visual para el registro de cultivo/animal '{identificador}'. Proporciona observaciones y recomendaciones."
                        
                        response = client.models.generate_content(
                            model='gemini-3.8-flash',
                            contents=[img, prompt]
                        )
                        analisis_ia = response.text
                    except Exception as ge:
                        analisis_ia = f"Nota sobre Gemini: {str(ge)}"
                else:
                    analisis_ia = "Sin API Key para análisis de fotos."

            registro = {
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "tipo": tipo_str,
                "identificador": identificador,
                "parametros": parametros,
                "resultado_nn": resultado_nn,
                "analisis_ia": analisis_ia,
                "foto_data": foto_base64
            }
            guardar_registro(registro)
            sujeto_seleccionado = identificador

        except Exception as e:
            traceback.print_exc()
            error_msg = f"{str(type(e).__name__)}: {str(e)}"

    sujetos = obtener_sujetos_unicos()
    historial = []
    fechas = []
    m1 = []
    m2 = []

    if sujeto_seleccionado:
        historial = obtener_historial_sujeto(sujeto_seleccionado)
        historial_asc = list(reversed(historial))
        for item in historial_asc:
            if not item["eliminado"]:
                fechas.append(item["fecha"])
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", item["parametros"])
                if len(nums) >= 2:
                    m1.append(float(nums[0]))
                    m2.append(float(nums[1]))
                else:
                    m1.append(0)
                    m2.append(0)
    else:
        historial = obtener_historial_sujeto(None)

    return render_template(
        "index.html",
        sujetos=sujetos,
        sujeto_seleccionado=sujeto_seleccionado,
        historial=historial,
        resultado=resultado,
        resultado_nn=resultado_nn,
        analisis_ia=analisis_ia,
        identificador=identificador,
        error=error_msg,
        fechas_js=json.dumps(fechas),
        valores_m1_js=json.dumps(m1),
        valores_m2_js=json.dumps(m2)
    )

@app.route("/eliminar/<int:registro_id>", methods=["POST"])
def eliminar_registro(registro_id):
    justificacion = request.form.get("justificacion", "Eliminado por el usuario.").strip()
    sujeto = request.form.get("sujeto_actual", "")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE registros 
        SET eliminado = 1, justificacion_eliminado = ? 
        WHERE id = ?
    """, (justificacion, registro_id))
    conn.commit()
    conn.close()

    if sujeto:
        return redirect(f"/?sujeto={sujeto}")
    return redirect("/")

if __name__ == "__main__":
    init_db()
    PORT = 5000

    if HAS_NGROK:
        try:
            public_url = ngrok.connect(PORT)
            print("=" * 60)
            print(f" NGROK TUNNEL ACTIVO: {public_url}")
            print("=" * 60)
        except Exception as e:
            print(f"No se pudo iniciar ngrok automáticamente: {e}")
    app.run(host="0.0.0.0", port=PORT, debug=False)