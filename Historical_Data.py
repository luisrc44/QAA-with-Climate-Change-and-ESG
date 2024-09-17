import requests

# Tu clave API de Cnaught
API_KEY = "C0-BT6R7ujYGkTEyTb0nGYVREEX0jwfKIY84"

# Definir la URL base para las emisiones de CO2 (ajústala según la documentación de Cnaught)
base_url = "https://api.cnaught.com/v1/co2/emissions"  # Ajusta el endpoint según la documentación

# Headers para la autenticación
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Parámetros opcionales (ajústalos según la API)
params = {
    "country": "world",  # Puedes especificar un país o dejarlo en "world" para global
    "start_year": 2000,   # Año de inicio
    "end_year": 2020      # Año de fin
}

# Realizar la solicitud GET
response = requests.get(base_url, headers=headers, params=params)

# Verificar el estado de la solicitud
if response.status_code == 200:
    # Convertir la respuesta en formato JSON
    data = response.json()
    print("Datos recibidos:", data)
    
    # Guardar los datos en un archivo JSON
    with open("co2_emissions_cnaught.json", "w") as f:
        f.write(response.text)
    print("Datos guardados en 'co2_emissions_cnaught.json'.")
else:
    print(f"Error en la solicitud: {response.status_code} - {response.text}")
