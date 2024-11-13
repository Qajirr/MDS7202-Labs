import os
import subprocess

# Configuración de variables
GITLAB_REPO = "https://gitlab.com/mds7202-2/proyecto-mds7202.git"
LOCAL_FOLDER = "Datos"

def run_command(command, cwd=None):
    """Ejecuta un comando en la terminal y maneja errores."""
    try:
        result = subprocess.run(command, cwd=cwd, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e.stderr}")

if os.path.isdir(LOCAL_FOLDER):
    # Actualiza el repositorio si la carpeta ya existe
    print("Actualizando el repositorio existente...")
    run_command("git pull origin main", cwd=LOCAL_FOLDER)
else:
    # Clona el repositorio si la carpeta no existe
    print("Clonando el repositorio desde GitLab...")
    run_command(f"git clone {GITLAB_REPO} {LOCAL_FOLDER}")

print("Sincronización completa.")
