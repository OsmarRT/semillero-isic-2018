# Semillero ISIC 2018

Proyecto para entrenamiento de modelo sobre ISIC 2018.

Requisitos
- Git
- Python 3.11 (recomendado)

Instalación rápida

1. Clona el repositorio:

```
git clone https://github.com/OsmarRT/semillero-isic-2018.git
cd semillero-isic-2018
```

2. Crear y activar entorno virtual:

Windows (PowerShell):

```
python -m venv venv_isic
.\venv_isic\Scripts\Activate.ps1
```

Windows (cmd):

```
python -m venv venv_isic
.\venv_isic\Scripts\activate.bat
```

3. Instalar dependencias:

```
pip install -r requirements.txt
```

Si `requirements.txt` no existe, generarlo desde el entorno original (si dispone del `venv_isic`):

```
venv_isic\Scripts\python -m pip freeze > requirements.txt
```

Notas
- La carpeta `data/` no está incluida en el repositorio por tamaño; debe descargar y colocar los datos en `data/` siguiendo las instrucciones de ISIC.
- El archivo `best_model.pth` está en `.gitignore` (no subir pesos grandes al repo).
