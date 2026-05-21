@echo off
echo ============================================================
echo  SIMULADOR BURSATIL - Maestria en Economia y Finanzas
echo ============================================================
echo.

echo [1/3] Instalando dependencias de Python...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion. Asegurate de tener Python instalado.
    pause
    exit /b 1
)

echo.
echo [2/3] Verificando instalacion...
python -c "import streamlit, pandas, numpy, scipy, plotly, sklearn; print('OK - Todas las librerias instaladas')"
if %errorlevel% neq 0 (
    echo ERROR: Algunas librerias no se instalaron correctamente.
    pause
    exit /b 1
)

echo.
echo [3/3] Iniciando el Simulador Bursatil...
echo.
echo >> Abre tu navegador en: http://localhost:8501
echo >> Para detener el servidor: Ctrl + C
echo.
streamlit run app.py --server.port 8501

pause
