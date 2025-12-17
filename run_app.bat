@echo off
title DataViz Pro - Professional Data Analytics
echo.
echo  =============================================
echo    DataViz Pro - Professional Data Analytics
echo  =============================================
echo.
echo  Starting the application...
echo  Your browser will open automatically.
echo.
echo  URL: http://localhost:8501
echo.
echo  Press Ctrl+C to stop the server
echo  =============================================
echo.
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.
streamlit run app.py --server.headless true
pause
