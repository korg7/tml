@echo off
chcp 65001 >nul
echo ============================================================
echo   Shadow-QC: автоматическая установка зависимостей
echo ============================================================
echo.

:: Проверка Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ОШИБКА] Python не найден в PATH.
    echo.
    echo   Скачайте и установите Python 3.11+ с https://www.python.org/downloads/
    echo   При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

:: Проверка версии Python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% найден.
echo.

:: Создание виртуального окружения
if not exist "shadow-qc\.venv" (
    echo [1/3] Создание виртуального окружения...
    python -m venv shadow-qc\.venv
    if %ERRORLEVEL% neq 0 (
        echo [ОШИБКА] Не удалось создать venv.
        pause
        exit /b 1
    )
    echo       Создано: shadow-qc\.venv
) else (
    echo [1/3] Виртуальное окружение уже существует.
)
echo.

:: Активация и установка зависимостей
echo [2/3] Установка Python-библиотек...
call shadow-qc\.venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r shadow-qc\requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
)
echo.

:: Проверка ExifTool
echo [3/3] Проверка ExifTool...
where exiftool >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo   [ВНИМАНИЕ] ExifTool не найден в PATH!
    echo   Он нужен ТОЛЬКО для записи метаданных в фото (--write).
    echo   Для анализа в режиме --dry-run он НЕ требуется.
    echo.
    echo   Установка:
    echo     1. Скачайте Windows Executable с https://exiftool.org/
    echo     2. Распакуте .zip
    echo     3. Переименуйте exiftool(-k).exe в exiftool.exe
    echo     4. Положите в папку, добавленную в PATH
    echo        (например C:\Windows или создайте C:\tools и добавьте в PATH)
    echo.
) else (
    echo       [OK] ExifTool найден.
)

echo.
echo ============================================================
echo   Установка завершена!
echo ============================================================
echo.
echo   Активация окружения:
echo     shadow-qc\.venv\Scripts\activate
echo.
echo   Запуск анализа (тест):
echo     cd shadow-qc
echo     python -m src --input data/input --dry-run --debug
echo.
echo   Подробнее: README.md
echo.
pause
