# Shadow-QC: автоотбраковка кадров с жёсткими тенями

CLI-инструмент для пакетного анализа фотографий стройплощадки.
Классифицирует кадры по наличию жёстких теней: **good / review / bad**.

---

## Быстрый старт (Windows)

### 1. Установите Python (вручную)

Скачайте и установите **Python 3.11+** с [python.org](https://www.python.org/downloads/).

> **Важно:** при установке поставьте галочку **"Add Python to PATH"**.

Проверка:
```cmd
python --version
```

### 2. Клонируйте репозиторий

```cmd
git clone https://github.com/korg7/tml.git
cd tml
```

### 3. Запустите установщик

```cmd
setup.bat
```

Скрипт автоматически:
- создаст виртуальное окружение (`shadow-qc\.venv`)
- установит все Python-библиотеки из `requirements.txt`
- проверит наличие ExifTool

### 4. Запустите тестовый анализ

```cmd
cd shadow-qc
.venv\Scripts\activate
python -m src --input data/input --dry-run --debug
```

Результат: CSV с метриками + debug-изображения (маски, overlay).

---

## Что устанавливается автоматически

| Компонент | Как |
|-----------|-----|
| opencv-python-headless | pip (из requirements.txt) |
| numpy | pip |
| scikit-learn | pip |
| pandas | pip |
| pyyaml | pip |
| tqdm | pip |

---

## Что НЕ установится само (ручная установка)

### Python 3.11+

Скачать: https://www.python.org/downloads/

При установке **обязательно** включить "Add Python to PATH".

### ExifTool (опционально)

Нужен **только** для записи метаданных в файлы (режим `--write`).
Для анализа в режиме `--dry-run` **не требуется**.

Установка:
1. Скачайте "Windows Executable" с https://exiftool.org/
2. Распакуйте `.zip`
3. Переименуйте `exiftool(-k).exe` → `exiftool.exe`
4. Положите в папку из PATH (например `C:\Windows` или `C:\tools`)

Проверка:
```cmd
exiftool -ver
```

### Git

Для клонирования репозитория. Скачать: https://git-scm.com/download/win

---

## Использование

```cmd
cd shadow-qc
.venv\Scripts\activate

:: Анализ папки с фото (без записи метаданных)
python -m src --input "путь/к/фото" --dry-run

:: Анализ с сохранением debug-визуализации
python -m src --input "путь/к/фото" --dry-run --debug

:: Анализ + запись метаданных (Rating/Label/Keywords для ACDSee)
python -m src --input "путь/к/фото" --write

:: Указать свой конфиг
python -m src --input "путь/к/фото" --config my_config.yaml

:: Число потоков
python -m src --input "путь/к/фото" --workers 4
```

### Выходные данные

- `data/predictions.csv` — метрики и класс каждого файла
- `debug/masks/` — бинарные теневые маски (при `--debug`)
- `debug/overlays/` — фото с наложенной маской (при `--debug`)

---

## Конфигурация

Все параметры в `shadow-qc/config.yaml`:
- пороги классификации
- параметры теневой маски
- пути к данным
- настройки метаданных

---

## Структура проекта

```
tml/
├── setup.bat                  # автоустановка (Windows)
├── README.md
├── docs/                      # документация и исследования
└── shadow-qc/
    ├── config.yaml            # конфигурация
    ├── requirements.txt       # Python-зависимости
    ├── src/
    │   ├── __main__.py        # точка входа (python -m src)
    │   ├── cli.py             # аргументы командной строки
    │   ├── pipeline.py        # оркестрация пайплайна
    │   ├── features.py        # извлечение признаков
    │   ├── shadow_mask.py     # построение теневой маски
    │   ├── classify.py        # классификация (правила / ML)
    │   ├── metadata.py        # запись через ExifTool
    │   ├── utils.py           # IO, resize, LAB
    │   └── visualize_debug.py # debug-визуализация
    ├── scripts/
    │   ├── calibrate.py       # калибровка порогов
    │   └── visualize_debug.py # визуализация (скрипт)
    └── data/
        ├── input/             # входные фото
        └── predictions.csv    # результат
```

---

## Системные требования

- Windows 10/11 (также работает на Linux/macOS)
- Python 3.11+
- ~500 МБ диска (зависимости)
- CPU: любой многоядерный (параллельная обработка)
- GPU: **не требуется**
