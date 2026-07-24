# Отбор кадров стройплощадки: контраст, жёсткие тени, метаданные для ACDSee — отчёт-исследование

<aside>
🎯

**Краткий вывод.** Для быстрого рабочего прототипа достаточно классического CV без GPU: **Python + OpenCV (headless) + NumPy** для метрик контраста и теневой маски, **ExifTool (через pyexiftool)** для записи рейтинга/ключевых слов в XMP так, чтобы ACDSee их видел и фильтровал. Готовой «под ключ» системы именно под вашу задачу нет, но есть репозитории, из которых собирается 80% решения: детектор теней для аэрофотосъёмки, фреймворки photo-culling с локальным веб-интерфейсом и IQA-библиотеки.

</aside>

# 1. Готовые инструменты и репозитории на GitHub

## 1.1. Ближайшие к задаче «фреймворк отбора фото на локальном сервере»

| Репозиторий | Что делает | Насколько подходит |
| --- | --- | --- |
| [Facet (ncoevoet/facet)](https://github.com/ncoevoet/facet) | Локальный движок анализа и отбора фото: скоринг каждого снимка по 9 измерениям, просмотр/culling через **локальную веб-галерею**, всё работает на своей машине, без облака | Лучший образец архитектуры «свой фреймворк на локальном сервере». Критерии скоринга придётся заменить своими (контраст/тени), но каркас — то, что вы описали |
| [photo-quality-analyzer](https://github.com/prasadabhishek/photo-quality-analyzer) | CLI/SDK на Python: пакетный анализ папки, метрики (резкость FFT, экспозиция по зонной системе, шум, динамический диапазон), вердикт и авто-раскладка по папкам; OpenCV headless + ONNX Runtime, ориентирован на CPU (заявлено 10 000+ фото) | Готовый каркас пакетного CLI-пайплайна «папка → метрики → классификация → сортировка». Метрики теней нужно дописать |
| [Тема photo-culling на GitHub](https://github.com/topics/photo-culling?o=desc&s=updated) | Десятки культ-инструментов: локальные AI-каллеры, группировка серий, скоринг резкости/экспозиции, **экспорт результатов в XMP** | Источник идей и кода: несколько проектов уже пишут вердикты в XMP — ровно ваш пункт 4 |
| [QuickRawPicker](https://github.com/RawLabo/QuickRawPicker) | Открытый GUI для отбора/рейтинга фото, совместим с XMP-sidecar (Bridge/Lightroom/Darktable) | Пример этапа ручной корректировки разметки поверх авторазметки |

## 1.2. Shadow detection: специализированные репозитории

**Классика (CPU, без обучения) — рекомендуемый старт:**

- [Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images) — Python-реализация метода Silva et al. (2017) «Near Real-Time Shadow Detection and Removal in Aerial Motion Imagery»: перевод RGB → CIELCh, модифицированный Specthem ratio, многоуровневая бинаризация (в репо — через K-Means), морфологическая очистка маски. Написан **именно для съёмки с высоты**, точность детекции у авторов метода ~93% в near real-time. Зависимости: OpenCV, NumPy, scikit-image, scikit-learn.
- [kittenish/Image-Shadow-Detection-and-Removal](https://github.com/kittenish/Image-Shadow-Detection-and-Removal) — детекция теней по парным регионам (mean shift сегментация + признаки YCbCr/HSI, градиент, текстура), **без датасетов и обучения**.
- [Гист адаптивной коррекции теней на OpenCV](https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df) и разбор [Multi-Scale Retinex + маски в LAB/HSV](https://www.edge-ai-vision.com/2026/02/enhancing-images-adaptive-shadow-correction-using-opencv/) — компактные рецепты построения теневой маски на чистом OpenCV.

**Предобученные нейросети (точнее, но тяжелее; CPU-инференс возможен, но медленный):**

- [BDRAR](https://github.com/zijundeng/BDRAR) (ECCV 2018) — эталонный детектор теней с выложенной обученной моделью; есть [обновлённый порт на PyTorch 1.7 + Colab](https://github.com/RolandGao/Shadow-Detection-and-Removal).
- [MTMT](https://github.com/eraserNut/MTMT) (CVPR 2020) — semi-supervised детекция теней, веса выложены.
- [ECA-ShadowDetection](https://github.com/AHU-VRV/ECA-ShadowDetection) (ACM MM 2021) — robust-детекция, есть pretrained-модели.
- [Detect-AnyShadow](https://github.com/harrytea/Detect-AnyShadow) — детекция теней на базе дообученного SAM (для видео, но применим покадрово).
- [Unveiling-Deep-Shadows](https://github.com/xw-hu/Unveiling-Deep-Shadows) — свежий survey + benchmark по всем DL-методам детекции/удаления теней, с моделями. Хорошая карта местности, если классики не хватит.

## 1.3. Оценка качества изображения (IQA)

- [IQA-PyTorch / pyiqa](https://github.com/chaofengc/iqa-pytorch) — большой toolbox no-reference метрик (BRISQUE, NIQE, PIQE, MUSIQ, TOPIQ и др.), `pip install pyiqa`, работает и на CPU.
- [brisque (PyPI)](https://pypi.org/project/brisque/) и [EadCat/NIQA](https://github.com/EadCat/NIQA) — лёгкие обёртки BRISQUE/NIQE без тяжёлых зависимостей.
- [Awesome-Image-Quality-Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) — каталог статей/кода по IQA.
- [AOtools contrast.py](https://github.com/AOtools/aotools/blob/master/aotools/image_processing/contrast.py) — готовые функции Michelson- и RMS-контраста на NumPy (можно просто скопировать).

⚠️ Важно: BRISQUE/NIQE меряют «техническое качество» (шум, блюр), а не тени. Для вашей задачи они — вспомогательный сигнал, основной должен быть теневой/контрастный.

---

# 2. Методы и метрики измерения контраста

## 2.1. Общепринятые метрики

| Метрика | Формула/суть | Применимость к задаче |
| --- | --- | --- |
| **Michelson** | (Imax − Imin) / (Imax + Imin) | Глобальная, чувствительна к единичным пикселям — использовать только по перцентилям (напр. P99 и P1) |
| **RMS-контраст** | Стандартное отклонение яркости (нормированное) | Простая и устойчивая глобальная метрика; хороший первый фильтр |
| **Weber** | (I − Iфон)/Iфон | Для объекта на фоне; полезна как «глубина тени»: отношение яркости внутри тени к соседнему освещённому участку |
| **Локальный контраст** | RMS в скользящих окнах (напр. 32–64 px), затем P95 по окнам | Ловит именно жёсткие локальные перепады, а не общий тон; одна из ключевых метрик для вас |
| **Entropy-based** | Энтропия гистограммы яркости | Косвенная: у «мягких» кадров гистограмма компактнее; полезна как доп. признак |
| **Гистограммные** | Межперцентильный размах (P95−P5), бимодальность (например, критерий Отсу: межклассовая дисперсия) | Бимодальная гистограмма яркости — типичный признак кадра «свет/тень» |

## 2.2. Что работает именно для жёстких теней

Жёсткая тень отличается от «просто контраста» **резкой границей** (узкая полутень) и **хроматическими свойствами**: в тени падает яркость (L), при этом цветность меняется слабо, а насыщенность и доля синего растут (небо подсвечивает тень). Отсюда практичный набор признаков:

1. **Теневая маска по цветовым пространствам**: порог по L (LAB) / V (HSV) с поправкой на соотношение каналов; Specthem ratio в CIELCh (метод Silva 2017 — тот самый аэрофото-репозиторий); RGB→LAB-подход — стандартный классический приём.
2. **Доля площади тени**: % пикселей в маске — прямой признак «плохого» кадра.
3. **Резкость границы тени**: средний градиент (Sobel/Scharr) вдоль контура теневой маски. У жёсткой тени градиент высокий, у мягкой — размазан. Это отделяет «ползущие» жёсткие тени от общего вечернего затемнения. Именно границы теней и цепляет ваша стабилизация, так что метрика напрямую моделирует проблему.
4. **Бимодальность гистограммы яркости** (двугорбость свет/тень) — быстрый глобальный признак.
5. Дополнительно: **время съёмки из EXIF** (высота солнца) — дешёвый априорный признак, у стационарной камеры отлично коррелирует с жёсткостью теней.

---

# 3. Инструменты и библиотеки (сравнение)

| Библиотека | Сильные стороны | Слабые стороны | Роль в проекте |
| --- | --- | --- | --- |
| **OpenCV (opencv-python-headless)** | Самая быстрая (C++ ядро), всё нужное: цветовые пространства, пороги, морфология, градиенты, CLAHE | API «сишный» | Основной движок метрик и маски теней |
| **scikit-image** | Удобные готовые функции (entropy, Otsu multi-level, measure.regionprops) | Медленнее OpenCV в 2–10 раз | Прототипирование метрик, свойства регионов |
| **Pillow** | Просто открыть/уменьшить файл, читает XMP | Нет серьёзного CV | Загрузка/превью, не для анализа |
| **NumPy** | Все метрики контраста — 5–10 строк | — | Расчёты поверх массивов |
| **pyiqa / brisque** | Готовые no-reference оценки качества | Не про тени; PyTorch-зависимость (pyiqa) | Опциональный доп. скоринг |
| **DL-репозитории теней (BDRAR/MTMT/ECA)** | Максимальная точность маски | GPU желателен, окружение старое (BDRAR — PyTorch 0.4/Python 2.7 в оригинале, лучше брать порт), тяжело для тысяч кадров на CPU | Этап 2, если классика не дотянет |

**Без GPU реально:** классический пайплайн (downscale до ~1024 px по длинной стороне + OpenCV) обрабатывает порядка 5–20 кадров/сек на обычном ПК c распараллеливанием через `multiprocessing`. Метод Silva 2017 позиционируется как near real-time.

---

# 4. Работа с метаданными под ACDSee

## 4.1. Какие поля использовать

Ключевые факты с форума ACDSee и ExifTool:

- **Rating**: ACDSee при встраивании пишет рейтинг в `XMP-acdsee:Rating` **и** в стандартный `XMP-xmp:Rating` (стандарт IPTC с 2017). Пишите **оба** — тогда рейтинг видят и ACDSee, и Lightroom/Bridge/XnView.
- **Цветные метки (Labels)**: стандартное поле `XMP-xmp:Label` — ACDSee встраивает метки туда же. Удобно: «хороший» = зелёный, «пограничный» = жёлтый, «плохой» = красный.
- **Keywords**: `XMP-dc:Subject` + `IPTC:Keywords` (например, `shadow_hard`, `shadow_soft`, `contrast_high`). Учтите: IPTC-ключевые слова не поддерживают иерархию, иерархию умеют только собственные keywords ACDSee.
- **Categories**: поле `XMP-acdsee:Categories` (XML-строка) ExifTool умеет писать, но по опыту форума ExifTool **ACDSee игнорирует категории, записанные сторонним софтом** — категории лучше не использовать, ограничьтесь Rating + Label + Keywords.
- **Пользовательские XMP-поля**: можно завести свой namespace (например, `xmp:ShadowScore`) через конфиг ExifTool — удобно хранить сырые числовые метрики для перекалибровки без пересчёта, но фильтровать по ним в ACDSee неудобно; фильтрация — по Rating/Label/Keywords.

**Критично для JPEG**: ACDSee **не читает XMP-sidecar для JPEG** — метаданные нужно **встраивать в сам файл** (sidecar он использует для RAW). После записи метаданных снаружи файлы нужно **перекаталогизировать** (Catalog), чтобы база ACDSee подхватила изменения.

## 4.2. Библиотеки записи метаданных

| Инструмент | Плюсы | Минусы |
| --- | --- | --- |
| [**ExifTool](https://exiftool.org/) (+ pyexiftool)** | Золотой стандарт; знает namespace `XMP-acdsee` (есть даже [официальный конфиг acdsee.config](https://github.com/exiftool/exiftool/blob/master/config_files/acdsee.config)); batch-режим; 400+ форматов; безопасная запись с бэкапом | Внешняя Perl-зависимость; на каждый файл — перезапись (в batch-режиме `-stay_open` быстро) |
| [**pyexiv2 (LeoHsiao1)**](https://github.com/LeoHsiao1/pyexiv2) | Нативный Python-биндинг exiv2: EXIF+IPTC+XMP, без внешних процессов | Регистрация кастомных namespace сложнее; GPL-3.0; менее «всеяден», чем ExifTool |
| **piexif** | Чистый Python, лёгкий | **Только EXIF**, XMP/IPTC не умеет — для вашей задачи не подходит |
| **python-xmp-toolkit (Adobe XMP SDK)** | Полноценная работа с XMP | Малоактивен, тяжёлая нативная зависимость (Exempi), обычно избыточен |

Рекомендация: **ExifTool через pyexiftool в batch-режиме** — надёжнее всего для смешанных форматов и ACDSee-специфичных тегов; pyexiv2 — запасной вариант «без внешних бинарников».

Пример команды разметки одного файла:

```bash
exiftool -overwrite_original \
  -XMP-xmp:Rating=2 -XMP-acdsee:Rating=2 \
  -XMP-xmp:Label="Red" \
  -XMP-dc:Subject+="shadow_hard" -IPTC:Keywords+="shadow_hard" \
  photo.jpg
```

---

# 5. Архитектура решения: 3 варианта

## Вариант A — CLI-скрипт (рекомендован как старт)

- **Пайплайн**: обход папки → downscale до 1024 px → расчёт метрик (RMS, локальный контраст P95, доля тени, градиент границы тени, бимодальность) → классификация по порогам (good / borderline / bad) → запись Rating/Label/Keywords через ExifTool batch → CSV-отчёт с сырыми метриками.
- **Библиотеки**: opencv-python-headless, numpy, pyexiftool, pyyaml, tqdm; опционально multiprocessing.
- **Объём/сложность**: ~300–500 строк, 1–3 дня работы. Можно взять за каркас photo-quality-analyzer и заменить метрики.
- **Пороги**: YAML-конфиг вида `shadow_area_max: 0.15`, `edge_gradient_max: ...`, `local_contrast_p95_max: ...`; режим `--dry-run` (только CSV, без записи в файлы).

## Вариант B — CLI + простой GUI/HTML-отчёт

- **Пайплайн**: тот же движок + генерация статического HTML-отчёта с превью, метриками и тепловизуализацией теневой маски (наглядно, почему кадр «плохой») или мини-GUI на **Gradio/Streamlit** с ползунками порогов и живым пересчётом классов по кэшированным метрикам (пересчитывать метрики не нужно — меняется только классификация).
- **Библиотеки**: + gradio или streamlit (или jinja2 для HTML).
- **Объём**: +200–400 строк, ещё 1–2 дня. Подбор порогов ускоряется на порядок.

## Вариант C — локальный веб-фреймворк с ручной корректировкой (то, что вы назвали «свой фреймворк на локальном сервере»)

- **Пайплайн**: FastAPI + SQLite: воркер считает метрики и складывает в БД → веб-галерея (сортировка по score, фильтры, теневые маски поверх фото) → ручное подтверждение/переопределение класса → кнопка «Применить» пишет финальную разметку в XMP батчем → накопленные ручные правки используются для перекалибровки порогов (или обучения логистической регрессии на 5–6 метриках).
- **Библиотеки**: fastapi, uvicorn, sqlite, opencv, pyexiftool; фронт — простой HTML+JS.
- **Объём**: 1,5–3 тыс. строк, 1–2 недели. **Перед написанием с нуля посмотрите Facet** — это почти готовая реализация такой архитектуры (локальный скоринг + веб-галерея + culling), которую можно форкнуть и заменить скоринговый модуль.
- **Пороги**: хранятся в БД, правятся из UI; версии порогов логируются.

---

# 6. Практические рекомендации

**Специфика промышленных фото (бетон, металл, однородные поверхности):**

- Большие однородные светлые поверхности (бетон) дают низкую энтропию и обманывают глобальные метрики — опирайтесь на **локальные** метрики и долю/границы теневой маски, а не на глобальный контраст.
- Металл даёт **блики**: это тоже высокий локальный контраст, но не тень. Отличать по знаку: тень = тёмный кластер с горячей границей; блик = насыщение в светах (проверка доли пикселей > 250).
- Стационарная камера — ваш козырь: сцена постоянна, значит пороги стабильны, а **медианный кадр по серии** можно использовать как reference «без теней» и мерить отклонение от него (это сильнее любой одиночной метрики).
- Время суток из EXIF: часы с низким солнцем почти гарантированно дают жёсткие тени — дешёвый предфильтр.

**Калибровка порогов:**

1. Разметьте вручную 100–300 кадров на 3 класса (это быстро — вы это и так делаете).
2. Посчитайте все метрики по размеченному набору → постройте распределения по классам; пороги ставьте по точке пересечения распределений или подберите по ROC (sklearn, отчёт precision/recall).
3. Если одиночные пороги дают много ошибок — логистическая регрессия/малое дерево решений на 5–6 метриках: обучается на CPU за секунды и остаётся интерпретируемой.
4. «Пограничный» класс делайте широким: дешевле вручную просмотреть 10% кадров, чем потерять хорошие.

**Подводные камни:**

- Разный баланс белого → все метрики считать по **яркостному каналу** (L из LAB или Y), а хроматические признаки использовать только как отношения каналов, не абсолютные значения.
- Автоэкспозиция камеры маскирует общий уровень света → нормализуйте гистограмму (или CLAHE) перед расчётом относительных метрик, но **градиент границы тени считайте по исходнику**.
- JPEG-сжатие даёт блочные артефакты 8×8 и ложные градиенты на однородных поверхностях → лёгкое размытие (Gaussian σ≈1) перед градиентными метриками.
- Облачная кромка может дать «псевдотень» с мягкой границей — метрика резкости границы как раз отделит её от жёсткой тени.
- Пишите метаданные **атомарно** (ExifTool с временным файлом) и сначала гоняйте в dry-run: массовая порча EXIF на потоке с дрона — реальный риск.

---

# 7. Итоговая рекомендация

**Стек прототипа (1-я неделя):** Python 3.11 + opencv-python-headless + NumPy → 5 метрик (RMS, локальный контраст P95, доля теневой маски по L-каналу, средний градиент границы маски, бимодальность гистограммы) → пороги в YAML → ExifTool (pyexiftool, batch) пишет `XMP-xmp:Rating` + `XMP-acdsee:Rating` + `XMP-xmp:Label` + keywords → CSV с сырыми метриками. За основу маски теней взять [аэрофото-репозиторий](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images) (метод Silva 2017), за основу CLI — [photo-quality-analyzer](https://github.com/prasadabhishek/photo-quality-analyzer).

**2-я итерация:** Streamlit-панель подбора порогов по кэшированным метрикам + калибровка на размеченном наборе (ROC/логрег).

**Если классики не хватит:** прикрутить предобученный [MTMT](https://github.com/eraserNut/MTMT) или [BDRAR (порт)](https://github.com/RolandGao/Shadow-Detection-and-Removal) только для «пограничных» кадров — так GPU не обязателен, а точность растёт там, где нужно.

**Если захочется полноценный локальный фреймворк с UI:** форк [Facet](https://github.com/ncoevoet/facet) с заменой скорингового модуля на ваши теневые метрики.

---

## Источники

- Silva et al., [Near real-time shadow detection and removal in aerial motion imagery](https://www.sciencedirect.com/science/article/abs/pii/S0924271617302253) (ISPRS 2017) и [Python-реализация](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images)
- [Unveiling Deep Shadows — survey/benchmark по shadow detection](https://github.com/xw-hu/Unveiling-Deep-Shadows); [BDRAR](https://github.com/zijundeng/BDRAR); [MTMT](https://github.com/eraserNut/MTMT); [ECA-ShadowDetection](https://github.com/AHU-VRV/ECA-ShadowDetection); [Detect-AnyShadow](https://github.com/harrytea/Detect-AnyShadow)
- [Адаптивная коррекция теней в OpenCV (MSR + LAB/HSV)](https://www.edge-ai-vision.com/2026/02/enhancing-images-adaptive-shadow-correction-using-opencv/)
- Метрики контраста: [pylinac docs](https://pylinac.readthedocs.io/en/latest/topics/contrast.html), [AOtools contrast.py](https://github.com/AOtools/aotools/blob/master/aotools/image_processing/contrast.py)
- IQA: [IQA-PyTorch](https://github.com/chaofengc/iqa-pytorch), [brisque на PyPI](https://pypi.org/project/brisque/), [Awesome-IQA](https://github.com/chaofengc/Awesome-Image-Quality-Assessment)
- ACDSee и метаданные: [XMP versus IPTC versus EXIF (форум ACDSee)](https://forum.acdsee.com/forum/main-category/acdsee-ultimate/60576-xmp-versus-iptc-versus-exif-data), [Embedding metadata — readable by other software](https://forum.acdsee.com/forum/main-category/acdsee-ultimate/53570-embedding-metadata-is-it-readable-by-other-software), [Can ACDSee read XMP rating for JPEG](https://forum.acdsee.com/forum/main-category/acdsee-ultimate/61704-can-acdsee-read-xmp-rating-metadata-for-jpeg-files), [ExifTool forum: XMP-acdsee namespace](https://exiftool.org/forum/index.php?topic=2561.0), [acdsee.config в ExifTool](https://github.com/exiftool/exiftool/blob/master/config_files/acdsee.config)
- Culling-фреймворки: [Facet](https://github.com/ncoevoet/facet), [photo-quality-analyzer](https://github.com/prasadabhishek/photo-quality-analyzer), [QuickRawPicker](https://github.com/RawLabo/QuickRawPicker), [топик photo-culling](https://github.com/topics/photo-culling?o=desc&s=updated)
- Метаданные-библиотеки: [pyexiv2](https://github.com/LeoHsiao1/pyexiv2), [обзор pyexiv2/аналогов на PyPI](https://pypi.org/project/pyexiv2/), [ExifTool](https://exiftool.org/)