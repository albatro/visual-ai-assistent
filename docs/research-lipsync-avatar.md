# Research: Real-Time Lip-Sync Avatar Generation

**Дата**: 2026-05-02
**GPU**: NVIDIA RTX 5060 Ti 16GB VRAM
**Задача**: Генерация живого видео-аватара с lip-sync из одного фото + аудио (голос)

---

## Требования

- Генерация говорящего аватара из 1 или нескольких фото
- Lip-sync по аудио (голосовое сообщение / TTS)
- Real-time или near-real-time (30+ FPS)
- Укладывается в 16GB VRAM
- Желательно open-source с коммерческой лицензией

---

## Сравнительная таблица решений

| Проект | GitHub Stars | VRAM | FPS | Real-time? | Из фото? | Лицензия |
|--------|:-----------:|:----:|:---:|:----------:|:--------:|----------|
| **MuseTalk 1.5** (Tencent) | ~5.7k | 4-6 GB | 30+ | **Да** | Да | Research (некоммерч.) |
| **LivePortrait** (Kuaishou) | ~18.2k | 6 GB | ~78 | **Да** | Да | MIT |
| **EchoMimicV3 Flash** (Ant Group) | ~0.9k | 12 GB | ~2.4 | Нет | Да | Apache 2.0 |
| **Hallo2** (Fudan) | ~3.7k | 24+ GB | <1 | Нет | Да | Apache 2.0 |
| **Sonic** (CVPR 2025) | ~4k | 16-32 GB | <1 | Нет | Да | Research |
| **SadTalker** | ~13.8k | 4 GB | <1 | Нет | Да | Apache 2.0 |
| **Wav2Lip** | ~12.7k | 1-2 GB | ~RT | Почти | Нет (видео) | Non-commercial |
| **LatentSync v1.5** (ByteDance) | ~5.6k | 8 GB | <1 | Нет | Нет (видео) | Apache 2.0 |
| **V-Express** (Tencent) | ~2.4k | 10-16 GB | <1 | Нет | Да | Research |
| **AniPortrait** | ~5k | 10-16 GB | <1 | Нет | Да | Apache 2.0 |

---

## Оценка качества (из 10)

| Проект | Lip-sync точность | Качество картинки | Естественность движений | Артефакты | **Итого** |
|--------|:-:|:-:|:-:|:-:|:-:|
| **LivePortrait** | — | 9 | 9 | 9 | **9.0** |
| **Hallo2** | 9 | 9 | 9 | 8 | **8.8** |
| **EchoMimicV3** | 8 | 9 | 8 | 8 | **8.5** |
| **Sonic** | 8 | 9 | 8 | 8 | **8.3** |
| **MuseTalk 1.5** | 8 | 7 | 7 | 7 | **7.5** |
| **LatentSync v1.5** | 8 | 8 | 7 | 7 | **7.5** |
| **SadTalker** | 6 | 6 | 7 | 6 | **6.2** |
| **Wav2Lip** | 9 | 4 | 3 | 4 | **5.0** |

---

## Детальный разбор топ-кандидатов

### 1. MuseTalk 1.5 (Tencent) — ЛУЧШИЙ ДЛЯ REAL-TIME

- **Repo**: https://github.com/TMElyralab/MuseTalk
- **Подход**: Латентный диффузионный inpainting области рта (256×256)
- **VRAM**: 4-6 GB (fp16), batch_size=4 даёт ~30 FPS
- **Плюсы**:
  - Единственное решение с настоящим real-time из фото + аудио
  - Очень экономичен по VRAM — остаётся место для TTS и других моделей
  - MuseTalk 1.5 добавил perceptual loss, GAN loss, sync loss — резче картинка
- **Минусы**:
  - Лицензия research-only (некоммерческая)
  - Установка сложная ("dependency hell")
  - Лицо кропается 256×256 — при увеличении видна размытость
  - Зубы иногда "плывут", кросс-языковая работа нестабильна
- **Вердикт**: Лучший выбор для real-time прототипа/исследования

### 2. LivePortrait (Kuaishou) — ЛУЧШЕЕ КАЧЕСТВО + MIT

- **Repo**: https://github.com/KlingTeam/LivePortrait
- **Подход**: Stitching + warping модулей, motion transfer с видео/вебкамеры
- **VRAM**: 6 GB, 12.8ms/frame на RTX 4090
- **Плюсы**:
  - Самое высокое качество визуала (9/10)
  - MIT лицензия — коммерческое использование
  - 18k+ звёзд, активная поддержка
  - Очень быстрый (~78 FPS)
- **Минусы**:
  - **Нет нативного audio-driven режима** — анимация только от видео/вебкамеры
  - Нужна прослойка audio → facial landmarks → LivePortrait
  - Требует инженерных усилий для полного pipeline
- **Вердикт**: Лучший для коммерческого продукта, если допилить audio pipeline

### 3. EchoMimicV3 Flash (Ant Group) — ЛУЧШЕЕ КАЧЕСТВО OFFLINE

- **Repo**: https://github.com/antgroup/echomimic_v3
- **Подход**: Diffusion-based, multi-modal (аудио + текст)
- **VRAM**: 12 GB (Flash mode, 8-step generation)
- **Плюсы**:
  - Разрешение до 768×768
  - Apache 2.0 — коммерчески свободен
  - AAAI 2026, активная разработка
- **Минусы**:
  - Не real-time (~50s на 120 кадров на A100, ~2.4 FPS)
- **Вердикт**: Лучший для offline-генерации контента на 16GB

---

## Другие заметные проекты

### Linly-Talker — Готовая end-to-end система
- **Repo**: https://github.com/Kedreamix/Linly-Talker
- Интегрирует LLM + TTS + аватар (MuseTalk/SadTalker backend)
- WebRTC стриминг
- Полезен как reference-архитектура

### FlowTalk — Новое поколение
- Flow-matching подход, заявляется 100+ FPS в motion-space
- Очень ранняя стадия

---

## Принятое решение

### Этап 1: Прототип на MuseTalk 1.5
- Real-time lip-sync из фото + аудио
- 4-6 GB VRAM — много запаса для других компонентов
- Быстрый старт — работает "из коробки" с аудио

### Этап 2 (опционально): Миграция на LivePortrait
- Если нужна коммерческая лицензия
- Если нужно лучшее качество картинки
- Потребует разработки audio → landmarks pipeline

---

## Архитектура решения

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  TTS Engine │────▶│  MuseTalk    │────▶│  Web Client  │
│  (голос)    │     │  (lip-sync   │     │  (video      │
│             │     │   генерация) │     │   stream)    │
└─────────────┘     └──────────────┘     └──────────────┘
       ▲                    ▲
       │                    │
   Текст ответа        Фото аватара
```

### Компоненты:
1. **Вход**: фото человека + аудио (от TTS или микрофон)
2. **MuseTalk**: генерирует видеокадры с lip-sync в реальном времени
3. **Стриминг**: кадры → WebSocket/WebRTC → браузер

---

## Ссылки

- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- [LivePortrait GitHub](https://github.com/KlingTeam/LivePortrait)
- [EchoMimicV3 GitHub](https://github.com/antgroup/echomimic_v3)
- [SadTalker GitHub](https://github.com/OpenTalker/SadTalker)
- [Wav2Lip GitHub](https://github.com/Rudrabha/Wav2Lip)
- [Hallo2 GitHub](https://github.com/fudan-generative-vision/hallo2)
- [LatentSync GitHub](https://github.com/bytedance/LatentSync)
- [Linly-Talker GitHub](https://github.com/Kedreamix/Linly-Talker)
