from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
import os
import aiohttp
import cv2
import numpy as np
from ultralytics import YOLO
import io
from telegram.request import HTTPXRequest
import asyncio
import tempfile

# Загрузка переменных окружения из .env
load_dotenv()

# Загрузка токена бота
TOKEN = os.environ.get("TOKEN")

# Инициализация моделей YOLOv8
models = {
    'corrosion': YOLO('yolov8s_11_KM.pt'),
    'graffiti': YOLO('yolov8s_12_NG.pt'),
    'defects of nodes': YOLO('yolov8s_16_NUOD.pt')
}

# Определение цветов для разных классов
class_colors = {
    'corrosion': (0, 0, 255),  # Красный
    'graffiti': (0, 255, 0),   # Зеленый
    'defects of nodes': (255, 0, 0)      # Синий
}

# Функция команды /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('Привет! Отправь фото или видео моста, и я проверю её на наличие дефектов.')

# Функция команды /help
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text(
        'Этот бот может обнаружить коррозию металла, граффити и неисправности узлов на фото и видео мостов. Просто отправь фотографию или видео моста.')

# Параллельная детекция с использованием всех моделей
async def ensemble_predict(image):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, model.predict, image, 0.4) for model in models.values()]
    results = await asyncio.gather(*tasks)
    return results

# Обработчик изображений
async def handle_image(update: Update, context: CallbackContext):
    await update.message.reply_text('Получили фотографию, обрабатываем...')

    try:
        # Получение файла изображения
        file = await update.message.photo[-1].get_file()
        file_url = file.file_path

        # Загрузка изображения
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                image_bytes = await response.read()

        # Открытие изображения с помощью OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Детекция с использованием всех моделей
        detections = await ensemble_predict(image)

        # Отслеживание обнаруженных дефектов
        detected_defects = set()

        # Объединение всех предсказаний
        for model_name, det in zip(models.keys(), detections):
            for result in det:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model_name  # Использование названия модели как метки
                    detected_defects.add(label)  # Добавление обнаруженного дефекта в набор
                    color = class_colors[model_name]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Сохранение результатов в буфер
        output_buffer = io.BytesIO()
        result_image_path = 'result_image.png'
        cv2.imwrite(result_image_path, image)  # Сохранение изображения с аннотациями

        # Считываем сохраненное изображение
        with open(result_image_path, 'rb') as f:
            output_buffer.write(f.read())

        # Сброс позиции указателя в начало
        output_buffer.seek(0)

        # Отправка изображения обратно пользователю
        await update.message.reply_photo(photo=output_buffer)

        # Формирование и отправка текстового мини-отчета
        if detected_defects:
            report = "Обнаружены следующие дефекты:\n" + "\n".join(f"- {defect}" for defect in detected_defects)
        else:
            report = "Дефекты не обнаружены."
        await update.message.reply_text(report)
    except Exception as e:
        await update.message.reply_text(f'Произошла ошибка при обработке изображения: {e}')

# Обработчик видео
async def handle_video(update: Update, context: CallbackContext):
    await update.message.reply_text('Получили видео, обрабатываем...')

    try:
        # Получение файла видео
        video_file = update.message.video
        if video_file.file_size > 50 * 1024 * 1024:
            await update.message.reply_text('Видео слишком большое. Пожалуйста, отправьте файл размером не более 50 МБ.')
            return

        file = await video_file.get_file()
        file_url = file.file_path

        # Загрузка видео
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as response:
                video_bytes = await response.read()

        # Использование временного файла для хранения видео
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            video_path = temp_video.name
            temp_video.write(video_bytes)

        # Открытие видео с помощью OpenCV
        cap = cv2.VideoCapture(video_path)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output_video:
            output_video_path = temp_output_video.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # Отслеживание обнаруженных дефектов
            detected_defects = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Детекция с использованием всех моделей
                detections = await ensemble_predict(frame)

                # Объединение всех предсказаний
                for model_name, det in zip(models.keys(), detections):
                    for result in det:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = model_name  # Использование названия модели как метки
                            detected_defects.add(label)  # Добавление обнаруженного дефекта в набор
                            color = class_colors[model_name]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Запись аннотированного кадра в выходное видео
                out.write(frame)

            cap.release()
            out.release()

        # Отправка видео обратно пользователю
        with open(output_video_path, 'rb') as video_file:
            await update.message.reply_video(video=video_file)

        # Удаление временных файлов
        os.remove(video_path)
        os.remove(output_video_path)

        # Формирование и отправка текстового мини-отчета
        if detected_defects:
            report = "Обнаружены следующие дефекты:\n" + "\n".join(f"- {defect}" for defect in detected_defects)
        else:
            report = "Дефекты не обнаружены."
        await update.message.reply_text(report)

    except Exception as e:
        await update.message.reply_text(f'Произошла ошибка при обработке видео: {e}')

def main():
    # Установка параметров тайм-аута для HTTP-запросов
    request = HTTPXRequest(read_timeout=60, write_timeout=60, connect_timeout=60, pool_timeout=60)

    # Точка входа в приложение
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # Добавление обработчиков команд
    application.add_handler(CommandHandler("start", start, block=False))
    application.add_handler(CommandHandler("help", help_command, block=False))

    # Добавление обработчика изображений
    application.add_handler(MessageHandler(filters.PHOTO, handle_image, block=False))
    # Добавление обработчика видео
    application.add_handler(MessageHandler(filters.VIDEO, handle_video, block=False))

    # Запуск приложения (для остановки нужно нажать Ctrl-C)
    application.run_polling()

if __name__ == "__main__":
    main()
