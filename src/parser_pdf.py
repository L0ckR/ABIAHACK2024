import fitz  # PyMuPDF
import json
import os
import numpy as np
import re
from collections import OrderedDict
from PIL import Image
from io import BytesIO
import imagehash
from pyzbar.pyzbar import decode
import cv2

# Параметры фильтрации изображений
MIN_WIDTH = 100      
MIN_HEIGHT = 100     # Минимальная высота изображения в пикселях
MAX_WIDTH = 3000     # Максимальная ширина изображения в пикселях
MAX_HEIGHT = 3000    # Максимальная высота изображения в пикселях
ASPECT_RATIO_MIN = 0.15  # Минимальное соотношение сторон
ASPECT_RATIO_MAX = 5.0  # Максимальное соотношение сторон

# Порог схожести для хешей (чем меньше, тем строже)
HASH_THRESHOLD = 5

def is_heading(block, min_fontsize=14):
    if 'lines' not in block:
        return False
    # Определяем максимальный размер шрифта в блоке
    max_size = max(span['size'] for line in block['lines'] for span in line['spans'])
    # Проверяем, есть ли жирные шрифты и размер шрифта превышает порог
    bold = any(span['flags'] & 2 for line in block['lines'] for span in line['spans'])
    return max_size >= min_fontsize and bold

def extract_tables(block):
    """
    Извлекает таблицы из блока и конвертирует их в формат Markdown.
    """
    if 'lines' not in block:
        return ""
    table = []
    for line in block['lines']:
        row = []
        for span in line['spans']:
            # Предполагаем, что ячейки таблицы разделены табуляцией или несколькими пробелами
            cells = re.split(r'\t+|\s{2,}', span['text'])
            row.extend(cells)
        # Убираем пустые ячейки
        row = [cell.strip() for cell in row if cell.strip()]
        if row:
            table.append(row)
    if not table:
        return ""
    # Конвертация в Markdown
    markdown_table = "| " + " | ".join(table[0]) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
    for row in table[1:]:
        markdown_table += "| " + " | ".join(row) + " |\n"
    return markdown_table.strip()

def is_valid_image(image):
    """
    Проверяет, соответствует ли изображение заданным критериям размера и соотношения сторон.
    """
    width, height = image.size
    aspect_ratio = width / height if height != 0 else 0
    if (MIN_WIDTH <= width <= MAX_WIDTH and
        MIN_HEIGHT <= height <= MAX_HEIGHT and
        ASPECT_RATIO_MIN <= aspect_ratio <= ASPECT_RATIO_MAX):
        return True
    return False

def contains_qr_code(image):
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    decoded_objects = decode(open_cv_image)
    return len(decoded_objects) > 0

def pdf_to_json(pdf_path, output_dir, heading_fontsize_threshold=14):
    """
    Извлекает текст, изображения и таблицы из PDF и структурирует их в JSON формате.
    Args:
        pdf_path (str): Путь к PDF файлу.
        output_dir (str): Путь к выходной директории для изображений и JSON файла.
        heading_fontsize_threshold (int): Минимальный размер шрифта для определения заголовка.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    doc = fitz.open(pdf_path)
    
    all_sections = []
    section_counter = 1  # Начинаем с 1 для лучшей читаемости

    # Используем set для хранения уникальных хешей изображений
    image_hashes = set()

    current_section = {
        "filename": os.path.basename(pdf_path).replace(".pdf", ""),
        "section_number": str(section_counter),
        "text": "",
        "image_paths": []
    }

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        images = page.get_images(full=True)
        
        # Извлечение и сохранение изображений
        image_paths = []
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                print(f"Ошибка при открытии изображения на странице {page_num+1}: {e}")
                continue

            # Проверка размеров изображения
            if not is_valid_image(image):
                print(f"Изображение на странице {page_num+1}, индекс {img_index+1} отклонено по размеру ({image.size})")
                continue

            # Проверка на наличие QR-кода
            if contains_qr_code(image):
                print(f"Изображение на странице {page_num+1}, индекс {img_index+1} содержит QR-код и было пропущено.")
                continue

            # Вычисление перцептуального хеша
            try:
                img_hash = imagehash.phash(image)
            except Exception as e:
                print(f"Ошибка при вычислении хеша изображения на странице {page_num+1}, индекс {img_index+1}: {e}")
                continue

            # Проверка на дубликаты
            if img_hash in image_hashes:
                print(f"Дубликат изображения на странице {page_num+1}, индекс {img_index+1} обнаружен и был пропущен.")
                continue
            
            # Добавление хеша в набор
            image_hashes.add(img_hash)

            # Сохранение изображения
            img_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
            section_images_dir = os.path.join(images_dir, f"page_{page_num+1}")
            if not os.path.exists(section_images_dir):
                os.makedirs(section_images_dir)
            
            image_path = os.path.join(section_images_dir, img_name)
            try:
                image.save(image_path)
                image_paths.append(image_path)
                print(f"Сохранено изображение: {image_path}")
            except Exception as e:
                print(f"Ошибка при сохранении изображения {image_path}: {e}")
        
        # Ассоциация изображений с текущей секцией
        current_section["image_paths"].extend(image_paths)

        # Обработка текстовых и табличных блоков
        for block in blocks:
            if block['type'] == 0:  # Текстовый блок
                if is_heading(block, heading_fontsize_threshold):
                    # Если текущая секция не пуста, добавляем её в all_sections
                    if current_section["text"].strip():
                        all_sections.append(OrderedDict(current_section))
                        section_counter += 1
                        current_section = {
                            "filename": os.path.basename(pdf_path).replace(".pdf", ""),
                            "section_number": str(section_counter),
                            "text": "",
                            "image_paths": []
                        }
                    # Добавляем заголовок в новую секцию
                    heading_text = " ".join([span['text'] for line in block['lines'] for span in line['spans']]).strip()
                    current_section["text"] += heading_text + "\n\n"
                else:
                    # Обычный текст
                    text = " ".join([span['text'] for line in block['lines'] for span in line['spans']]).strip()
                    # Разделяем текст на абзацы по двойным переносам строк или точкам
                    paragraphs = re.split(r'\n{2,}|\.\s', text)
                    for para in paragraphs:
                        para = para.strip()
                        if para:
                            current_section["text"] += para + "\n\n"
            elif block['type'] == 2:  # Блок таблицы
                markdown_table = extract_tables(block)
                if markdown_table:
                    current_section["text"] += markdown_table + "\n\n"

        # Ограничение на максимальный размер текста в секции (например, 1000 символов)
        if len(current_section["text"]) > 1000:
            all_sections.append(OrderedDict(current_section))
            section_counter += 1
            current_section = {
                "filename": os.path.basename(pdf_path).replace(".pdf", ""),
                "section_number": str(section_counter),
                "text": "",
                "image_paths": []
            }
            current_section["image_paths"].extend(image_paths)

    # Добавляем последнюю секцию
    if current_section["text"].strip():
        all_sections.append(OrderedDict(current_section))
    
    # Сохранение в JSON
    json_file_path = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}.json")
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(all_sections, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import numpy as np  

    pdf_file = "C:/Users/User/Desktop/mena/avia_hack/aviahack/input_docs/Руководство_по_разбрасователю_Амазон_ZG_TS_7501.pdf"  
    output_directory = "aviahack/support/output4_amazon_ZG_TS_7501"
    pdf_to_json(pdf_file, output_directory)
    print(f"JSON и изображения сохранены в {output_directory}")
