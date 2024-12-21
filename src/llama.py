import json
import os
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_image_description(text, image_path, ollama_model="llama3.2-11b"):
    prompt = f'''
    Ты — эксперт по анализу изображений из технической документации в области сельского хозяйства. 
    Твоя задача — предоставить точное и краткое описание изображений, основываясь на предоставленном тексте и изображениях.

    **Входные данные:**

    *   **Текст:** Фрагмент текста из технической документации, описывающий определенный раздел или процесс. 
        Этот текст служит контекстом для понимания назначения изображения.
    *   **Изображения:** Одно изображение, связанных с предоставленным текстом.

    **Задача:**

    1.  **Анализ контекста:** Внимательно изучи текст, чтобы понять, какая информация должна быть представлена на изображении.
    2.  **Приоритизация изображения:**
        2.1 **Игнорируй** неинформативные изображения: однотонные фоны, минималистичные изображения, QR-коды, нечеткие изображения.
        Если изображение неинформативно, не предоставляй никакого описания.
        
    3.  **Описание изображений:**
        3.1  **Краткость:** Описывай только ключевые визуальные элементы, важные для понимания контекста, подробно не описывай общие элементы.
        3.2  **Точность:** Избегай домыслов и описания деталей, которых нет на изображении.
        3.3 **Структура:**
            3.3.1   Начни с общего описания изображения (тип, общая композиция).
            3.3.2   Затем опиши конкретные элементы, которые имеют значение, основываясь на контексте.
            3.3.3   Описывай элементы в логичном порядке, например, сверху вниз или слева направо.
        3.4 **Техническая направленность:** Сосредоточься на описании технических элементов, таких как:
            3.4.1  Элементы интерфейса (кнопки, экраны, меню).
            3.4.2   Компоненты оборудования (части тракторов, прицепных устройств).
            3.4.3   Схемы и диаграммы (если они есть).

    4. Не описывай общие фотографии, такие как фото тракторов, изображение на фоне и так далее. Оставляй пустое поле!

    **Формат вывода:**

    *   Если изображение информативно, предоставь краткое и точное описание, следуя структуре, описанной выше. БЕЗ ПОВТОРОВ!
    *   Если изображение неинформативно, не описывай его.
    *   Не дублируй информацию из текста, а только описывай то, что видишь на изображении.
    *   ОТВЕЧАЙ НА РУССКОМ ЯЗЫКЕ! 
    *   НЕ ПРИДУМЫВАЙ лишние деталей, описывай только то, что есть на картинке.

    ## **Пример:**

    **Вход:**
    ``` 
    {{
        "text": "Терминал ISOBUS CCI 100200 Обслуживание 21 5.3 Настройка терминала 5.3.1 Главное меню Откройте главное меню...",
        "image_path": "img_3.jpeg"
    }}
    ```

    **Выход**
    * Случай 1: Если изображение информативно
    Изображение показывает главное меню терминала ISOBUS с иконками приложений и машин.

    *Случай 2: Изображение неинформативно
    (пустая строка, модель ничего не выводит)
    '''

    try:
        command = [
            "ollama", "run", ollama_model,
            prompt,
            "-i", image_path
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=100)

        if stderr:
            print(f"Ошибка Ollama: {stderr}")
            return None

        output = stdout.strip()
        
        if output:
             return output
        else:
            return None
    except subprocess.TimeoutExpired:
        print(f"Превышено время ожидания для изображения: {image_path}")
        return None
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")
        return None
def get_embeddings(texts, model):
    return model.encode(texts)


def remove_duplicate_descriptions(data, threshold=0.9):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    all_descriptions = []
    for item in data:
        if "image_descriptions" in item:
            for description in item["image_descriptions"]:
                if description: # Игнорируем пустые строки
                    all_descriptions.append(description)

    if not all_descriptions:
        return data
    
    embeddings = get_embeddings(all_descriptions, model)
    
    unique_data = []
    added_indices = set()
    
    for i, item in enumerate(data):
        if "image_descriptions" not in item:
            unique_data.append(item)
            continue
        
        new_descriptions = []
        for j, description in enumerate(item["image_descriptions"]):
            if not description: 
                new_descriptions.append(description)
                continue
            
            if j in added_indices:
                continue
            
            is_duplicate = False
            for k in range(j + 1, len(all_descriptions)):
                if k in added_indices or not all_descriptions[k]: 
                    continue
                similarity = cosine_similarity([embeddings[j]], [embeddings[k]])[0][0]
                if similarity > threshold:
                    is_duplicate = True
                    added_indices.add(k)
            
            if not is_duplicate:
                new_descriptions.append(description)
                added_indices.add(j)
        
        if new_descriptions:
            item["image_descriptions"] = new_descriptions
            unique_data.append(item)
    
    return unique_data

def process_json_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                print(f"Обрабатываю файл: {json_path}")

                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Ошибка декодирования JSON в файле: {json_path}")
                        continue

                for item in data:
                    text = item.get("text", "")
                    image_paths = item.get("image_paths", [])
                    
                    if not image_paths:
                        continue

                    image_descriptions = []
                    for image_path in image_paths:
                        full_image_path = os.path.join(root_dir, image_path)
                        
                        if not os.path.exists(full_image_path):
                            print(f"Изображение не найдено: {full_image_path}")
                            image_descriptions.append(f"Изображение не найдено: {image_path}")
                            continue

                        description = generate_image_description(text, full_image_path)
                        if description:
                            image_descriptions.append(description)
                        else:
                            image_descriptions.append(f"Описание не сгенерировано: {image_path}")
                        
                    item["image_descriptions"] = image_descriptions
                    
                    # Удаляем старое поле с путями к изображениям
                    if "image_paths" in item:
                        del item["image_paths"]
                
                # Удаляем дубликаты описаний
                data = remove_duplicate_descriptions(data)

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"Файл {json_path} обработан и обновлен.")

if __name__ == "__main__":
    root_directory = "C:/Users/User/Desktop/mena/avia_hack/aviahack/output_docs"  
    process_json_files(root_directory)