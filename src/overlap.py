import json

def create_chunks_with_context(data, context_size=200):
    
    chunks = []
    num_sections = len(data)
    
    for i, item in enumerate(data):
        text = item['text']
        filename = item['filename']
        section_number = item['section_number']
        image_paths = item.get('image_paths', [])
        
        context = ""
        
        if i > 0:
            prev_item = data[i-1]
            prev_text = prev_item['text']
            context += prev_text[-context_size:] if len(prev_text) > context_size else prev_text

        context += text 

        if i < num_sections - 1:
            next_item = data[i+1]
            next_text = next_item['text']
            context += next_text[:context_size] if len(next_text) > context_size else next_text
        
        chunks.append({
            "filename": filename,
            "section_number": section_number,
            "chunk": context,
            "text": text, 
            "image_paths": image_paths
        })
    
    return chunks

def process_json_file(filepath, context_size=100):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: инварид {filepath}")
        return
    
    if not isinstance(data, list):
      print("Error: JSON data must be a list of objects")
      return

    chunked_data = create_chunks_with_context(data, context_size)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunked_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully processed and saved to {filepath}")
    except Exception as e:
       print(f"Error: Could not save file{filepath}, {e}")

if __name__ == '__main__':

    filepath = 'C:/Users/User/Desktop/mena/avia_hack/aviahack/support/output1/MG7147_ZG_TS_2022_ru_1.json' 
    process_json_file(filepath, context_size=200)
    
