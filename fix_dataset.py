import json

def transform_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Преобразование строки JSON в объект
            data = json.loads(line)

            # Объединение столбцов user и input в новый столбец input
            combined_input = f"{data['user']} {data['input']}"
            data['input'] = combined_input
            del data['user']

            # Заменяем output значением full_output, если оно не равно null
            if 'full_output' in data and data['full_output']:
                data['output'] = data['full_output']
            
            # Удаление столбца full_output
            if 'full_output' in data:
                del data['full_output']

            # Запись измененного объекта в новый файл
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# Пример использования
input_file = 'C:\multim\\shuffled_data.jsonl'
output_file = 'fixed_data.jsonl'
transform_jsonl(input_file, output_file)
