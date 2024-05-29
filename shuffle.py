import json
import random

def shuffle_jsonl(input_file, output_file=None):
    # Чтение всех строк из исходного .jsonl файла
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Перетасовка строк
    random.shuffle(lines)

    # Определение выходного файла
    if output_file is None:
        output_file = input_file

    # Запись перетасованных строк в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Пример использования
#input_file = 'data.jsonl'
shuffle_jsonl(r'C:\multim\fixed_data3.jsonl', 'fixed_data3.jsonl')
