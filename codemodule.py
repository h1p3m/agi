# here neural network avatar bot must test code answers
# add code recognition

#catch error and send answer for new generation 


import subprocess
import os
import time
import signal
import sys
import traceback
import threading
import queue


# Функция для выполнения команды с обработкой ошибок
def execute_command(command, timeout=10):
    try:
        # Запуск команды с таймаутом
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Command timed out"
    except Exception as e:
        return "", str(e)

# Функция для создания файлов с ограничением на время выполнения
def create_file_with_timeout(filename, content, timeout=5):
    def handler(signum, frame):
        raise TimeoutError("File creation timed out")
    
    # Установка обработчика сигнала для ограничения времени выполнения
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    
    try:
        with open(filename, 'w') as file:
            file.write(content)
        signal.alarm(0)  # Отключение сигнала
        return f"File {filename} created successfully"
    except TimeoutError as e:
        return str(e)
    except Exception as e:
        return str(e)
    finally:
        signal.alarm(0)  # Отключение сигнала

# Логирование ошибок в файл
def log_error(error_message):
    with open("error_log.txt", "a") as error_file:
        error_file.write(f"{error_message}\n")

# Функция для выполнения Python кода в изолированном окружении
def execute_python_code(code, timeout=10):
    local_vars = {}
    error_queue = queue.Queue()
    
    def exec_code():
        try:
            exec(code, {"__builtins__": __builtins__}, local_vars)
        except Exception as e:
            error_queue.put(traceback.format_exc())
    
    thread = threading.Thread(target=exec_code)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return "Execution timed out", True
    else:
        if not error_queue.empty():
            return error_queue.get(), False
        return "Execution completed successfully", False

# Функция для непрерывного выполнения задач
def interactive_loop():
    while True:
        task = input("Enter Python code to execute or 'exit' to quit: ").strip()
        
        if task.lower() in ["exit", "quit"]:
            print("Exiting the interactive loop.")
            break
        
        result, is_timeout = execute_python_code(task)
        
        if is_timeout:
            log_error("TimeoutError: Code execution exceeded the time limit.")
        elif "Exception" in result or "Error" in result:
            log_error(result)
        
        print(result)
        
        if is_timeout or ("Exception" in result or "Error" in result):
            print("Retrying execution...")
            result, is_timeout = execute_python_code(task)
            
            if is_timeout:
                log_error("TimeoutError: Code execution exceeded the time limit on retry.")
            elif "Exception" in result or "Error" in result:
                log_error(result)
            
            print(result)

# Запуск интерактивного цикла
if __name__ == "__main__":
    interactive_loop()
