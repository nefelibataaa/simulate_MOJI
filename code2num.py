import os
import csv

# 读取指定目录下的所有文件并返回文件内容列表
def read_files_from_directory(directory_path):
    file_contents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
    return file_contents

# 将字符串转化为数值表示
def string_to_numeric_representation(input_string):
    return [ord(char) for char in input_string]

# 读取目录下的文件，转化为数值表示，并打上标签
def process_directory(directory_path, label):
    file_contents = read_files_from_directory(directory_path)
    labeled_data = [(string_to_numeric_representation(content), label) for content in file_contents]
    return labeled_data

# 将数据保存到CSV文件
def save_to_csv(data, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NumericalRepresentation', 'Label'])
        for numeric_rep, label in data:
            writer.writerow([numeric_rep, label])

# 主函数，处理良性和恶意代码目录并输出结果到文件
def main(benign_path, malicious_path, output_path):
    benign_data = process_directory(benign_path, 0)
    malicious_data = process_directory(malicious_path, 1)
    all_data = benign_data + malicious_data
    save_to_csv(all_data, output_path)
    print(f"处理结果已保存到 {output_path}")

# 替换为你的数据集路径
benign_code_path = "E:/python_project/Dataset/test_py_benign"
malicious_code_path = "E:/python_project/Dataset/test_py_malicious"
output_file_path = "E:/python_project/MOJI/pycode_test.csv"

main(benign_code_path, malicious_code_path, output_file_path)