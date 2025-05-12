import sys
import re
import os

def replace_symbols_in_file(filename):
    try:
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 进行替换
        # 替换 \[ 和 \] 为 $$
        content = re.sub(r'\\\[|\\\]', r'$$', content)
        # 替换 \( 和 \) 为 $
        content = re.sub(r'\\\(|\\\)', r'$', content)
        
        # 生成新文件名
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_modified{ext}"
        
        # 写入新文件
        with open(new_filename, 'w', encoding='utf-8') as file:
            file.write(content)
            
        print(f"文件已处理完成，新文件生成为：{new_filename}")
        
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到！")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法：python MarkForm.py <文件名>")
    else:
        filename = sys.argv[1]
        replace_symbols_in_file(filename)