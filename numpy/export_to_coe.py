import os

def txt_to_coe(txt_path, coe_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    with open(coe_path, 'w') as f:
        # Vivado BRAM 초기화 포맷 헤더
        f.write("memory_initialization_radix=10;\n")
        f.write("memory_initialization_vector=\n")
        
        for i, line in enumerate(lines):
            val = line.strip()
            # 마지막 줄은 세미콜론(;), 나머지는 콤마(,)
            if i == len(lines) - 1:
                f.write(f"{val};\n")
            else:
                f.write(f"{val},\n")

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(BASE_DIR, "weights_int8_txt")
    output_dir = os.path.join(BASE_DIR, "weights_coe")
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    print("Vivado용 .coe 파일 변환 시작...")
    for f in files:
        txt_path = os.path.join(input_dir, f)
        coe_name = f.replace(".txt", ".coe")
        coe_path = os.path.join(output_dir, coe_name)
        
        txt_to_coe(txt_path, coe_path)
        print(f"변환 완료: {coe_name}")
        
    print("\n[성공] weights_coe 폴더에 모든 .coe 파일이 생성되었습니다!")

if __name__ == "__main__":
    main()