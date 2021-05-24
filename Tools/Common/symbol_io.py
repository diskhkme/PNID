
def read_symbol_type_txt(filename):
    class_index = 0
    class_name_to_type_dict = {}

    with open(filename, 'r') as f:
        for l in f.readlines():
            strs = l.rstrip().split("|")
            class_name_to_type_dict[strs[1]] = strs[0]

    return class_name_to_type_dict

def read_symbol_txt(filename, include_text_as_class, include_text_orientation_as_class):
    class_index = 0
    class_name_to_index_dict = {}
    class_index_to_name_dict = {}

    with open(filename, 'r') as f:
        for l in f.readlines():
            _, class_name = l.rstrip().split("|")
            class_name_to_index_dict[class_name] = class_index
            class_index_to_name_dict[class_index] = class_name

            class_index += 1

    if include_text_as_class == True:
        class_name_to_index_dict["text"] = len(class_name_to_index_dict.items())

    if include_text_orientation_as_class == True:
        class_name_to_index_dict["text_rotated"] = len(class_name_to_index_dict.items())
        class_name_to_index_dict["text_rotated_45"] = len(class_name_to_index_dict.items())

    return class_name_to_index_dict

def read_symbol_pbtxt(filename, start_id = 0, merge=True):
    symbol_dict = {}
    source_symbol_dict = {}
    id = start_id
    f = open(filename, "r")

    while True:
        line = f.readline()
        if line.find("name") > 0:
            start = line.find("\"")
            end = line.rfind("\"")
            if merge == True:
                name = line[start+1:end].split("-")[0]
                if name not in symbol_dict.keys():
                    symbol_dict[name] = id
                    id += 1
                source_symbol_dict[line[start + 1:end]] = id-1
            else:
                name = line[start+1:end]
                symbol_dict[name] = id
                id += 1

        if not line: break

    f.close()

    return symbol_dict, source_symbol_dict

def symbol_simple_dump_for_mmdetection(filename, symbol_dict):
    f = open(filename, "w")
    f.write("(\n")
    for key, val in symbol_dict.items():
        f.write(f"\"{key}\",\n")
    f.write(")\n")
    f.close()

def symbol_simple_dump_to_txt(filename, symbol_dict):
    f = open(filename, "w")
    id = 0
    for key, val in symbol_dict.items():
        f.write(f"{id}|{key}\n")
        id += 1
    f.close()


if __name__ == '__main__':
    symbol_filepath = "D:/Test_Models/PNID/HyundaiEng/210520_Data/Hyundai_SymbolClass_Sym_Only.txt"  # (방향 제거된) symbol index txt 파일 경로
    symbol_type_filepath = "D:/Test_Models/PNID/HyundaiEng/210520_Data/Hyundai_SymbolClass_Type.txt"  # 심볼이름-타입 매칭 txt

    symbol_dict = read_symbol_txt(symbol_filepath, False, False)
    symbol_type_dict = read_symbol_type_txt(symbol_type_filepath)

    for k,v in symbol_dict.items():
        if k not in symbol_type_dict:
            print(k)