
def read_symbol_txt(filename):
    class_index = 0
    class_name_to_index_dict = {}
    class_index_to_name_dict = {}

    with open(filename, 'r') as f:
        for l in f.readlines():
            _, class_name = l.rstrip().split("|")
            class_name_to_index_dict[class_name] = class_index
            class_index_to_name_dict[class_index] = class_name

            class_index += 1
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

def symbol_simple_dump(filename, symbol_dict):
    f = open(filename, "w")
    f.write("(\n")
    for key, val in symbol_dict.items():
        f.write(f"\"{key}\",\n")
    f.write(")\n")
    f.close()