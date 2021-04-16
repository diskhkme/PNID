
def read_symbol_txt(filename, start_id = 1, merge=False):
    symbol_dict = {}
    id = start_id
    f = open(filename, "r")

    # TODO : merge 구현

    while True:
        line = f.readline()
        if line.find("name") > 0:
            start = line.find("\"")
            end = line.rfind("\"")
            name = line[start+1:end]
            symbol_dict[name] = id
            id = id+1
        if not line: break

    f.close()

    return symbol_dict

def symbol_simple_dump(filename, symbol_dict):
    f = open(filename, "w")
    f.write("(\n")
    for key, val in symbol_dict.items():
        f.write(f"\"{key}\",\n")
    f.write(")\n")
    f.close()