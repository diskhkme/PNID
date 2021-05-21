from Common.symbol_io import read_symbol_pbtxt, symbol_simple_dump_to_txt

pbtxt_path = 'D:/Test_Models/PNID/HyundaiEng/210520_Data/Symbol Class List.pbtxt'
txtout_path = 'D:/Test_Models/PNID/HyundaiEng/210520_Data/Hyundai_SymbolClass_Sym_Only.txt'

symbol_dict,_ = read_symbol_pbtxt(pbtxt_path,0,False)
symbol_simple_dump_to_txt(txtout_path, symbol_dict)