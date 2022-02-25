







with open('stylegan2ada.log', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if '/data/Workdir' in line:
            continue
        if 'warnings.warn' in line:
            continue
        print(line)






