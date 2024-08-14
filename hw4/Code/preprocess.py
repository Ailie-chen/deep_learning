import json


if __name__ == "__main__":
    f1 = open("dataset/concode/test_codexglue.json", "r", encoding="UTF-8")
    f2 = open("dataset/concode/test_shuffled.json", "r", encoding="UTF-8")

    codexglue_ls = []
    codexglue_dict = {}
    lines = f1.readlines()
    for line in lines:
        if line == '':
            continue
        item = json.loads(line)
        codexglue_ls.append(item)
        key = item["nl"]
        if key in codexglue_dict:
            codexglue_dict[key].append(len(codexglue_ls) - 1)
        else:
            codexglue_dict[key] = [len(codexglue_ls) - 1]

    fill_count = 0
    for i, line in enumerate(f2.readlines()):
        if line == '':
            continue
        item = json.loads(line)
        nlToks = [tok.lower() for tok in item["nlToks"] if tok not in ["-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-", "@link", "@code", "@inheritDoc"]]
        memberVariables = []
        memberFunctions = []

        for k, v in item["memberVariables"].items():
            sep = "concode_field_sep" if len(memberVariables) == 0 else "concode_elem_sep"
            memberVariables.append(' '.join([sep, v, k.split('=')[0]]))
        for k, v in item["memberFunctions"].items():
            funcs = []
            for vj in v:
                sep = "concode_field_sep" if len(memberFunctions) == 0 and len(funcs) == 0 else "concode_elem_sep"
                funcs += [sep, vj[0], k]
            memberFunctions.append(' '.join(funcs))
        if len(memberVariables) == 0:
            memberVariables = ["concode_field_sep", "PlaceHolder", "placeHolder"]
        if len(memberFunctions) == 0:
            memberFunctions = ["concode_field_sep", "placeholderType", "placeHolder"]

        key = ' '.join(nlToks + memberVariables + memberFunctions).strip()
        if key in codexglue_dict:
            indices = codexglue_dict[key]
            if len(indices) != 0:
                codexglue_ls[indices[0]]["code"] = ' '.join(item["code"])
                codexglue_dict[key].pop(0)
                fill_count += 1


    f1.close()
    f2.close()

    f3 = open("dataset/concode/test.json", "w", encoding="UTF-8")

    for item in codexglue_ls:
        if item["code"] == '':
            print(item["nl"])
        f3.write(json.dumps(item) + '\n')

    f3.close()

    print(fill_count)
