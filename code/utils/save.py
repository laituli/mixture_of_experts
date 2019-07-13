import os

def matrix_strings(matrix, cell_format="%i\t"):
    strs = []
    for i, row in enumerate(matrix):
        for cell in row:
            strs.append(cell_format%cell)
        strs.append("\n")
    return strs

def save_activation(activation,outfile):
    with open(outfile,"w") as file:
        file.write("row indicate super label, column indicate expert\n")
        file.writelines(matrix_strings(activation,cell_format="%.2f\t"))

def save_confusion(confusion, outfile):
    with open(outfile, 'w') as file:
        file.write("overall confusion: (integer count)\n")
        file.writelines(matrix_strings(confusion["overall"]))
        file.write("overall confusion percent: (rows normalized in %)\n")
        file.writelines(matrix_strings(confusion["overall_percent"]))

        for i, confusion_matrix in enumerate(confusion["experts"]):
            file.write("\n" + str(i) + ":th expert's confusion: (integer count)\n")
            file.writelines(matrix_strings(confusion_matrix))


def confusion(results, folder):
    for case, result in results.items():
        with open(os.path.join(folder,"%s confusion.txt"%case), 'w') as file:
            file.write("overall confusion:\n")
            file.writelines(matrix_strings(result["overall confusion"]))
            for i, confusion_matrix in enumerate(result["expert confusion"]):
                file.write("\n" + str(i) + ":th expert's confusion:\n")
                file.writelines(matrix_strings(confusion_matrix))

def activation(results, folder):
    for case, result in results.items():
        with open(os.path.join(folder,"%s activation.txt"%case), 'w') as file:
            file.write("y activation:\n")
            file.writelines(matrix_strings(result["y activation"],cell_format="%f\t"))
            file.write("z activation:\n")
            file.writelines(matrix_strings(result["z activation"],cell_format="%f\t"))

