import os
import sys
import fileinput
import spacy
import random
import re
nlp = spacy.load('en_core_web_sm')
nlp = list(nlp.vocab.strings)
nlp = nlp[5620:83670]
nlp = sorted(nlp)


def functn(s):
    if s in ["main", "printf", "scanf", "write", "writeln", "fclose", "fgets", "fopen", "fprintf", "fputc", "fscanf", "gets", "pclose", "popen", "putw", "setbuf", "sprintf", "sscanf", "system", "ungetc", "srand", "rand", "calloc", "malloc", "realloc", "free", "strcat", "strncat", "strcpy", "strncpy", "strlen", "strcmp", "strcmpi", "strchr", "strrchr", "strstr", "strrstr", "strdup", "strlwr", "strupr", "strrev", "strset", "strnset", "strtok", "atoi", "assert", "ferror", "fflush", "clearerr", "feof", "fseek", "ftell", "remove", "rename", "rewind", "puts", "putchar", "putc", "atof", "atol", "abort", "exit", "system",]:
        return True
    return False


def Preprocessor(s):

    if s[0] == '#' and len(s) > 1:
        return True
    return False


def isDelimiter(ch):

    if ch == ' ' or ch == '+' or ch == '-' or ch == '*' or ch == '/' or ch == ',' or ch == ';' or ch == '>' or ch == '<' or ch == '=' or ch == '(' or ch == ')' or ch == '[' or ch == ']' or ch == '{' or ch == '}':
        return True
    return False


def isOperator(ch):

    if ch == '+' or ch == '-' or ch == '*' or ch == '/' or ch == '>' or ch == '<' or ch == '=':
        return True
    return False


def validIdentifier(s):
    sp = re.compile('[,]')
    i = len(s)
    if s[0].isalpha() or s[0] == '_' and not isKeyword(s) and (len(s.split(" ")) == 1):
        for w in range(1, i):
            if not (s[w].isalpha() or s[w].isnumeric() or s[w] == '_'):
                return False
        return True
    else:
        return False


def isKeyword(s):
    if s in ["if", "else", "while", "do", "break", "size_t", "fpos_t", "FILE", "continue", "int", "double", "extern", "float", "return", "char", "case", "const", "default", " enum", "sizeof", "long", "short", "typedef", "switch", "auto", "for", "unsigned", "void", "static", "struct", "goto", "register", "union", "typedef", "volatile"]:
        return True
    return False


def isInteger(s):
    # print(s[2])
    # print(len(s))
    i = 0
    l = len(s)

    if l == 0:
        return False

    for i in range(l):

        if s[i] != '0' and s[i] != '1' and s[i] != '2' and s[i] != '3' and s[i] != '4' and s[i] != '5' and s[i] != '6' and s[i] != '7' and s[i] != '8' and s[i] != '9' or (s[i] == '-' and i > 0):
            return False
    return True


def isRealNumber(s):
    i = 0
    l = len(s)
    hasDecimal = False

    if l == 0:
        return False
    for i in range(l):
        if s[i] != '0' and s[i] != '1' and s[i] != '2' and s[i] != '3' and s[i] != '4' and s[i] != '5' and s[i] != '6' and s[i] != '7' and s[i] != '8' and s[i] != '9' and s[i] != '.' or (s[i] == '-' and i > 0):
            return False
        if s[i] == '.':
            hasDecimal = True
    return hasDecimal


def subString(s, l, r):
    i = 0
    subStr = ' '*(r - l + 2)

    for i in range(l, r+1):
        subStr[i - l] = s[i]

    return subStr


def tag(f, f1):
    i = 0
    for line in f.readlines():
        label = str(line[0])
        line = line.split(" ")[1:]
        nline = line
        i = i+1
        print(f'Line No --- {i}')
        s = []
        for w in line:
            if len(w) == 0:
                continue
            elif Preprocessor(w):
                print(f'{w} IS A PREPROCESSOR OR HEADER FILE')
            elif isOperator(w) and len(w) == 1:
                print(f'{w} IS AN OPERATION')
            elif isKeyword(w) == True:
                print(f'{w} IS A KEYWORD')
            elif (functn(w) == True):
                print(f'{w} IS A FUNCTION')
            elif (isInteger(w) == True):
                print(f'{w} IS AN INTEGER')
            elif (isRealNumber(w) == True):
                print(f'{w} IS A REAL NUMBER')
            elif (validIdentifier(w) == True and isDelimiter(w[-1]) == False):
                print(f'{w} IS A VALID IDENTIFIER')
                if w not in s:
                    s.append(w)
            elif (validIdentifier(w) == False and isDelimiter(w[-1]) == False):
                print(f'{w} IS NOT A VALID IDENTIFIER')
        # print(s)
        nline = " ".join(map(str, nline))
        for w in s:
            w = " "+w+" "
            a = random.choice(nlp)
            while not validIdentifier(a):
                a = random.choice(nlp)
            a = " "+a+" "
            nline = nline.replace(w, a)
        nline = label+" "+nline
        f1.write(nline)
        print(nline)
    f1.close()
    f.close()


if __name__ == "__main__":

    lab_dir = '/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27-scramble'
    for lab in os.listdir(lab_dir):
        lab_path = os.path.join(lab_dir, lab)
        submissions_dir = os.path.join(lab_path, "submissions")
        submissions_data = {}

        # Traverse through each submission directory
        for submission_dir in os.listdir(submissions_dir):
            submission_path = os.path.join(submissions_dir, submission_dir)
            if os.path.isdir(submission_path):
                submission_files = os.listdir(submission_path)
                for file_name in submission_files:
                    # Assuming all submissions are C++ files
                    if file_name.startswith("sc"):
                        continue
                    if file_name.endswith(".cpp"):
                        original_file = file_name
                        scrambled_file = 'sc' + original_file
                        file_path = os.path.join(submission_path, file_name)
                        with open(file_path, 'r') as file:
                            submission_code = file.read()
                        submission_key = submission_dir.split('@')[0]
                        submissions_data[submission_key] = submission_code.strip(
                        )
                        f = open(file_path, 'r')
                        scrambled_path = os.path.join(
                            submission_path, scrambled_file)
                        f1 = open(scrambled_path, 'w+')
                        tag(f, f1)
                        os.remove(file_path)
                        break
