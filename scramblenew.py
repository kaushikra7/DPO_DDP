import os
import random
import re
import subprocess
from collections import OrderedDict
import sys


def scramble_name(name):
    # Simple example: Replace characters with random ones
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(len(name)))


def generate_unique_name(name, var_mapping, fun_mapping):
    new_name = scramble_name(name)
    global counter
    counter = 1
    # print('mapping')
    # print(mapping)
    # print('mapping values')
    # print(mapping.values())
    while new_name in (var_mapping.values() or fun_mapping.values()):
        new_name = f'{new_name}{counter}'
        counter += 1
    return new_name


def scramble_code(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # lines = code_string.split('\n')

    excluded_lines = []
    filtered_lines = []

    for line in lines:
        if line.startswith('#include'):
            excluded_lines.append(line)
        else:
            filtered_lines.append(line)

    filtered_code_string = '\n'.join(filtered_lines)
    excluded_code_string = '\n'.join(excluded_lines)

    # print("Filtered Code:")
    # print(filtered_code_string)
    # print("\nExcluded Code:")
    # print(excluded_code_string)

    # Regular expression patterns to match variable and function names
    variable_pattern = r'\b([a-zA-Z_]\w*)\b(?!\s*\()'
    function_pattern = r'\b([a-zA-Z_]\w*)\('

    reserved_keywords = [
        "int", "char", "float", "double", "void", "bool", "auto", "break", "case", "const", "continue", "default",
        "do", "else", "enum", "extern", "for", "goto", "if", "inline", "mutable", "namespace", "register", "return",
        "sizeof", "static", "struct", "switch", "typedef", "union", "volatile", "while", "main", "printf", "scanf",
        "cin", "cout", "getline", "malloc", "free", "new", "delete", "include", "define", "ifdef", "ifndef", "endif",
        "if", "else", "elif", "pragma", "error", "warning", "undef", "iostream", "cstdio", "cstdlib", "cstring",
        "cmath", "ctime", "cctype", "cassert", "vector", "string", "algorithm", "queue", "stack", "map", "set", 'using', 'endl', "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
        "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr",
        "constinit", "const_cast", "continue", "co_await", "co_return", "co_yield", "decltype", "default", "delete",
        "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float", "for",
        "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
        "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register", "reinterpret_cast",
        "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch",
        "synchronized", "template", "this", "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
        "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "std",
        "abort", "abs", "acos", "aligned_alloc", "asin", "atan", "atan2", "atexit", "atof", "atoi", "atol", "bsearch",
        "calloc", "ceil", "clearerr", "clock", "cos", "cosh", "ctime", "difftime", "div", "exit", "exp", "fabs",
        "fclose", "feof", "ferror", "fflush", "fgetc", "fgetpos", "fgets", "fgetwc", "fgetws", "floor", "fmod", "fopen",
        "fprintf", "fputc", "fputs", "fputwc", "fputws", "fread", "free", "freopen", "frexp", "fscanf", "fseek",
        "fsetpos", "ftell", "fwprintf", "fwrite", "fwscanf", "getc", "getchar", "getenv", "gets", "getwc", "getwchar",
        "gmtime", "isalnum", "isalpha", "isblank", "iscntrl", "isdigit", "isgraph", "islower", "isprint", "ispunct",
        "isspace", "isupper", "iswalnum", "iswalpha", "iswblank", "iswcntrl", "iswctype", "iswdigit", "iswgraph",
        "iswlower", "iswprint", "iswpunct", "iswspace", "iswupper", "iswxdigit", "isxdigit", "labs", "ldexp", "ldiv",
        "localeconv", "localtime", "log", "log10", "longjmp", "malloc", "mblen", "mbstowcs", "mbtowc", "memchr",
        "memcmp", "memcpy", "memmove", "memset", "mktime", "modf", "perror", "pow", "printf", "putc", "putchar",
        "puts", "putwc", "putwchar", "qsort", "raise", "rand", "realloc", "remove", "rename", "rewind", "scanf",
        "setbuf", "setjmp", "setlocale", "setvbuf", "signal", "sin", "sinh", "snprintf", "sprintf", "sqrt", "srand",
        "sscanf", "strcat", "strchr", "strcmp", "strcoll", "strcpy", "strcspn", "strerror", "strftime", "strlen",
        "strncat", "strncmp", "strncpy", "strpbrk", "strrchr", "strspn", "strstr", "strtod", "strtof", "strtok",
        "strtol", "strtold", "strtoll", "strtoul", "strtoull", "strxfrm", "swprintf", "swscanf", "system", "tan", "tanh",
        "time", "tolower", "toupper", "towlower", "towupper", "ungetc", "ungetwc", "vfprintf", "vfwprintf", "vprintf",
        "vsprintf", "vsnprintf", "vswprintf", "vswprintf", "vswscanf", "vwprintf", "vwscanf", "wcrtomb", "wcscat",
        "wcschr", "wcscmp", "wcscoll", "wcscpy", "wcscspn", "wcsftime", "wcslen", "wcsncat", "wcsncmp", "wcsncpy",
        "wcspbrk", "wcsrchr", "wcsspn", "wcsstr", "wcstod", "wcstof", "wcstok", "wcstol", "wcstold", "wcstoll",
        "wcstoul", "wcstoull", "wcsxfrm", "wctob", "wctomb", "wctrans", "wctype", "wmemchr", "wmemcmp", "wmemcpy",
        "wmemmove", "wmemset", "wprintf", "wscanf", "swap", "fixed", "precision", "main_program", "setprecision", "simplecpp", "repeat", "define", "DEFINE", "PI"]

    # Extract user-defined variable and function names from the code
    variables = re.findall(variable_pattern, filtered_code_string)
    # print(variables)
    functions = re.findall(function_pattern, filtered_code_string)
    # print(functions)

    # Exclude reserved keywords from variable and function names
    variables = [name for name in variables if name not in reserved_keywords]
    # print(variables)
    functions = [name for name in functions if name not in reserved_keywords]
    # print(functions)

    # Remove duplicates and sort the names to ensure consistent scrambling
    variables = sorted(set(variables))
    # print(variables)
    functions = sorted(set(functions))
    # print(functions)

    # Create dictionaries to store mapping of old names to new names
    variable_mapping = {}
    function_mapping = {}

    # Scramble variable names
    for var_name in variables:
        new_name = generate_unique_name(
            var_name, variable_mapping, function_mapping)
        variable_mapping[var_name] = new_name
    print('variable mapping')
    print(variable_mapping)
    # Scramble function names
    for func_name in functions:
        new_name = generate_unique_name(
            func_name, variable_mapping, function_mapping)
        function_mapping[func_name] = new_name
    print('function mapping')
    print(function_mapping)
    # Replace variable and function names in the code
    for old_name, new_name in variable_mapping.items():
        filtered_code_string = re.sub(
            rf'\b{old_name}\b', new_name, filtered_code_string)
    for old_name, new_name in function_mapping.items():
        filtered_code_string = re.sub(
            rf'\b{old_name}\b', new_name, filtered_code_string)

    # Write both excluded and filtered lines back to the file
    # print('after scrambled code')
    # print(filtered_code_string)
    # Write scrambled lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(excluded_code_string)
        file.writelines('\n')
        file.writelines(filtered_code_string)


def get_submission_scarmbled(course_code):
    submissions_dir = os.path.join(course_code, "submissions")
    # Traverse through each submission directory
    for submission_dir in os.listdir(submissions_dir):
        submission_path = os.path.join(submissions_dir, submission_dir)
        if os.path.isdir(submission_path):
            submission_files = os.listdir(submission_path)
            for file_name in submission_files:
                # Assuming all submissions are C++ files
                if file_name.endswith(".cpp"):
                    file_path = os.path.join(submission_path, file_name)
                    # print(file_path)
                    scramble_code(file_path)
                    # break  # Assuming there's only one .cpp file per submission

# root_directory = "/Users/kalyani/Desktop/ICER_CODE_AND_DATA/ICER_DATA_scrambled/dataset/CP_00101_loop_cs101f22_LE01_A_Q1"
# get_submission_scarmbled(root_directory)


def main(root_directory):
    # # Assuming root_directory contains course codes
    for course_code in os.listdir(root_directory):
        course_dir = os.path.join(root_directory, course_code)
        if os.path.isdir(course_dir):

            get_submission_scarmbled(course_dir)
            # get_submission_scarmbled(root_directory)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py root_directory")
    #     sys.exit(1)

    # root_directory = sys.argv[1]
    root_directory = "/raid/ganesh/nagakalyani/nagakalyani/autograding/huggingface_codellama/sigcse24labs/dataset27-scramblenew"
    main(root_directory)
