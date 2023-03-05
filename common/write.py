import fileinput
import sys


def replace_global(var_name, var_value):

    dum = 0
    for line in fileinput.input("common/constants.py", inplace=1):

        if var_name + " = " in line and dum == 0:
            dum = 1

            if isinstance(var_value, int):
                line = line.replace(line, "%s = %d \n" % (var_name, var_value))
            elif isinstance(var_value, str):
                line = line.replace(line, "%s = %s \n" % (var_name, var_value))
            elif isinstance(var_value, float):
                line = line.replace(line, "%s = %.3f \n" % (var_name, var_value))

        sys.stdout.write(line)
