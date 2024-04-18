import pandas as pd

def tab(input, indent=4, color=None):
    color_dict = {'red':'\033[91m', 'green':'\033[92m', 'yellow':'\033[93m', 'blue':'\033[94m', 'pink':'\033[95m', 'teal':'\033[96m', 'grey':'\033[97m'}
    ret = ''
    if type(input)==str:
        ret = ' '*indent +input.replace('\n','\n'+' '*indent)
    elif (type(input) == pd.core.frame.DataFrame) or (type(input) == pd.core.frame.Series):
        ret = " "*indent + input.to_string().replace("\n", "\n"+" "*indent)
    elif type(input) in [str, float, list, dict, int]:
        ret = ' '*indent +str(input).replace('\n','\n'+' '*indent)
        # ret = ' '*indent + str(input)
    if (color != None):
        ret = color_dict[color]+ret
    return ret+"\033[0m"