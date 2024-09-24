import os
import builtins
import sys
import io
from datetime import datetime

from loguru import logger 

now = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

'''
experiment_dir = os.path.join("experiments")
log_dir = os.path.join(experiment_dir, "logs")

if not os.path.exists(log_dir):
    print(f"{log_dir} does not exist")
    os.makedirs(log_dir, exist_ok = True)
#assert os.path.exists(log_dir), f"{log_dir} does not exist"
command = "_".join(sys.argv).replace("/","_").replace(".py","").replace(" ","_").replace("-","")
#logger.add(os.path.join(log_dir,f"{now}_{command}.log"))
'''

def print(*args, **kwargs):
    logger.opt(colors=True).info(print_to_string(*args, **kwargs)) # loguru color markup

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    builtins.print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


if __name__ == "__main__":
    print("1")
    print("2")
    print("3")
    print("4")