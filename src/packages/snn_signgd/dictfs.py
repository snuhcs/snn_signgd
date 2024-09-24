import os
import numpy as np
MAX_DEPTH = 100000

class DictFS:
    def __init__(self):
        self.root = {}
        self.cursor = self.root

    def cd(self, dirpath, verbose = False):
        if verbose: 
            print("Dirpath:", dirpath)

        hierarchy = dirpath.split(os.sep)
        top_directory_name = hierarchy.pop(0)

        self.cursor = self.cursor[top_directory_name]

        if len(hierarchy) != 0:
            self.cd(os.path.join(*hierarchy))

        return self.cursor
    
    def exists(self, filepath):
        hierarchy = filepath.split(os.sep)
        cursor = self.root
        for name in hierarchy:
            if name not in cursor:
                return False
            cursor = cursor[name]
        return True
    
    def makedirs(self, dirpath):
        if not dirpath:
            return self
        
        hierarchy = dirpath.split(os.sep)

        cursor = self.root # Root Directory
        for dirname in hierarchy:
            if dirname not in cursor:
                cursor[dirname] = {}
            cursor = cursor[dirname]

        self.cursor = cursor
        return self.cursor

    def tree(self, depth:int, cursor = None):
        if cursor is None:
            cursor = self.cursor # Tree from cwd

        output = []
        for dirname, subdir in cursor.items():
            if isinstance(subdir, dict): # is directory
                if depth > 1:
                    filepaths = self.tree(depth-1, cursor = subdir)
                    output.extend(list(map(lambda x: os.path.join(dirname, x), filepaths)))
                elif depth == 1:
                    output.append(dirname)
            else: # is file
                output.append(dirname)         

        return output
    
    def traverse(self):
        return self.tree(depth = MAX_DEPTH)
    
    def __setitem__(self, filepath, value):
        cursor = self.cursor

        dirname, filename = os.path.split(filepath)

        self.makedirs(dirname)
        self.cursor[filename] = value

        self.cursor = cursor

    def __getitem__(self, filepath):
        cursor = self.cursor

        dirname, filename = os.path.split(filepath)

        try:
            self.cd(dirname)
            value = self.cursor[filename]
        except KeyError:
            raise KeyError(f"File not found: {filepath}")
        
        self.cursor = cursor
        return value
    
    def print_keys(self, cursor = None, indent=0):
        if cursor is None:
            cursor = self.root
            
        for key, value in cursor.items():
            
            print('  ' * indent + "- " + str(key))

            if isinstance(value, dict): # is directory
                self.print_keys(cursor = value, indent = indent + 1)