from deepforest import CascadeForestClassifier
import inspect


parameters = inspect.signature(CascadeForestClassifier).parameters
for param_name, param in parameters.items():
    if param.default != inspect.Parameter.empty:
        print(f'{param_name}: {param.default}')