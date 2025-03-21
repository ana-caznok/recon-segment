from typing import List, Any


class Compose():
    def __init__(self, transform_list: List[Any]):
        self.transform_list = transform_list

    def __call__(self, x, y, m):
        for transform in self.transform_list:
            x, y, m = transform(x, y, m)

        return x, y, m
    
    def __str__(self) -> str:
        buffer = ''
        for i, transform in enumerate(self.transform_list):
            buffer += f"\n{i}: {str(transform)}"
        
        return buffer
