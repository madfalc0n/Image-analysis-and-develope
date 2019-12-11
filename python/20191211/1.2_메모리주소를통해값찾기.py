import ctypes
value = 10
print(id(value))

print(ctypes.cast(value, ctypes.py_object).value)
