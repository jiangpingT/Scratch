Python 是一种高级、通用、解释型的编程语言，以其简洁易读的语法和丰富的库而闻名。下面为你介绍 Python 的基础语法：

### 1. 注释
在 Python 里，注释用于解释代码，不会被执行。它分为单行注释和多行注释。
```python
# 这是单行注释

"""
这是多行注释，
可以跨越多行。
"""
```

### 2. 变量与数据类型
#### 变量
变量用于存储数据，在使用前无需声明类型。
```python
# 变量赋值
name = "Alice"
age = 25
```

#### 数据类型
- **数字类型**：包含整数（`int`）、浮点数（`float`）等。
```python
x = 10
y = 3.14
```
- **字符串类型**（`str`）：用单引号或双引号括起来。
```python
message = 'Hello, World!'
```
- **布尔类型**（`bool`）：只有两个值，`True` 和 `False`。
```python
is_student = True
```
- **列表类型**（`list`）：可存储多个元素，元素类型可以不同，使用方括号表示。
```python
numbers = [1, 2, 3, 4, 5]
```
- **元组类型**（`tuple`）：和列表类似，但元素不可修改，使用圆括号表示。
```python
coordinates = (10, 20)
```
- **集合类型**（`set`）：无序且元素唯一，使用花括号表示。
```python
fruits = {'apple', 'banana', 'cherry'}
```
- **字典类型**（`dict`）：存储键 - 值对，使用花括号表示。
```python
person = {'name': 'Alice', 'age': 25}
```

### 3. 控制流语句
#### 条件语句
使用 `if`、`elif` 和 `else` 来实现条件判断。
```python
x = 10
if x > 5:
    print("x 大于 5")
elif x == 5:
    print("x 等于 5")
else:
    print("x 小于 5")
```

#### 循环语句
- **`for` 循环**：用于遍历可迭代对象。
```python
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)
```
- **`while` 循环**：在条件为真时持续执行。
```python
i = 0
while i < 5:
    print(i)
    i = i + 1
```

### 4. 函数
函数是可重复使用的代码块，使用 `def` 关键字定义。
```python
def add_numbers(a, b):
    return a + b

result = add_numbers(3, 5)
print(result)
```

### 5. 类与对象
Python 是面向对象的编程语言，可使用 `class` 关键字定义类。
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"我叫 {self.name}，今年 {self.age} 岁。")

p = Person("Alice", 25)
p.introduce()
```

### 6. 模块与包
- **模块**：是一个包含 Python 代码的文件，可通过 `import` 语句导入。
```python
# 导入 math 模块
import math

# 使用 math 模块的函数
result = math.sqrt(16)
print(result)
```
- **包**：是包含多个模块的目录，目录下必须有 `__init__.py` 文件（Python 3.3 之后不是必需的）。

这些就是 Python 基础语法的主要内容，通过这些知识，你可以构建出各种复杂的 Python 程序。 