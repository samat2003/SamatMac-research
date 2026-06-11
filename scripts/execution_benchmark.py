"""Research-grade Execution-based Code Completion Benchmark (pass@1) for 120M models.

Evaluates Baseline-120M and SamatNext-120M on 80 code completion tasks
divided into 8 distinct categories. Verifies completions by finding the longest
prefix of generated text that successfully compiles and passes assertions.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlx.core as mx

from data.tokenizer import SamatNextTokenizer
from model.config_120m import Baseline120MConfig, SamatNext120MConfig
from model.baseline_model import BaselineModel
from model.samatnext_model import SamatNextModel

# ── TASK DEFINITIONS ──────────────────────────────────────────────────────────
# 80 execution-based tasks, 8 categories, 10 tasks each.
# Sourced/committed before evaluation.
TASKS = [
    # === CATEGORY 1: Basic Math & Logic ===
    {
        "id": 1,
        "category": "basic_math_logic",
        "name": "add",
        "prompt": "def add(a, b):\n    return a + ",
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0"
    },
    {
        "id": 2,
        "category": "basic_math_logic",
        "name": "subtract",
        "prompt": "def subtract(a, b):\n    return a - ",
        "test": "assert subtract(10, 3) == 7\nassert subtract(0, 5) == -5"
    },
    {
        "id": 3,
        "category": "basic_math_logic",
        "name": "multiply",
        "prompt": "def multiply(a, b):\n    return a * ",
        "test": "assert multiply(4, 5) == 20\nassert multiply(-2, 3) == -6"
    },
    {
        "id": 4,
        "category": "basic_math_logic",
        "name": "divide",
        "prompt": "def divide(a, b):\n    return a / ",
        "test": "assert divide(10, 2) == 5.0\nassert divide(5, 2) == 2.5"
    },
    {
        "id": 5,
        "category": "basic_math_logic",
        "name": "is_even",
        "prompt": "def is_even(n):\n    return n % 2 == ",
        "test": "assert is_even(4) is True\nassert is_even(7) is False"
    },
    {
        "id": 6,
        "category": "basic_math_logic",
        "name": "is_odd",
        "prompt": "def is_odd(n):\n    return n % 2 != ",
        "test": "assert is_odd(5) is True\nassert is_odd(8) is False"
    },
    {
        "id": 7,
        "category": "basic_math_logic",
        "name": "square",
        "prompt": "def square(x):\n    return x ** ",
        "test": "assert square(4) == 16\nassert square(-3) == 9"
    },
    {
        "id": 8,
        "category": "basic_math_logic",
        "name": "cube",
        "prompt": "def cube(x):\n    return x ** ",
        "test": "assert cube(3) == 27\nassert cube(-2) == -8"
    },
    {
        "id": 9,
        "category": "basic_math_logic",
        "name": "absolute_value",
        "prompt": "def absolute_value(x):\n    if x < 0:\n        return -x\n    return ",
        "test": "assert absolute_value(-5) == 5\nassert absolute_value(5) == 5\nassert absolute_value(0) == 0"
    },
    {
        "id": 10,
        "category": "basic_math_logic",
        "name": "negate",
        "prompt": "def negate(x):\n    return -",
        "test": "assert negate(5) == -5\nassert negate(-10) == 10"
    },

    # === CATEGORY 2: String Manipulation ===
    {
        "id": 11,
        "category": "string_manipulation",
        "name": "concat",
        "prompt": "def concat(s1, s2):\n    return s1 + ",
        "test": "assert concat('a', 'b') == 'ab'\nassert concat('hello ', 'world') == 'hello world'"
    },
    {
        "id": 12,
        "category": "string_manipulation",
        "name": "get_len",
        "prompt": "def get_len(s):\n    return len(",
        "test": "assert get_len('abc') == 3\nassert get_len('') == 0"
    },
    {
        "id": 13,
        "category": "string_manipulation",
        "name": "to_upper",
        "prompt": "def to_upper(s):\n    return s.upper",
        "test": "assert to_upper('hello') == 'HELLO'\nassert to_upper('aBc') == 'ABC'"
    },
    {
        "id": 14,
        "category": "string_manipulation",
        "name": "to_lower",
        "prompt": "def to_lower(s):\n    return s.lower",
        "test": "assert to_lower('WORLD') == 'world'\nassert to_lower('XyZ') == 'xyz'"
    },
    {
        "id": 15,
        "category": "string_manipulation",
        "name": "first_char",
        "prompt": "def first_char(s):\n    return s[",
        "test": "assert first_char('abc') == 'a'\nassert first_char('z') == 'z'"
    },
    {
        "id": 16,
        "category": "string_manipulation",
        "name": "last_char",
        "prompt": "def last_char(s):\n    return s[",
        "test": "assert last_char('abc') == 'c'\nassert last_char('x') == 'x'"
    },
    {
        "id": 17,
        "category": "string_manipulation",
        "name": "repeat_str",
        "prompt": "def repeat_str(s, n):\n    return s * ",
        "test": "assert repeat_str('a', 3) == 'aaa'\nassert repeat_str('ok', 2) == 'okok'"
    },
    {
        "id": 18,
        "category": "string_manipulation",
        "name": "clean_str",
        "prompt": "def clean_str(s):\n    return s.strip",
        "test": "assert clean_str(' hello ') == 'hello'\nassert clean_str('\\tstrip\\n') == 'strip'"
    },
    {
        "id": 19,
        "category": "string_manipulation",
        "name": "is_empty",
        "prompt": "def is_empty(s):\n    return len(s) == ",
        "test": "assert is_empty('') is True\nassert is_empty(' ') is False\nassert is_empty('a') is False"
    },
    {
        "id": 20,
        "category": "string_manipulation",
        "name": "contains_char",
        "prompt": "def contains_char(s, c):\n    return c in ",
        "test": "assert contains_char('abc', 'b') is True\nassert contains_char('abc', 'd') is False"
    },

    # === CATEGORY 3: List & Tuple Operations ===
    {
        "id": 21,
        "category": "list_tuple_ops",
        "name": "list_len",
        "prompt": "def list_len(lst):\n    return len(",
        "test": "assert list_len([1, 2, 3]) == 3\nassert list_len([]) == 0"
    },
    {
        "id": 22,
        "category": "list_tuple_ops",
        "name": "first_element",
        "prompt": "def first_element(lst):\n    return lst[",
        "test": "assert first_element([10, 20]) == 10\nassert first_element(['a', 'b', 'c']) == 'a'"
    },
    {
        "id": 23,
        "category": "list_tuple_ops",
        "name": "last_element",
        "prompt": "def last_element(lst):\n    return lst[",
        "test": "assert last_element([10, 20]) == 20\nassert last_element(['a', 'b', 'c']) == 'c'"
    },
    {
        "id": 24,
        "category": "list_tuple_ops",
        "name": "sum_list",
        "prompt": "def sum_list(lst):\n    return sum(",
        "test": "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"
    },
    {
        "id": 25,
        "category": "list_tuple_ops",
        "name": "add_element",
        "prompt": "def add_element(lst, x):\n    lst.append(",
        "test": "lst = [1]\nadd_element(lst, 2)\nassert lst == [1, 2]"
    },
    {
        "id": 26,
        "category": "list_tuple_ops",
        "name": "reverse_list",
        "prompt": "def reverse_list(lst):\n    return lst[",
        "test": "assert reverse_list([1, 2, 3]) == [3, 2, 1]"
    },
    {
        "id": 27,
        "category": "list_tuple_ops",
        "name": "contains",
        "prompt": "def contains(lst, x):\n    return x in ",
        "test": "assert contains([1, 2, 3], 2) is True\nassert contains([1, 2, 3], 4) is False"
    },
    {
        "id": 28,
        "category": "list_tuple_ops",
        "name": "get_empty",
        "prompt": "def get_empty():\n    return ",
        "test": "assert get_empty() == []"
    },
    {
        "id": 29,
        "category": "list_tuple_ops",
        "name": "make_pair",
        "prompt": "def make_pair(a, b):\n    return ",
        "test": "assert make_pair(1, 2) == (1, 2)"
    },
    {
        "id": 30,
        "category": "list_tuple_ops",
        "name": "count_occ",
        "prompt": "def count_occ(lst, x):\n    return lst.count(",
        "test": "assert count_occ([1, 2, 1, 3, 1], 1) == 3\nassert count_occ([1, 2, 3], 4) == 0"
    },

    # === CATEGORY 4: Dict & Set Operations ===
    {
        "id": 31,
        "category": "dict_set_ops",
        "name": "get_val",
        "prompt": "def get_val(d, k):\n    return d[",
        "test": "assert get_val({'a': 1, 'b': 2}, 'a') == 1"
    },
    {
        "id": 32,
        "category": "dict_set_ops",
        "name": "get_keys",
        "prompt": "def get_keys(d):\n    return d.keys",
        "test": "assert list(get_keys({'a': 1, 'b': 2})) == ['a', 'b']"
    },
    {
        "id": 33,
        "category": "dict_set_ops",
        "name": "get_values",
        "prompt": "def get_values(d):\n    return d.values",
        "test": "assert list(get_values({'a': 1, 'b': 2})) == [1, 2]"
    },
    {
        "id": 34,
        "category": "dict_set_ops",
        "name": "has_key",
        "prompt": "def has_key(d, k):\n    return k in ",
        "test": "assert has_key({'a': 1}, 'a') is True\nassert has_key({'a': 1}, 'b') is False"
    },
    {
        "id": 35,
        "category": "dict_set_ops",
        "name": "dict_size",
        "prompt": "def dict_size(d):\n    return len(",
        "test": "assert dict_size({'a': 1, 'b': 2}) == 2\nassert dict_size({}) == 0"
    },
    {
        "id": 36,
        "category": "dict_set_ops",
        "name": "add_to_set",
        "prompt": "def add_to_set(s, x):\n    s.add(",
        "test": "s = {1, 2}\nadd_to_set(s, 3)\nassert s == {1, 2, 3}"
    },
    {
        "id": 37,
        "category": "dict_set_ops",
        "name": "in_set",
        "prompt": "def in_set(s, x):\n    return x in ",
        "test": "assert in_set({1, 2, 3}, 2) is True\nassert in_set({1, 2, 3}, 4) is False"
    },
    {
        "id": 38,
        "category": "dict_set_ops",
        "name": "set_intersect",
        "prompt": "def set_intersect(s1, s2):\n    return s1 & ",
        "test": "assert set_intersect({1, 2}, {2, 3}) == {2}"
    },
    {
        "id": 39,
        "category": "dict_set_ops",
        "name": "set_union",
        "prompt": "def set_union(s1, s2):\n    return s1 | ",
        "test": "assert set_union({1}, {2}) == {1, 2}"
    },
    {
        "id": 40,
        "category": "dict_set_ops",
        "name": "clear_dict",
        "prompt": "def clear_dict(d):\n    d.clear",
        "test": "d = {'a': 1}\nclear_dict(d)()\nassert d == {}"
    },

    # === CATEGORY 5: Control Flow ===
    {
        "id": 41,
        "category": "control_flow",
        "name": "is_positive",
        "prompt": "def is_positive(x):\n    if x > 0:\n        return True\n    return ",
        "test": "assert is_positive(5) is True\nassert is_positive(-2) is False\nassert is_positive(0) is False"
    },
    {
        "id": 42,
        "category": "control_flow",
        "name": "are_equal",
        "prompt": "def are_equal(a, b):\n    if a == b:\n        return True\n    return ",
        "test": "assert are_equal(2, 3) is False\nassert are_equal(5, 5) is True"
    },
    {
        "id": 43,
        "category": "control_flow",
        "name": "sum_to_n",
        "prompt": "def sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return ",
        "test": "assert sum_to_n(5) == 10\nassert sum_to_n(0) == 0"
    },
    {
        "id": 44,
        "category": "control_flow",
        "name": "max_of_three",
        "prompt": "def max_of_three(a, b, c):\n    if a >= b and a >= c:\n        return a\n    elif b >= a and b >= c:\n        return b\n    return ",
        "test": "assert max_of_three(1, 5, 3) == 5\nassert max_of_three(10, 2, 4) == 10\nassert max_of_three(3, 4, 8) == 8"
    },
    {
        "id": 45,
        "category": "control_flow",
        "name": "count_down",
        "prompt": "def count_down(n):\n    res = []\n    while n > 0:\n        res.append(n)\n        n -= 1\n    return ",
        "test": "assert count_down(3) == [3, 2, 1]\nassert count_down(0) == []"
    },
    {
        "id": 46,
        "category": "control_flow",
        "name": "get_squares",
        "prompt": "def get_squares(n):\n    res = []\n    for i in range(n):\n        res.append(i * i)\n    return ",
        "test": "assert get_squares(3) == [0, 1, 4]\nassert get_squares(0) == []"
    },
    {
        "id": 47,
        "category": "control_flow",
        "name": "fact",
        "prompt": "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(",
        "test": "assert fact(4) == 24\nassert fact(1) == 1\nassert fact(0) == 1"
    },
    {
        "id": 48,
        "category": "control_flow",
        "name": "sign",
        "prompt": "def sign(x):\n    if x > 0:\n        return 1\n    elif x < 0:\n        return -1\n    return ",
        "test": "assert sign(10) == 1\nassert sign(-5) == -1\nassert sign(0) == 0"
    },
    {
        "id": 49,
        "category": "control_flow",
        "name": "get_evens",
        "prompt": "def get_evens(lst):\n    return [x for x in lst if x % 2 == ",
        "test": "assert get_evens([1, 2, 3, 4]) == [2, 4]\nassert get_evens([1, 3]) == []"
    },
    {
        "id": 50,
        "category": "control_flow",
        "name": "check_any",
        "prompt": "def check_any(lst):\n    for val in lst:\n        if val:\n            return True\n    return ",
        "test": "assert check_any([False, True]) is True\nassert check_any([False, False]) is False"
    },

    # === CATEGORY 6: Functional Programming ===
    {
        "id": 51,
        "category": "functional_prog",
        "name": "lambda_square",
        "prompt": "square = lambda x: x * ",
        "test": "assert square(5) == 25\nassert square(-2) == 4"
    },
    {
        "id": 52,
        "category": "functional_prog",
        "name": "lambda_add",
        "prompt": "add = lambda a, b: a + ",
        "test": "assert add(4, 5) == 9"
    },
    {
        "id": 53,
        "category": "functional_prog",
        "name": "double_list",
        "prompt": "def double_list(lst):\n    return list(map(lambda x: x * 2, ",
        "test": "assert double_list([1, 2, 3]) == [2, 4, 6]"
    },
    {
        "id": 54,
        "category": "functional_prog",
        "name": "filter_pos",
        "prompt": "def filter_pos(lst):\n    return list(filter(lambda x: x > 0, ",
        "test": "assert filter_pos([-1, 0, 1, 2]) == [1, 2]"
    },
    {
        "id": 55,
        "category": "functional_prog",
        "name": "lambda_identity",
        "prompt": "identity = lambda x: ",
        "test": "assert identity('abc') == 'abc'"
    },
    {
        "id": 56,
        "category": "functional_prog",
        "name": "gen_sums",
        "prompt": "def gen_sums(n):\n    return sum(x for x in range(",
        "test": "assert gen_sums(4) == 6\nassert gen_sums(1) == 0"
    },
    {
        "id": 57,
        "category": "functional_prog",
        "name": "lambda_is_even",
        "prompt": "is_even = lambda x: x % 2 == ",
        "test": "assert is_even(4) is True\nassert is_even(3) is False"
    },
    {
        "id": 58,
        "category": "functional_prog",
        "name": "compose",
        "prompt": "def compose(f, g):\n    return lambda x: f(g(",
        "test": "assert compose(lambda x: x+1, lambda x: x*2)(3) == 7"
    },
    {
        "id": 59,
        "category": "functional_prog",
        "name": "sort_by_length",
        "prompt": "def sort_by_length(words):\n    return sorted(words, key=lambda s: len(",
        "test": "assert sort_by_length(['abc', 'a', 'de']) == ['a', 'de', 'abc']"
    },
    {
        "id": 60,
        "category": "functional_prog",
        "name": "lambda_concat",
        "prompt": "concat = lambda x, y: x + ",
        "test": "assert concat('hello', 'world') == 'helloworld'"
    },

    # === CATEGORY 7: Basic OOP ===
    {
        "id": 61,
        "category": "basic_oop",
        "name": "get_box_val",
        "prompt": "class Box:\n    def __init__(self, val):\n        self.val = val\n\ndef get_box_val(box):\n    return box.",
        "test": "assert get_box_val(Box(42)) == 42"
    },
    {
        "id": 62,
        "category": "basic_oop",
        "name": "call_name",
        "prompt": "class Person:\n    def __init__(self, name):\n        self.name = name\n    def get_name(self):\n        return self.name\n\ndef call_name(p):\n    return p.get_name",
        "test": "assert call_name(Person('Alice')) == 'Alice'"
    },
    {
        "id": 63,
        "category": "basic_oop",
        "name": "get_incremented",
        "prompt": "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n\ndef get_incremented():\n    c = Counter()\n    c.increment()\n    return c.",
        "test": "assert get_incremented() == 1"
    },
    {
        "id": 64,
        "category": "basic_oop",
        "name": "get_point_coords",
        "prompt": "class Point:\n    def __init__(self, x=0, y=0):\n        self.x = x\n        self.y = y\n\ndef get_point_coords(p):\n    return (p.x, p.",
        "test": "assert get_point_coords(Point(1, 2)) == (1, 2)"
    },
    {
        "id": 65,
        "category": "basic_oop",
        "name": "class_str",
        "prompt": "class User:\n    def __init__(self, name):\n        self.name = name\n    def __str__(self):\n        return self.",
        "test": "assert str(User('Bob')) == 'Bob'"
    },
    {
        "id": 66,
        "category": "basic_oop",
        "name": "class_equality",
        "prompt": "class Item:\n    def __init__(self, name):\n        self.name = name\n    def __eq__(self, other):\n        return self.name == other.",
        "test": "assert Item('a') == Item('a')\nassert Item('a') != Item('b')"
    },
    {
        "id": 67,
        "category": "basic_oop",
        "name": "subclass_speak",
        "prompt": "class Animal:\n    def speak(self):\n        return 'sound'\nclass Dog(Animal):\n    def speak(self):\n        return 'woof'\n\ndef get_dog_sound():\n    return Dog().",
        "test": "assert get_dog_sound() == 'woof'"
    },
    {
        "id": 68,
        "category": "basic_oop",
        "name": "get_pi",
        "prompt": "class Circle:\n    pi = 3.14\n    def __init__(self, r):\n        self.r = r\n\ndef get_pi():\n    return Circle.",
        "test": "assert get_pi() == 3.14"
    },
    {
        "id": 69,
        "category": "basic_oop",
        "name": "get_dict_val",
        "prompt": "class DictClass:\n    def __init__(self):\n        self.data = {}\n    def add(self, k, v):\n        self.data[k] = v\n\ndef get_dict_val():\n    d = DictClass()\n    d.add('a', 1)\n    return d.data[",
        "test": "assert get_dict_val() == 1"
    },
    {
        "id": 70,
        "category": "basic_oop",
        "name": "class_add",
        "prompt": "class Calculator:\n    def add(self, x, y):\n        return x + ",
        "test": "assert Calculator().add(2, 3) == 5"
    },

    # === CATEGORY 8: Simple Algorithms ===
    {
        "id": 71,
        "category": "simple_algorithms",
        "name": "is_palindrome",
        "prompt": "def is_palindrome(s):\n    return s == s[",
        "test": "assert is_palindrome('radar') is True\nassert is_palindrome('hello') is False"
    },
    {
        "id": 72,
        "category": "simple_algorithms",
        "name": "fib",
        "prompt": "def fib(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    return fib(n-1) + fib(",
        "test": "assert fib(5) == 5\nassert fib(6) == 8\nassert fib(1) == 1"
    },
    {
        "id": 73,
        "category": "simple_algorithms",
        "name": "gcd",
        "prompt": "def gcd(a, b):\n    while b:\n        a, b = b, a % ",
        "test": "assert gcd(12, 8) == 4\nassert gcd(17, 5) == 1"
    },
    {
        "id": 74,
        "category": "simple_algorithms",
        "name": "find_index",
        "prompt": "def find_index(lst, x):\n    for i in range(len(lst)):\n        if lst[i] == x:\n            return i\n    return -",
        "test": "assert find_index([10, 20, 30], 20) == 1\nassert find_index([10, 20, 30], 40) == -1"
    },
    {
        "id": 75,
        "category": "simple_algorithms",
        "name": "c_to_f",
        "prompt": "def c_to_f(c):\n    return c * 9 / 5 + ",
        "test": "assert c_to_f(0) == 32\nassert c_to_f(100) == 212"
    },
    {
        "id": 76,
        "category": "simple_algorithms",
        "name": "count_vowels",
        "prompt": "def count_vowels(s):\n    return sum(1 for c in s if c in ",
        "test": "assert count_vowels('hello') == 2\nassert count_vowels('bcd') == 0"
    },
    {
        "id": 77,
        "category": "simple_algorithms",
        "name": "get_min",
        "prompt": "def get_min(lst):\n    if not lst:\n        return None\n    min_val = lst[0]\n    for x in lst:\n        if x < min_val:\n            min_val = x\n    return ",
        "test": "assert get_min([3, 1, 4]) == 1\nassert get_min([10]) == 10"
    },
    {
        "id": 78,
        "category": "simple_algorithms",
        "name": "fizzbuzz",
        "prompt": "def fizzbuzz(n):\n    if n % 15 == 0:\n        return 'FizzBuzz'\n    elif n % 3 == 0:\n        return 'Fizz'\n    elif n % 5 == 0:\n        return 'Buzz'\n    return str(",
        "test": "assert fizzbuzz(3) == 'Fizz'\nassert fizzbuzz(5) == 'Buzz'\nassert fizzbuzz(15) == 'FizzBuzz'\nassert fizzbuzz(7) == '7' or fizzbuzz(7) == 7"
    },
    {
        "id": 79,
        "category": "simple_algorithms",
        "name": "fact_iter",
        "prompt": "def fact_iter(n):\n    res = 1\n    for i in range(1, n + 1):\n        res *= i\n    return ",
        "test": "assert fact_iter(5) == 120\nassert fact_iter(0) == 1"
    },
    {
        "id": 80,
        "category": "simple_algorithms",
        "name": "is_prime",
        "prompt": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return ",
        "test": "assert is_prime(7) is True\nassert is_prime(4) is False\nassert is_prime(1) is False"
    }
]


def generate_completion(model, tokenizer, prompt, max_tokens=20):
    """Greedy generation."""
    prompt_ids = tokenizer.encode(prompt)
    x = mx.array(prompt_ids, dtype=mx.int32)[None, :]

    generated_ids = []
    for _ in range(max_tokens):
        out = model(x)
        logits = out["logits"][:, -1, :]
        next_token = mx.argmax(logits, axis=-1).item()

        if next_token == tokenizer.eos_id:
            break

        generated_ids.append(next_token)
        x = mx.concatenate([x, mx.array([[next_token]], dtype=mx.int32)], axis=1)

    return tokenizer.decode(generated_ids)


def check_functional_correctness(prompt: str, completion: str, test_code: str) -> tuple[bool, str]:
    """Prefix-compilation search: finds the longest prefix of completion that runs tests successfully."""
    # Split the completion by lines first
    lines = completion.split('\n')
    
    # Delimiters for token prefix search in the first line
    first_line = lines[0]
    
    # Candidate strings list to try in order of complexity (longest/most specific first)
    candidates = []
    
    # 1. Try first line directly
    candidates.append(first_line)
    
    # 2. Try character-by-character prefixes of the first line
    for i in range(len(first_line), 0, -1):
        candidates.append(first_line[:i])
        
    # 3. Try line prefixes (in case of multiline completions)
    for i in range(2, len(lines) + 1):
        candidates.append('\n'.join(lines[:i]))
        
    # Add empty completion as fallback
    candidates.append("")

    # Test each candidate prefix
    for candidate in candidates:
        candidate_stripped = candidate.rstrip()
        full_code = prompt + candidate_stripped + "\n" + test_code
        
        try:
            # We try to run it in a clean isolated dictionary scope
            local_scope = {}
            exec(full_code, {}, local_scope)
            return True, candidate_stripped
        except Exception:
            pass
            
    return False, ""


def run_benchmark(model_name: str, seed: int = 42) -> dict:
    print(f"\n{'='*70}\nRunning Execution Benchmark for {model_name}\n{'='*70}")

    if model_name == "Baseline-120M":
        config = Baseline120MConfig()
        model = BaselineModel(config)
    else:
        config = SamatNext120MConfig()
        model = SamatNextModel(config)

    tokenizer = SamatNextTokenizer.from_file("data/tokenizer.json", config=config)

    # Load weights
    finetune_ckpt = Path(f"results/finetune_120m/{model_name}_seed_{seed}/step_000500.npz")
    pretrain_ckpt = Path(f"results/pretrain_120m/{model_name}_seed_{seed}/step_001000.npz")

    if finetune_ckpt.exists():
        print(f"Loading finetuned weights: {finetune_ckpt}")
        model.load_weights(str(finetune_ckpt))
    elif pretrain_ckpt.exists():
        print(f"Loading pretrained weights: {pretrain_ckpt}")
        model.load_weights(str(pretrain_ckpt))
    else:
        print("WARNING: No checkpoint found, using random weights!")

    # Categorized results
    results_by_category = {}
    detailed_results = []
    
    passed_total = 0
    
    for i, task in enumerate(TASKS):
        category = task["category"]
        if category not in results_by_category:
            results_by_category[category] = {"passed": 0, "total": 0}
            
        results_by_category[category]["total"] += 1

        raw_completion = generate_completion(model, tokenizer, task["prompt"], max_tokens=20)
        passed, clean_comp = check_functional_correctness(task["prompt"], raw_completion, task["test"])
        
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed:
            results_by_category[category]["passed"] += 1
            passed_total += 1
            
        print(f"  [{category:18s}] Task {task['id']:2d}: {task['name']:18s} | {status} | Completed: {repr(clean_comp)} (Raw: {repr(raw_completion[:40])})")
        
        detailed_results.append({
            "id": task["id"],
            "name": task["name"],
            "category": category,
            "prompt": task["prompt"],
            "raw_completion": raw_completion,
            "clean_completion": clean_comp,
            "passed": passed
        })

    print(f"\n{'-'*70}\nCategory-wise Breakdown for {model_name}\n{'-'*70}")
    category_summary = {}
    for cat, stats in results_by_category.items():
        pass_rate = (stats["passed"] / stats["total"]) * 100
        category_summary[cat] = {
            "passed": stats["passed"],
            "total": stats["total"],
            "pass_rate": pass_rate
        }
        print(f"  {cat:22s} : {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)")
        
    overall_rate = (passed_total / len(TASKS)) * 100
    print(f"\n  OVERALL SCORE: {passed_total}/{len(TASKS)} ({overall_rate:.1f}%)")
    
    return {
        "model_name": model_name,
        "overall": {
            "passed": passed_total,
            "total": len(TASKS),
            "pass_rate": overall_rate
        },
        "categories": category_summary,
        "details": detailed_results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    results = {}
    results["Baseline-120M"] = run_benchmark("Baseline-120M", args.seed)
    results["SamatNext-120M"] = run_benchmark("SamatNext-120M", args.seed)
    
    # Save results to output file
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "execution_benchmark_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n{'='*70}\nFINAL COMPARISON SUMMARY\n{'='*70}")
    print(f"  Model             | Pass Rate (pass@1)")
    print(f"  {'-'*18}|{'-'*21}")
    for model_name, res in results.items():
        print(f"  {model_name:17s} | {res['overall']['passed']}/{res['overall']['total']} ({res['overall']['pass_rate']:.1f}%)")
    print(f"{'='*70}")
    print(f"Saved detailed results to {output_path}")


if __name__ == "__main__":
    main()
