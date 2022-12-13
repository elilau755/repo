class Person:
    def __call__(self, name):  # __表示内置函数
        print("__call__" + "Hello" + name)
    def hello(self, name):
        print("hello" + "name")


person = Person()
person("Zhangsan")  # 内置了__call__(self, name)，则不需要".xx"
person.hello("Lisi")