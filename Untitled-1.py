# 基类 Person
class Person:
    def __init__(self):
        self.id = ""     # 编号
        self.name = ""   # 姓名

    # 录入公共信息
    def input(self):
        self.id = input("请输入编号：")
        self.name = input("请输入姓名：")

    # 展示公共信息
    def display(self):
        print(f"编号：{self.id}")
        print(f"姓名：{self.name}")


# 学生类，继承 Person
class Student(Person):
    def __init__(self):
        super().__init__()
        self.class_no = ""  # 班号
        self.score = 0.0    # 成绩

    # 重写输入方法
    def input(self):
        print("===== 录入学生信息 =====")
        super().input()
        self.class_no = input("请输入班号：")
        self.score = float(input("请输入成绩："))

    # 重写展示方法
    def display(self):
        print("\n===== 学生详细信息 =====")
        super().display()
        print(f"班号：{self.class_no}")
        print(f"成绩：{self.score}")


# 教师类，继承 Person
class Teacher(Person):
    def __init__(self):
        super().__init__()
        self.title = ""       # 职称
        self.department = ""  # 所属部门

    # 重写输入方法
    def input(self):
        print("===== 录入教师信息 =====")
        super().input()
        self.title = input("请输入职称：")
        self.department = input("请输入部门：")

    # 重写展示方法
    def display(self):
        print("\n===== 教师详细信息 =====")
        super().display()
        print(f"职称：{self.title}")
        print(f"部门：{self.department}")


# 主程序调用
if __name__ == "__main__":
    stu = Student()
    tea = Teacher()

    # 录入数据
    stu.input()
    tea.input()

    # 打印数据
    stu.display()
    tea.display()