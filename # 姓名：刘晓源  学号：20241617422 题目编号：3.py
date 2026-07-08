# 姓名：张三 学号：20240101 题目编号：3
# 获取输入并转为整数
score = int(input("请输入学生Python成绩(0-100整数)："))
# 判断成绩合法性
if score < 0 or score > 100:
    print("输入成绩无效，请输入0-100之间的整数")
else:
    if 90 <= score <= 100:
        print("A级")
    elif 80 <= score <= 89:
        print("B级")
    elif 70 <= score <= 79:
        print("C级")
    elif 60 <= score <= 69:
        print("D级")
    else:
        print("不及格")