# 姓名：张三 学号：20240101 题目编号：1
# 1.定义不同类型变量
name = "张三"
stu_id = 20240101
target_score = 95.5

# 2.打印变量值与对应数据类型
print("姓名值：", name, "，数据类型：", type(name))
print("学号值：", stu_id, "，数据类型：", type(stu_id))
print("目标成绩值：", target_score, "，数据类型：", type(target_score))

# 3.数据类型转换操作
int_score = int(target_score)
print("成绩转整数后值：", int_score, "，数据类型：", type(int_score))
str_id = str(stu_id)
new_id = "2025-" + str_id
print("拼接结果：", new_id)