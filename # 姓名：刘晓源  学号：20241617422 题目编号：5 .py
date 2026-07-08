# 姓名：张三 学号：20240101 题目编号：6
# 1.定义学生列表，元素为学生字典
student_list = [
    {"name": "小明", "student_id": 2024001, "python_score": 88, "class_": "计科1班"},
    {"name": "小红", "student_id": 2024002, "python_score": 95, "class_": "计科1班"},
    {"name": "小李", "student_id": 2024003, "python_score": 95, "class_": "计科2班"}
]

# 2.遍历打印所有学生信息
print("=====原始学生信息=====")
for stu in student_list:
    print(f"姓名:{stu['name']},学号:{stu['student_id']},班级:{stu['class_']},Python成绩:{stu['python_score']}")

# 3.计算平均分
score_total = 0
for stu in student_list:
    score_total += stu["python_score"]
avg_score = score_total / len(student_list)
print(f"\n所有学生Python成绩的平均分为：{avg_score:.1f}")

# 4.查找最高分学生
# 先拿到最高分
max_score = max(stu["python_score"] for stu in student_list)
print(f"\nPython成绩最高({max_score}分)的学生：")
for stu in student_list:
    if stu["python_score"] == max_score:
        print(f"姓名:{stu['name']},学号:{stu['student_id']},班级:{stu['class_']},Python成绩:{stu['python_score']}")

# 5.新增学生、修改已有学生成绩
student_list.append({"name": "小张", "student_id": 2024004, "python_score": 76, "class_": "计科2班"})
student_list[0]["python_score"] = 92  # 修改第一个学生成绩

# 打印修改后全部信息
print("\n=====修改后学生信息=====")
for stu in student_list:
    print(f"姓名:{stu['name']},学号:{stu['student_id']},班级:{stu['class_']},Python成绩:{stu['python_score']}")