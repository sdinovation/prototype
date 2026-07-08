# 姓名：张三 学号：20240101 题目编号：7
import os

# 1.定义成绩统计函数
def calculate_score(score_list):
    """
    统计成绩列表最高分、最低分、平均分
    :param score_list: 整数成绩列表
    :return: max_s(最高分), min_s(最低分), avg_s(平均分，保留1位小数)
    """
    max_s = max(score_list)
    min_s = min(score_list)
    avg_s = sum(score_list) / len(score_list)
    avg_s = round(avg_s, 1)
    return max_s, min_s, avg_s

# 2.输入5名学生成绩
score_data = []
for i in range(5):
    s = int(input(f"请输入第{i+1}名学生Python成绩："))
    score_data.append(s)

# 调用函数获取统计结果
high, low, average = calculate_score(score_data)

# 3.写入txt文件到G:\demo
save_dir = "G:/demo"
os.makedirs(save_dir, exist_ok=True)  # 自动创建文件夹，不存在就建，已存在不报错
file_path = os.path.join(save_dir, "python成绩统计.txt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"学生成绩列表:{score_data}\n")
    f.write(f"最高分:{high}\n")
    f.write(f"最低分:{low}\n")
    f.write(f"平均分:{average}\n")
print(f"数据已保存至：{file_path}")