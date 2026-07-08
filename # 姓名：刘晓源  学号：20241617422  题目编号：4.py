# 姓名：张三 学号：20240101 题目编号：4
# 任务1 for循环打印1-50能被3整除的数，每5个换行
print("=====任务1=====")
count = 0
for num in range(1, 51):
    if num % 3 == 0:
        print(num, end="\t")
        count += 1
        if count % 5 == 0:
            print()
print()

# 任务2 while循环计算1-100偶数和
print("=====任务2=====")
sum_even = 0
i = 2
while i <= 100:
    sum_even += i
    i += 2
print(f"1到100之间所有偶数的和为：{sum_even}")