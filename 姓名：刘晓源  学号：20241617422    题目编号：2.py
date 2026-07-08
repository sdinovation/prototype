# 姓名：张三 学号：20240101 题目编号：2
# 定义图形尺寸
rec_long = 15.6
rec_wide = 8.3
sq_side = 12

# 1.长方形周长、面积
rec_peri = 2 * (rec_long + rec_wide)
rec_area = rec_long * rec_wide
print("长方形周长：{:.1f}cm".format(rec_peri))
print("长方形面积：{:.1f}cm²".format(rec_area))

# 2.正方形周长、面积
sq_peri = 4 * sq_side
sq_area = sq_side ** 2
print("正方形周长：{:.1f}cm".format(sq_peri))
print("正方形面积：{:.1f}cm²".format(sq_area))

# 3.面积差值与布尔判断
diff = rec_area - sq_area
judge = diff > 0
print("长方形与正方形面积差值：", diff)
print("差值是否大于0：", judge)