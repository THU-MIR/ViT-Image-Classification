import matplotlib.pyplot as plt
import csv
import numpy as np

train_data = []
test_datas = []
ir_date = []

# 需要手动修改可视化的路径和文件名（训练日志文件存储在logs目录下）
with open('logs/vit_c10_aa_ls/{}/metrics.csv'.format("version_0"), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if(row[0] == ''):
            if(row[4] == ''): # 记录学习率 lr
                ir_date.append(row)
            else:
                test_datas.append(row) # 记录 val_loss,val_acc
                
        elif(row[0] != 'loss'):
            train_data.append(row)


# 获取列数 (有3列不用可视化)
num_columns = len(train_data[0]) - 3

# 创建子图
fig, axes = plt.subplots(num_columns, 1, figsize=(10, num_columns*5-10))

axes[0].set_title('Dropout {}'.format("0.1"))

# 绘制 Val_Loss
val_loss_datas = []
for val_loss_data in test_datas:
    val_loss_datas.append(float(val_loss_data[4]))

axes[0].plot(range(len(val_loss_datas)), val_loss_datas, color='red')
axes[0].set_ylabel('Val_Loss')
axes[0].set_xlabel('Batch')
axes[0].grid(True)
axes[0].set_xlim(0, 200)
axes[0].set_ylim(0, 3)

# 选择要标注的点的索引
index = -1

# 获取要标注的点的坐标和值
x_point = range(len(val_loss_datas))[index]
y_point = val_loss_datas[index]

# 添加文字标注
axes[0].annotate(f'Val Loss: {y_point}', (x_point, y_point), xytext=(x_point-50, y_point+1),
             arrowprops=dict(arrowstyle='->'))



# 绘制 Val_Acc
val_acc_datas = []
for val_acc_data in test_datas:
    val_acc_datas.append(float(val_acc_data[5]))

axes[1].plot(range(len(val_acc_datas)), val_acc_datas, color='red')
axes[1].set_ylabel('Val_Acc')
axes[1].set_xlabel('Batch')
axes[1].grid(True)
axes[1].set_xlim(0, 200)
axes[1].set_ylim(0, 1)

# 选择要标注的点的索引
index = -1

# 获取要标注的点的坐标和值
x_point = range(len(val_acc_datas))[index]
y_point = val_acc_datas[index]

# 添加文字标注
axes[1].annotate(f'Val Acc: {y_point}', (x_point, y_point), xytext=(x_point-50, y_point-0.3),
             arrowprops=dict(arrowstyle='->'))

# # 绘制 Ir
# ir_datas = []
# for ir_data in ir_date:
#     ir_datas.append(float(ir_data[6]))

# axes[2].plot(range(len(ir_datas)), ir_datas)
# axes[2].set_ylabel('Ir')
# axes[2].set_xlabel('Batch')
# # 设置纵轴限制范围为0到100
# # axes[1].set_ylim(bottom=0, top=1)
# # 设置纵轴刻度为10的倍数
# # axes[1].set_yticks(range(0, 1, 10))
# axes[2].grid(True)
# axes[2].set_xlim(0)

# 绘制 Loss
loss_datas = []
for loss_data in train_data:
    loss_datas.append(float(loss_data[0]))

axes[2].plot(range(len(loss_datas)), loss_datas)
axes[2].set_ylabel('Loss')
axes[2].set_xlabel('Step')
axes[2].grid(True)
axes[2].set_xlim(0, 1600)
axes[2].set_ylim(0, 3)


# 绘制 Acc
acc_datas = []
for acc_data in train_data:
    acc_datas.append(float(acc_data[1]))

axes[3].plot(range(len(acc_datas)), acc_datas)
axes[3].set_ylabel('Acc')
axes[3].set_xlabel('Step')
axes[3].grid(True)
axes[3].set_xlim(0, 1600)
axes[3].set_ylim(0, 1)

plt.tight_layout()
plt.show()
