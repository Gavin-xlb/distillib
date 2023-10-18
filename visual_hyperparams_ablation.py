import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt


# x1 =          [0.06, 0.07, 0.08, 0.09, 0.10]
# y1 = [0.6827, 0.6376, 0.6981, 0.6725, 0.6823]  # 参数精度

# x2 =        [0.19, 0.199, 0.1999, 0.2099, 0.2199]
# y2 =        [0.6822, 0.6627, 0.6981, 0.6696, 0.6196]  # 训练时间精度

# 设置更好看的字体和样式
sns.set(font_scale=1.2, style="white", font = "sans-serif")
sns.set_style('ticks')

# 创建一个包含两个正方形子图的图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})

# 第一个子图：
data1 = {'α':[0.06, 0.07, 0.08, 0.09, 0.10],'DSC(%)':[68.27, 63.76, 69.81, 67.25, 68.23]}
sns.lineplot(x='α', y='DSC(%)', data=data1, ax=ax1, color='orange')
# ax1.set_xlabel("α")
# ax1.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
# 添加第一个子图的辅助线
y_value = 52.61
ax1.axhline(y_value, color='gray', linestyle='--', lw=2)
# ax1.annotate(f'y = {y_value}', xy=(0, y_value), xytext=(5, y_value+1), color='gray')
# 添加"Accuracy"标签到左上角，略偏离子图位置
# ax1.text(min(α)-0.0083, max(DSC(%)) + 0.019, "DSC(%)", ha='left', va='top', color='black', fontsize=12)
ax1.set_xlim(0.0545, 0.106)
ax1.set_ylim(51, 72)
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
for i, y in enumerate(data1['DSC(%)']):
    if i == 2:
        ax1.text(data1['α'][i], y, str(y), va='bottom', ha='center', weight = 'bold')
    else:
        ax1.text(data1['α'][i], y, str(y), va='bottom', ha='center')
ax1.text(0.08, y_value + 1, "baseline", ha='center', va='top', color='black', fontsize=15)

# 第二个子图：精度与训练时间的关系图
data2 = {'β':[0.19, 0.199, 0.1999, 0.2099, 0.2199],'DSC(%)':[68.22, 66.27, 69.81, 66.96, 61.96]}
sns.lineplot(x='β', y='DSC(%)', data=data2, ax=ax2, color=(0.4, 0.6, 0.8))
# ax2.set_xlabel("β")
# ax2.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])
for i, y in enumerate(data2['DSC(%)']):
    if i == 2:
        ax2.text(data2['β'][i], y, str(y), va='bottom', ha='center', weight = 'bold')
    else:
        ax2.text(data2['β'][i], y, str(y), va='bottom', ha='center')
# 添加第二个子图的辅助线
y_value = 52.61
ax2.axhline(y_value, color='gray', linestyle='--', lw=2)
ax2.annotate(f'y = {y_value}', xy=(0, y_value), xytext=(5, y_value+1), color='gray')
# ax2.text(min(x2) - 0.0063, max(y2) + 0.019, "DSC(%)", ha='left', va='top', color='black', fontsize=12)
ax2.text(0.205, y_value + 1, "baseline", ha='center', va='top', color='black', fontsize=15)
ax2.set_xlim(0.186, 0.224)
ax2.set_ylim(51, 72)
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')


# 调整子图之间的间距
plt.tight_layout()

# 保存图表为图像文件
plt.savefig("hyperparams_ablation.png")  # 可以更改文件名和格式

# 显示图表
plt.show()

