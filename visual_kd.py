import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 设置更好看的字体和样式
sns.set(font_scale=1.2, style="white", font = "sans-serif")
sns.set_style('ticks')
# 创建示例数据
# UNet->ENet KiTS Tumor
data1 = pd.DataFrame({
    'kd_method': ["EM\nKD", "Review\nKD", "MGD", "OFD", "Tf-KD", "SP", "GID", "AT", "RKD", "Ours"],
    'DSC(%)': [66.280, 61.410, 57.79, 53.02, 52.68, 59.51, 53.520, 65.90, 60.57, 69.81],
    'baseline': 52.61
})
# UNet->ENet LiTS Tumor
data2 = pd.DataFrame({
    'kd_method': ["EM\nKD", "Review\nKD", "MGD", "OFD", "Tf-KD", "SP", "GID", "AT", "RKD", "Ours"],
    '': [59.49, 60.07, 56.89, 56.92, 58.53, 56.20, 57.30, 59.94, 59.09, 60.31],
    'baseline': 57.69
})
# PSPNet->ENet KiTS Tumor
data3 = pd.DataFrame({
    'kd_method': ["EM\nKD", "Review\nKD", "MGD", "OFD", "Tf-KD", "SP", "GID", "AT", "RKD", "Ours"],
    'DSC(%)': [58.29, 61.05, 59.80, 60.20, 58.80, 58.18, 53.77, 60.20, 55.87, 59.40],
    'baseline': 52.61
})
# PSPNet->ENet LiTS Tumor
data4 = pd.DataFrame({
    'kd_method': ["EM\nKD", "Review\nKD", "MGD", "OFD", "Tf-KD", "SP", "GID", "AT", "RKD", "Ours"],
    '': [62.57, 58.81, 56.89, 61.51, 58.41, 58.35, 58.28, 60.50, 58.78, 62.60],
    'baseline': 57.69
})

# 创建一个包含四个子图的图形
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes[0, 0].spines['top'].set_visible(False)
# axes[0, 0].spines['right'].set_visible(False)
# 子图1 - 柱形图1
sns.barplot(x="kd_method", y="DSC(%)", data=data1, ax=axes[0, 0])
axes[0, 0].axhline(y=52.61, color='black', linestyle='--', lw=2)
axes[0, 0].set_title('UNet->ENet KiTS Tumor')
axes[0, 0].set_ylim(0, 75)
# 在每个柱子上方添加数值标签到子图1
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

# 子图2 - 柱形图2
sns.barplot(x="kd_method", y="", data=data2, ax=axes[0, 1])
axes[0, 1].axhline(y=57.69, color='black', linestyle='--', lw=2)
axes[0, 1].set_title('UNet->ENet LiTS Tumor')
axes[0, 1].set_ylim(0, 75)
# 在每个柱子上方添加数值标签到子图2
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

# 子图3 - 柱形图3
sns.barplot(x="kd_method", y="DSC(%)", data=data3, ax=axes[1, 0])
axes[1, 0].axhline(y=52.61, color='black', linestyle='--', lw=2)
axes[1, 0].set_title('PSPNet->ENet KiTS Tumor')
axes[1, 0].set_ylim(0, 75)
# 在每个柱子上方添加数值标签到子图3
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

# 子图4 - 柱形图4
sns.barplot(x="kd_method", y="", data=data4, ax=axes[1, 1])
axes[1, 1].axhline(y=57.69, color='black', linestyle='--', lw=2)
axes[1, 1].set_title('PSPNet->ENet LiTS Tumor')
axes[1, 1].set_ylim(0, 75)
# 在每个柱子上方添加数值标签到子图4
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')

# 调整子图之间的间距
plt.tight_layout()

# 保存图表为图像文件
plt.savefig("kd.png")  # 可以更改文件名和格式

# 显示图形
plt.show()
