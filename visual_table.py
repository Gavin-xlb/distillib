import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据（请替换为您自己的数据）
parameters =          [22.10, 46.70, 34.50, 20.60, 0.353, 2.200, 0.183, 11.20, 56.80, 2.100, 0.353]
accuracy_parameters = [78.56, 74.94, 64.33, 66.49, 52.61, 67.83, 51.24, 50.68, 77.59, 65.95, 69.81]  # 参数精度
# accuracy_parameters = [0.7856, 0.7494, 0.6433, 0.6649, 0.5261, 0.6783, 0.5124, 0.5068, 0.7759, 0.6595, 0.6981]  # 参数精度
model_data_labels = ["RA-UNet", "PSPNet", "UNet", "UNet++", "ENet", "MobileNetV2", "ESPNet", "ResNet-18", "DeeplabV3+", "ERFNet", "ENet(Ours)"]

accuracy_time =        [66.280, 61.410, 57.79, 53.02, 52.68, 59.51, 53.520, 65.90, 60.57, 69.81]  # 训练时间精度
training_time =        [164.05, 154.28, 249.64, 176.42, 192.64, 181.82, 116.21, 163.35, 121.92, 147.13]
training_data_labels = ["EMKD", "ReviewKD", "MGD", "OFD", "Tf-KD", "SP", "GID", "AT", "RKD", "Ours"]

# 设置更好看的字体和样式
sns.set(font_scale=1.2, style="white", font = "sans-serif")
sns.set_style('ticks')

# 创建一个包含两个正方形子图的图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
# 设置颜色调色板，确保颜色足够多以匹配数据点
color_palette = sns.color_palette("deep", n_colors=len(parameters) + 10)
# print(color_palette)
color_palette = [color for color in color_palette if color != (1.0, 1.0, 0.2) and color != (0.8941176470588236, 0.10196078431372549, 0.10980392156862745)]  # 去掉黄色、红色

# 第一个子图：精度与参数量的散点图
for i, label in enumerate(model_data_labels[:len(model_data_labels)]):
    color = color_palette[i]
    if i < len(model_data_labels) - 1:
        sns.scatterplot(x=[parameters[i]], y=[accuracy_parameters[i]], color=color, marker='o', s=120, ax=ax1)
        ax1.text(parameters[i], accuracy_parameters[i] - 0.4, label, ha='center', va='top', color=color, fontsize=12, weight='bold')
        # # 使用 ax.plot() 绘制从起始点到横坐标的辅助线
        # ax1.plot([parameters[i], parameters[i]], [0, accuracy_parameters[i]], color='gray', linestyle='--')
        # ax1.plot([0, parameters[i]], [accuracy_parameters[i], accuracy_parameters[i]], color='gray', linestyle='--', )

    else:
        sns.scatterplot(x=[parameters[i]], y=[accuracy_parameters[i]], color='red', marker='*', s=500, ax=ax1)
        ax1.text(parameters[i], accuracy_parameters[i] - 0.4, label, ha='center', va='top', color='red', fontsize=12, weight='bold')
ax1.set_xlabel("Parameters(M)")
# ax1.set_title("Accuracy vs. Parameters")
# ax1.legend(["Accuracy"])
# 添加"Accuracy"标签到左上角，略偏离子图位置
ax1.text(min(parameters)-17, max(accuracy_parameters) + 2.8, "DSC(%)", ha='left', va='top', color='black', fontsize=12)
ax1.set_xlim(-9, 69)
ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
# custom_xticks = [0, 1, 2, 3, 4]  # 刻度位置
# custom_xticklabels = ["A", "B", "C", "D", "E"]  # 刻度标签
# ax1.set_xticks(custom_xticks)

# ax1.set_ylim(50, 80)

# 第二个子图：精度与训练时间的关系图
for i, label in enumerate(training_data_labels[:len(training_data_labels)]):
    color = color_palette[i]
    if i < len(training_data_labels) - 1:
        sns.scatterplot(x=[training_time[i]], y=[accuracy_time[i]], color=color, marker='^', s=120, ax=ax2)
        if i == 0: # EMKD
            ax2.text(training_time[i], accuracy_time[i] + 0.8, label, ha='center', va='top', color=color, fontsize=12, weight='bold')
        else:
            ax2.text(training_time[i], accuracy_time[i] - 0.5, label, ha='center', va='top', color=color, fontsize=12, weight='bold')
    else:
        sns.scatterplot(x=[training_time[i]], y=[accuracy_time[i]], color='red', marker='*', s=500, ax=ax2)
        ax2.text(training_time[i], accuracy_time[i] - 0.5, label, ha='center', va='top', color='red', fontsize=12, weight='bold')
ax2.set_xlabel("Average Training Time per Epoch(s)")
# ax2.set_title("Accuracy vs. Training Time")
# 添加"Accuracy"标签到左上角，略偏离子图位置
ax2.text(min(training_time) - 25, max(accuracy_time) + 2.1, "DSC(%)", ha='left', va='top', color='black', fontsize=12)
ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
ax2.set_xlim(110, 260)
ax2.set_ylim(51, 71)

# 调整子图之间的间距
plt.tight_layout()

# 保存图表为图像文件
plt.savefig("combined_plot.png")  # 可以更改文件名和格式

# 显示图表
plt.show()
