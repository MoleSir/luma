# 梯度下降 vs grad boost

## 梯度下降

有预测值 $y_{pred}$ 需要逼近真实值 $y$，利用 $\text{loss}$ 函数评估二者的差距，并且希望 $\text {loss}$ 尽可能小。计算 $g = \text d \text {loss} / \text d w$，得到当前参数如何影响输出损失。因为梯度的方向表示：按照梯度方向变化一点 $\Delta w$，使得损失变化 $\Delta \text{loss} = \Delta w \cdot g$，如果 $g < 0$，$\Delta w$ 应该大于 0，反之小于 0。因为 $g$ 和 $\Delta w$ 变化方向相反，正好用 $\Delta w = - \eta \cdot g$。所以这里的学习率 $\eta $ 似乎找不到什么合适的物理意义？他将梯度数值转为 $w$ 的变化量。



## grad boost

而对 grad boost，初始化 $y_{pred} = 0$，进行迭代：

同样需要预测值 $y_{pred}$ 需要逼近真实值 $y$，利用 $\text{loss}$ 函数评估二者的差距，并且希望 $\text {loss}$ 尽可能小。但和梯度下降不同，需要计算的梯度不是直接对参数的，只是对 $y_{pred}$ 的：$g = \text d \text {loss} / y_{pred}$。

我们利用一个网络直接拟合 $-g$ 本身。即 $g' = Model(x)$，表示希望模型预测 $-g$，训练后预测值为 $g'$。

得到 $g'$ 后如何更新 $y_{pred}$？$y_{pred} = \eta g'$。将预测的梯度乘 $\eta $，这里的学习率和梯度下降完全不同了，这里的 $\eta$ 物理意义上是步长，我们用梯度乘步长来近似这个模型可以带来的 $y_{pred}$。