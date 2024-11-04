# DeepFilterNet2

## 摘要
  基于深度学习的语音增强技术已经取得了巨大的进步，最近还扩展到了全频带音频(48 kHz)。然而，许多方法都有相当高的计算复杂度，需要大的时间缓冲来实时使用，例如由于时间卷积或注意力。这两者都使得这些方法在嵌入式设备上不可行。这项工作进一步扩展了DeepFilterNet，利用语音的谐波结构实现了有效的语音增强。训练过程、数据增强和网络结构中的几项优化使SE性能达到了最先进的水平，同时在笔记本Core-i5 CPU上将实时因子降低到0.04。这使得算法可以在嵌入式设备上实时运行。DeepFilterNet框架可以在开源许可下获取。
## 关键词：DeepFilterNet，语音增强，全波段，两级建模

## 我们在DNS4数据集上训练DeepFilterNet2，总共使用超过500小时的全波段纯净语音(大约)。150 小时的噪声以及150个真实的和60000个模拟的HRTFs。我们将数据分为训练集、验证集和测试集(70%、15%、15%)。Voicebank集split为speaker-exclusive，与测试集没有重叠。我们在Voicebank+Demand测试集[8]和DNS4盲测试集[9]上评估了我们的方法。我们用AdamW对模型进行了100个epoch的训练，并根据验证损失选择最佳模型。
  在这项工作中，我们使用20毫秒的窗口，50%的重叠，以及两个帧的look-ahead，导致总体算法延迟40毫秒。我们取32个ERB波段，f D F = 5 k H z f_{DF}= 5kHzf 
DF
​
 =5kHz，DF阶数N = 5 N=5N=5，look-ahead = 2帧。损失参数λ s p e c = l e 3 \lambda_{spec}=le3λ 
spec
​
 =le3和λ s p e c = 5 e 2 \lambda_{spec}=5e2λ 
spec
​
 =5e2的选择使两个损失的数量级相同。源代码和一个预先训练的DeepFilterNet2可以在https://github.com/Rikorose/DeepFilterNet获得。

## 结果
  我们使用Valentini语音库+需求测试集[8]来评估DeepFilterNet2的语音增强性能。因此，我们选择WB-PESQ [19]， STOI[20]和综合指标CSIG, CBAK, COVL[21]。表1显示了DeepFilterNet2与其他先进(SOTA)方法的比较结果。可以发现，DeepFilterNet2实现了sota级别的结果，同时需要最小的每秒乘法累积运算(MACS)。在DeepFilterNet(第2.5节)上，参数的数量略有增加，但该网络能够以两倍多的速度运行，并获得0.27的高PESQ评分。GaGNet[5]实现了类似的RTF，同时具有良好的SE性能。然而，它只在提供整个音频时运行得很快，由于它使用了大的时间卷积核，需要大的时间缓冲区。FRCRN[3]在大多数指标上都能获得最好的结果，但具有较高的计算复杂度，这在嵌入式设备上是不可实现的。



音频降噪
